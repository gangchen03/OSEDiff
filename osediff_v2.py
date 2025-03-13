import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
import torch_xla.core.xla_model as xm
from my_utils.vaehook import VAEHook, perfcount

def initialize_vae(args):
    device = xm.xla_device()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    # vae.train() # training is not used here, remove it to reduce confusion
    vae.to(device)  # Move to TPU

    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    device = xm.xla_device()
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    # unet.train() # training is not used here, remove it to reduce confusion
    unet.to(device)  # Move to TPU

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        device = xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device) # move model to device
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=device)  # Set timesteps on TPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device) # move tensor to device
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        # self.unet.to("cuda") # unet already initialize in initialize_unet function
        # self.vae.to("cuda")  # vae already initialize in initialize_vae function
        self.timesteps = torch.tensor([999], device=device).long() # set timesteps on TPU
        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        device = xm.xla_device()
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):
        device = xm.xla_device()
        c_t = c_t.to(device)  # Move control tensor to TPU
        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32)).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)

class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        device = xm.xla_device()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)

        self.text_encoder.to(device, dtype=weight_dtype)
        self.unet_fix.to(device, dtype=weight_dtype)
        self.unet_update.to(device)
        self.vae.to(device)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):
        device = xm.xla_device()
        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long() # timestep to TPU
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        device = xm.xla_device()
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=device).long() # timesteps to TPU
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss

class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        device = xm.xla_device()

        self.args = args
        #self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device # Use xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=device) # Set timesteps on TPU
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.bfloat16

        # Forcefully cast all layers to the correct dtype
        for module in self.vae.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.to(dtype=self.weight_dtype)
        self.vae.to(dtype=self.weight_dtype)

        osediff = torch.load(args.osediff_path, map_location = torch.device('cpu')) # Model loaded from CPU, then moved to TPU in load_ckpt
        
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to(device, dtype=self.weight_dtype) # All models loaded into the TPU
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder.to(device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device=device).long() # create timesteps on TPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device) # move tensor to TPU

    def load_ckpt(self, model):
        device = xm.xla_device() # get device here to make sure device is available.
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n].to(device)) # Model loaded into the TPU
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n].to(device)) # Model loaded into the TPU
        self.vae.set_adapter(['default_encoder'])

    def encode_prompt(self, prompt_batch):
        device = xm.xla_device()
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt):
        device = xm.xla_device()
        import time
        time0 = time.time()
        lq = lq.to(device)  # move lq to device
        prompt_embeds = self.encode_prompt([prompt])
        time1 = time.time()
        print(f"encode prompt time is: {time1 - time0}")
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        print(f"vae encode time is: {time.time() - time1}")
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            time2 = time.time()
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            print(f"tiny pic unet time is: {time.time() - time2}")
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            time2 = time.time()
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            time3 = time.time()
            print(f"gaussian weights time is: {time3 - time2}")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        time4 = time.time()
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        print(f"need tile unet time is: {time.time() - time4}")
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred
        time7 = time.time()
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        time8 = time.time()
        print(f"one diffuse time is: {time8- time7}")
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        time9 = time.time()
        print(f"vae decode time is: {time9 - time8}")
        print(output_image[0])
        print(f"print time is: {time.time() - time9}")
        return output_image[0]

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        device = xm.xla_device()
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=False) #Set the argument of vae_to_gpu to False, since the to_gpu operation is unnecessary in TPU programming
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=False) #Set the argument of vae_to_gpu to False, since the to_gpu operation is unnecessary in TPU programming

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        device = xm.xla_device()
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=device), (nbatches, self.unet.config.in_channels, 1, 1))


class OSEDiff_inference_time(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        device = xm.xla_device()

        self.args = args
        self.device = device # Use xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=device) # set to TPU
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        #self.weight_dtype = torch.float32
        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.bfloat16

        # Forcefully cast all layers to the correct dtype
        for module in self.vae.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.to(dtype=self.weight_dtype)
        self.vae.to(dtype=self.weight_dtype)

        osediff = torch.load(args.osediff_path, map_location = torch.device('cpu'))  # Model loaded from CPU, then moved to TPU in load_ckpt
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to(device, dtype=self.weight_dtype) # All models loaded into the TPU
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder.to(device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device=device).long() # create timesteps on TPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device) # move tensor to TPU

    def load_ckpt(self, model):
        device = xm.xla_device() # get device here to make sure device is available.

        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n].to(device))  # Move to TPU

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n].to(device))  # Move to TPU
        self.vae.set_adapter(['default_encoder'])

    def encode_prompt(self, prompt_batch):
        device = xm.xla_device()
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                text_input_ids = text_input_ids.to(device) # Move to TPU

                prompt_embeds = self.text_encoder(text_input_ids)[0]
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt):
        device = xm.xla_device()

        lq = lq.to(device, dtype=self.weight_dtype) # Move lq to device and cast to correct dtype.

        prompt_embeds = self.encode_prompt([prompt]) # prompt embeds already on device

        # lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        # for test_inference_time call
        lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image
