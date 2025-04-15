# osediff.py

# ... (other imports remain the same) ...
import os
import math
import sys
import time
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL # Use the modified AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel # Use the modified UNet
from peft import LoraConfig
import torch_xla.core.xla_model as xm
from my_utils.vaehook import VAEHook # Use the refactored VAEHook
# from my_utils.vaehook import perfcount # Keep if needed and defined

# ... (initialize_vae, initialize_unet remain the same) ...
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
            # Ensure the pattern exists and it's part of the encoder or specific conv layers
            if pattern in n and ("encoder" in n or 'quant_conv' in n):
                 # Exclude post_quant_conv explicitly if needed
                 if 'post_quant_conv' not in n:
                      l_target_modules_encoder.append(n.replace(".weight",""))
                      break # Move to next parameter once a pattern is matched

    # Remove duplicates if any pattern matched multiple times for the same layer
    l_target_modules_encoder = sorted(list(set(l_target_modules_encoder)))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    device = xm.xla_device()
    # Use provided path if available, otherwise use args path
    path = pretrained_model_name_or_path if pretrained_model_name_or_path else args.pretrained_model_name_or_path
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet")
    unet.requires_grad_(False)
    # unet.train() # training is not used here, remove it to reduce confusion
    unet.to(device)  # Move to TPU

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    # Refined grep patterns to be more specific
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", # Attention projections
              "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", # ResNet convs
              "proj_in", "proj_out", # Transformer block projections
              "ff.net.0.proj", "ff.net.2"] # Feed-forward layers

    for n, p in unet.named_parameters():
        # Skip biases and normalization layers
        if "bias" in n or "norm" in n or "time_emb_proj" in n: # Added time_emb_proj exclusion
            continue

        # Check if the parameter name ends with '.weight' and contains a pattern
        base_name = n.replace(".weight", "")
        matched = False
        for pattern in l_grep:
            # Check if the pattern is a suffix or part of the layer name
            if n.endswith(f"{pattern}.weight"):
                 target_list = None
                 if "down_blocks" in n or "conv_in" in n:
                      target_list = l_target_modules_encoder
                 elif "up_blocks" in n or "conv_out" in n:
                      target_list = l_target_modules_decoder
                 elif "mid_block" in n: # Added mid_block handling
                      target_list = l_modules_others
                 # else: # Parameter doesn't belong to known blocks

                 if target_list is not None:
                      target_list.append(base_name)
                      matched = True
                      break # Move to next parameter once matched

        # If not matched by specific patterns, maybe log or ignore
        # if not matched:
        #     print(f"Parameter not matched for LoRA: {n}")


    # Remove duplicates that might arise from overlapping patterns or structure
    l_target_modules_encoder = sorted(list(set(l_target_modules_encoder)))
    l_target_modules_decoder = sorted(list(set(l_target_modules_decoder)))
    l_modules_others = sorted(list(set(l_modules_others)))

    # Debug print counts
    # print(f"UNet LoRA targets: Encoder({len(l_target_modules_encoder)}), Decoder({len(l_target_modules_decoder)}), Others({len(l_modules_others)})")


    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


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
        # Pass return_lora_module_names=True to get the names
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args, return_lora_module_names=True)
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
        # Check if conv_in exists before setting requires_grad_
        if hasattr(self.unet, 'conv_in'):
            self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        device = xm.xla_device()
        prompt_embeds_list = []
        # Use model's device instead of re-fetching xm.xla_device() inside loop
        model_device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(model_device), # Use model's device
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        # Concatenate on the correct device
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):
        device = xm.xla_device() # Or use self.unet.device
        c_t = c_t.to(device)  # Move control tensor to TPU
        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"]) # Ensure neg_prompt exists in batch

        # Ensure VAE and UNet are on the correct device (should be done in init)
        # self.vae.to(device)
        # self.unet.to(device)

        # Use appropriate dtype, especially if using mixed precision
        input_dtype = next(self.vae.parameters()).dtype
        prompt_dtype = next(self.text_encoder.parameters()).dtype

        encoded_control = self.vae.encode(c_t.to(input_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred = self.unet(encoded_control.to(input_dtype),
                               self.timesteps,
                               encoder_hidden_states=prompt_embeds.to(prompt_dtype) # Match UNet's expected prompt dtype
                               ).sample

        # Ensure scheduler step uses consistent dtypes and device
        x_denoised = self.noise_scheduler.step(model_pred.to(encoded_control.dtype),
                                               self.timesteps,
                                               encoded_control,
                                               return_dict=True).prev_sample

        output_image = (self.vae.decode(x_denoised.to(input_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        # Ensure state dicts are moved to CPU before saving
        sd["state_dict_unet"] = {k: v.cpu() for k, v in self.unet.state_dict().items() if "lora" in k or ("conv_in" in k and hasattr(self.unet, 'conv_in'))}
        sd["state_dict_vae"] = {k: v.cpu() for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)


class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator): # accelerator might not be needed with XLA
        super().__init__()
        device = xm.xla_device()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        # Determine weight dtype based on args.mixed_precision
        weight_dtype = torch.float32
        # if accelerator.mixed_precision == "fp16": # Use args instead of accelerator
        if args.mixed_precision == "fp16":
            weight_dtype = torch.bfloat16 # Use bfloat16 on TPU
        # elif accelerator.mixed_precision == "bf16":
        #     weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        # Pass return_lora_module_names=True
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args, return_lora_module_names=True) # Pass args here

        self.text_encoder.to(device, dtype=weight_dtype)
        self.unet_fix.to(device, dtype=weight_dtype)
        self.unet_update.to(device) # LoRA weights usually stay float32, base model might be bf16
        self.vae.to(device) # VAE often kept in float32 or specific precision (e.g., fp16 fix)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

        # Move scheduler alphas to device
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)


    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):
        device = latents.device # Use device from input tensor
        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long() # timestep to TPU
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Ensure inputs to unet_update match its expected dtype
        unet_input_dtype = next(self.unet_update.parameters()).dtype
        prompt_dtype = prompt_embeds.dtype # Use provided prompt dtype

        noise_pred = self.unet_update(
            noisy_latents.to(unet_input_dtype),
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds.to(prompt_dtype), # Match expected prompt dtype
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        # Ensure alphas_cumprod is on the correct device
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        # Ensure timesteps are on the correct device
        timesteps = timesteps.to(sample.device)

        # Check if timesteps are within bounds
        if timesteps.max() >= len(alphas_cumprod) or timesteps.min() < 0:
             raise ValueError(f"Timesteps out of bounds: {timesteps.min()}-{timesteps.max()}, "
                              f"alphas_cumprod length: {len(alphas_cumprod)}")

        alpha_prod_t = alphas_cumprod[timesteps]
        # Expand dims for broadcasting
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / (alpha_prod_t ** (0.5) + 1e-6) # Add eps for stability
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        device = latents.device # Use device from input tensor
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=device).long() # timesteps to TPU
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Dtypes
        unet_update_dtype = next(self.unet_update.parameters()).dtype
        unet_fix_dtype = self.weight_dtype # From init
        prompt_dtype = prompt_embeds.dtype

        with torch.no_grad():
            # Prediction from the LoRA-updated UNet
            noise_pred_update = self.unet_update(
                noisy_latents.to(unet_update_dtype),
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.to(prompt_dtype), # Use original prompt embeds
            ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            # Prepare inputs for fixed UNet (classifier-free guidance)
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            guidance_prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            # Prediction from the fixed UNet
            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(unet_fix_dtype), # Use fixed unet dtype
                timestep=timesteps_input,
                encoder_hidden_states=guidance_prompt_embeds.to(unet_fix_dtype), # Match fixed unet dtype
            ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            # Ensure guidance calculation happens in float32 for precision
            noise_pred_guidance = noise_pred_uncond.float() + args.cfg_vsd * (noise_pred_text.float() - noise_pred_uncond.float())
            # Cast back if needed, but x0 calculation might benefit from float32
            # noise_pred_guidance = noise_pred_guidance.to(dtype=noisy_latents.dtype)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_guidance, noisy_latents, timesteps)

        # Ensure latents and predictions are float32 for loss calculation if needed
        latents_f = latents.float()
        x0_pred_fix_f = x0_pred_fix.float()
        x0_pred_update_f = x0_pred_update.float()

        # Calculate weighting factor (consider adding eps for stability)
        weighting_factor = torch.mean(torch.abs(latents_f - x0_pred_fix_f), dim=[1, 2, 3], keepdim=True) + 1e-6

        # Calculate gradient (proxy)
        grad = (x0_pred_update_f - x0_pred_fix_f) / weighting_factor

        # Calculate loss - compare latents to the "denoised" version using the gradient proxy
        # Ensure detach() is used correctly to prevent gradients flowing back through the target
        loss = F.mse_loss(latents_f, (latents_f - grad).detach())

        return loss


class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        device = xm.xla_device()

        self.args = args
        self.device = device # Use xm.xla_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=device) # Set timesteps on TPU
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        # Patch VAE forward methods *before* loading checkpoint or moving to device
        self._patch_vae_forward() # Call the patching method
        # Now initialize the VAEHook with actual tile sizes from args
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.bfloat16 # Use bfloat16 on TPU

        # Load checkpoint onto CPU first
        osediff = torch.load(args.osediff_path, map_location = torch.device('cpu'))
        # Load state dicts from checkpoint (will be moved to TPU within load_ckpt)
        self.load_ckpt(osediff)

        # merge lora AFTER loading checkpoint but BEFORE moving base model to device
        if self.args.merge_and_unload_lora:
            print(f'===> MERGING LoRA <===')
            # Ensure merge_and_unload is called correctly
            if hasattr(self.vae, 'merge_and_unload'):
                 self.vae = self.vae.merge_and_unload()
            else:
                 print("VAE does not have merge_and_unload (using stock Diffusers VAE?)")

            if hasattr(self.unet, 'merge_and_unload'):
                 self.unet = self.unet.merge_and_unload()
            else:
                 print("UNet does not have merge_and_unload (using stock Diffusers UNet?)")
            print(f'===> LoRA MERGED <===')
            # Force garbage collection after merge might help release LoRA weights memory
            gc.collect()


        # Move models to TPU *after* potential merge
        self.unet.to(device, dtype=self.weight_dtype)
        # VAE precision: often kept at float32 or fp16-fix. Use float32 unless fp16-fix VAE is used.
        self.vae.to(device) # Keep VAE in float32 by default for stability
        self.text_encoder.to(device, dtype=self.weight_dtype)

        self.timesteps = torch.tensor([999], device=device).long() # create timesteps on TPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device) # move tensor to TPU

    def _patch_vae_forward(self):
        """Adds original_forward methods to VAE encoder/decoder if they don't exist."""
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)


    def load_ckpt(self, model):
        device = self.device # Use the class's device attribute

        # --- Load UNet LoRA ---
        # Check if LoRA modules exist in the checkpoint
        if "unet_lora_encoder_modules" not in model or \
           "unet_lora_decoder_modules" not in model or \
           "unet_lora_others_modules" not in model or \
           "rank_unet" not in model:
            print("[LoRA Load Warning] UNet LoRA config not found in checkpoint. Skipping UNet LoRA loading.")
        else:
            lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
            lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
            lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])

            # Add adapters to the UNet (which is currently on CPU)
            self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

            # Load state dict for LoRA weights (and conv_in if present)
            if "state_dict_unet" in model:
                 unet_state_dict = model["state_dict_unet"]
                 # Filter state dict for keys that actually exist in the current unet model
                 current_unet_keys = set(self.unet.state_dict().keys())
                 filtered_unet_state_dict = {
                     k: v for k, v in unet_state_dict.items()
                     if k in current_unet_keys and ("lora" in k or "conv_in" in k)
                 }
                 missing_keys, unexpected_keys = self.unet.load_state_dict(filtered_unet_state_dict, strict=False)
                 if missing_keys: print(f"[UNet LoRA Load] Missing keys: {missing_keys}")
                 if unexpected_keys: print(f"[UNet LoRA Load] Unexpected keys: {unexpected_keys}")
            else:
                 print("[LoRA Load Warning] 'state_dict_unet' not found in checkpoint.")

            # Set adapters - this assumes PEFT's handling
            self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
            print("UNet LoRA loaded.")


        # --- Load VAE LoRA ---
        if "vae_lora_encoder_modules" not in model or "rank_vae" not in model:
             print("[LoRA Load Warning] VAE LoRA config not found in checkpoint. Skipping VAE LoRA loading.")
        else:
            vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
            # Add adapter to VAE (currently on CPU)
            self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")

            # Load state dict for VAE LoRA weights
            if "state_dict_vae" in model:
                 vae_state_dict = model["state_dict_vae"]
                 # Filter state dict for keys that actually exist in the current vae model
                 current_vae_keys = set(self.vae.state_dict().keys())
                 filtered_vae_state_dict = {
                     k: v for k, v in vae_state_dict.items()
                     if k in current_vae_keys and "lora" in k
                 }
                 missing_keys, unexpected_keys = self.vae.load_state_dict(filtered_vae_state_dict, strict=False)
                 if missing_keys: print(f"[VAE LoRA Load] Missing keys: {missing_keys}")
                 if unexpected_keys: print(f"[VAE LoRA Load] Unexpected keys: {unexpected_keys}")
            else:
                 print("[LoRA Load Warning] 'state_dict_vae' not found in checkpoint.")

            # Set VAE adapter
            self.vae.set_adapter(['default_encoder'])
            print("VAE LoRA loaded.")


    def encode_prompt(self, prompt_batch):
        # Use model's device
        model_device = self.device # Or next(self.text_encoder.parameters()).device
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(model_device), # Move IDs to device
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        # Concatenate on the correct device
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount # Keep if defined and useful
    @torch.no_grad()
    def forward(self, lq, prompt):
        device = self.device
        #import time # Keep for internal timing if needed
        #time0 = time.time()

        # Ensure input is on the correct device and dtype
        lq = lq.to(device=device, dtype=self.weight_dtype) # Use model's weight dtype

        #time_prompt_start = time.time()
        prompt_embeds = self.encode_prompt([prompt])
        #time_prompt_end = time.time()
        #print(f"encode prompt time is: {time_prompt_end - time_prompt_start:.4f}s")

        # --- VAE Encode ---
        #time_vae_enc_start = time.time()
        # VAE encode expects float32 usually, or fp16 if using fp16-fix VAE
        vae_input_dtype = next(self.vae.parameters()).dtype # Get VAE's actual dtype
        lq_latent_dist = self.vae.encode(lq.to(vae_input_dtype)) # Pass input in VAE's dtype
        lq_latent = lq_latent_dist.latent_dist.sample() * self.vae.config.scaling_factor
        #time_vae_enc_end = time.time()
        #print(f"vae encode time is: {time_vae_enc_end - time_vae_enc_start:.4f}s")

        # Ensure latent is in the UNet's expected dtype
        lq_latent = lq_latent.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype) # Also ensure prompt matches

        # --- UNet Tiling Logic ---
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)

        # Ensure tile size and overlap are valid
        tile_size = max(1, tile_size)
        tile_overlap = max(0, min(tile_overlap, tile_size - 1)) # Overlap must be less than tile size

        model_pred = None # Initialize model_pred

        # Check if tiling is needed
        if h <= tile_size and w <= tile_size:
            print(f"[Tiled Latent]: Input {h}x{w} <= tile size {tile_size}. No tiling needed.")
            #time_unet_start = time.time()
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            #time_unet_end = time.time()
            #print(f"unet (no tile) time is: {time_unet_end - time_unet_start:.4f}s")
        else:
            print(f"[Tiled Latent]: Input {h}x{w} > tile size {tile_size}. Tiling...")
            # --- Gaussian Weights ---
            time_gauss_start = time.time()
            # Calculate weights on device using torch
            tile_weights = self._gaussian_weights(tile_size, tile_size, lq_latent.shape[0]) # Pass batch size
            tile_weights = tile_weights.to(dtype=lq_latent.dtype, device=device) # Ensure dtype/device match
            #time_gauss_end = time.time()
            #print(f"gaussian weights time is: {time_gauss_end - time_gauss_start:.4f}s")

            # --- Tiling Calculation ---
            stride = tile_size - tile_overlap
            stride = max(1, stride) # Ensure stride is positive

            # Calculate number of tiles needed
            num_tiles_h = math.ceil(max(1, h - tile_overlap) / stride)
            num_tiles_w = math.ceil(max(1, w - tile_overlap) / stride)
            print(f"Grid size: {num_tiles_h} x {num_tiles_w}")

            noise_preds = torch.zeros_like(lq_latent)
            contributors = torch.zeros_like(lq_latent)

            unet_total_time = 0

            for row in range(num_tiles_h):
                for col in range(num_tiles_w):
                    # Calculate tile coordinates
                    y_start = row * stride
                    x_start = col * stride
                    y_end = min(y_start + tile_size, h)
                    x_end = min(x_start + tile_size, w)
                    # Adjust start if end goes out of bounds (for last tiles)
                    y_start = max(0, y_end - tile_size)
                    x_start = max(0, x_end - tile_size)

                    # Extract tile
                    input_tile = lq_latent[:, :, y_start:y_end, x_start:x_end]

                    # Pad if tile is smaller than tile_size (boundary cases)
                    pad_h = tile_size - input_tile.shape[2]
                    pad_w = tile_size - input_tile.shape[3]
                    if pad_h > 0 or pad_w > 0:
                         # Pad using replication or reflection (check which works better)
                         input_tile = F.pad(input_tile, (0, pad_w, 0, pad_h), mode='replicate')


                    # Predict noise for the tile
                    #time_unet_tile_start = time.time()
                    # Ensure prompt embeds batch dim matches tile batch dim if needed (usually 1 for inference)
                    current_prompt_embeds = prompt_embeds[:input_tile.shape[0]] # Match batch size
                    tile_pred = self.unet(input_tile, self.timesteps, encoder_hidden_states=current_prompt_embeds).sample
                    #time_unet_tile_end = time.time()
                    #unet_total_time += (time_unet_tile_end - time_unet_tile_start)

                    # Crop prediction back if padding was applied
                    if pad_h > 0 or pad_w > 0:
                         tile_pred = tile_pred[:, :, :tile_size-pad_h, :tile_size-pad_w]

                    # Crop weights to match the actual tile prediction size
                    current_weights = tile_weights[:, :, :tile_pred.shape[2], :tile_pred.shape[3]]

                    # Add weighted prediction to the corresponding region
                    noise_preds[:, :, y_start:y_end, x_start:x_end] += tile_pred * current_weights
                    contributors[:, :, y_start:y_end, x_start:x_end] += current_weights

            #print(f"unet (tiled) total time is: {unet_total_time:.4f}s")

            # Average overlapping areas
            noise_preds /= (contributors + 1e-6) # Add epsilon for stability
            model_pred = noise_preds
            # --- End Tiling ---

        # --- Diffusion Step ---
        #time_diff_start = time.time()
        # Ensure dtypes match for scheduler step
        model_pred_sched = model_pred.to(lq_latent.dtype)
        x_denoised = self.noise_scheduler.step(model_pred_sched, self.timesteps, lq_latent, return_dict=True).prev_sample
        #time_diff_end = time.time()
        #print(f"diffusion step time is: {time_diff_end - time_diff_start:.4f}s")

        # --- VAE Decode ---
        #time_vae_dec_start = time.time()
        # Pass input in VAE's dtype
        vae_input_denoised = x_denoised.to(vae_input_dtype) / self.vae.config.scaling_factor
        output_image = (self.vae.decode(vae_input_denoised).sample).clamp(-1, 1)
        #time_vae_dec_end = time.time()
        #print(f"vae decode time is: {time_vae_dec_end - time_vae_dec_start:.4f}s")

        # --- Final Output ---
        # Keep output on device, move to CPU only when saving in test script
        # print(output_image[0]) # Avoid printing tensor data for performance
        #print(f"print time is: {time.time() - time9}")
        #print(f"Total forward pass time: {time.time() - time0:.4f}s")
        return output_image # Return single image from batch

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False, # Keep fast mode flags if VAEHook uses them
            fast_encoder = False,
            color_fix = False):
        # device = xm.xla_device() # Not needed here, VAEHook uses self.device

        # Ensure original_forward is patched correctly (called in __init__)
        # self._patch_vae_forward() # Already called in __init__

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        # Update VAE forward methods with VAEHook instances
        # Remove to_gpu argument
        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix)
        print("VAE forward methods patched with VAEHook for tiling.")


    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions (using torch)."""
        device = self.device # Use class device

        # Create coordinate grids
        y_coords = torch.linspace(- (tile_height - 1) / 2, (tile_height - 1) / 2, tile_height, device=device)
        x_coords = torch.linspace(- (tile_width - 1) / 2, (tile_width - 1) / 2, tile_width, device=device)

        # Use a simpler variance calculation (adjust sigma as needed)
        sigma = min(tile_width, tile_height) / 4.0 # Example sigma, adjust for desired blend width
        var = sigma ** 2

        # Calculate Gaussian values
        y_probs = torch.exp(-y_coords**2 / (2 * var))
        x_probs = torch.exp(-x_coords**2 / (2 * var))

        # Outer product to create 2D weights
        weights_2d = torch.outer(y_probs, x_probs)

        # Normalize to have max value of 1 (optional, but common)
        weights_2d = weights_2d / weights_2d.max()

        # Expand dimensions for batch and channels
        # Get channel count from UNet config
        num_channels = self.unet.config.in_channels
        weights_4d = weights_2d.unsqueeze(0).unsqueeze(0) # Shape (1, 1, H, W)

        # Repeat for batch and channels
        weights_final = weights_4d.repeat(nbatches, num_channels, 1, 1)

        return weights_final


class OSEDiff_inference_time(torch.nn.Module):
    # This class seems very similar to OSEDiff_test, maybe consolidate?
    # Assuming it's for benchmarking without tiling?
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

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.bfloat16 # Use bfloat16 on TPU

        # Load checkpoint onto CPU first
        osediff = torch.load(args.osediff_path, map_location = torch.device('cpu'))
        # Load state dicts (will be moved to TPU within load_ckpt)
        self.load_ckpt(osediff)

        # merge lora AFTER loading checkpoint but BEFORE moving base model to device
        if self.args.merge_and_unload_lora:
            print(f'===> MERGING LoRA (inference_time) <===')
            if hasattr(self.vae, 'merge_and_unload'):
                 self.vae = self.vae.merge_and_unload()
            if hasattr(self.unet, 'merge_and_unload'):
                 self.unet = self.unet.merge_and_unload()
            print(f'===> LoRA MERGED (inference_time) <===')
            gc.collect()

        # Move models to TPU *after* potential merge
        self.unet.to(device, dtype=self.weight_dtype)
        self.vae.to(device) # Keep VAE float32 default
        self.text_encoder.to(device, dtype=self.weight_dtype)

        self.timesteps = torch.tensor([999], device=device).long() # create timesteps on TPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device) # move tensor to TPU

    def load_ckpt(self, model):
        # --- Identical to OSEDiff_test.load_ckpt ---
        # Consider making this a shared utility function if classes remain separate
        device = self.device

        # --- Load UNet LoRA ---
        if "unet_lora_encoder_modules" not in model or \
           "unet_lora_decoder_modules" not in model or \
           "unet_lora_others_modules" not in model or \
           "rank_unet" not in model:
            print("[LoRA Load Warning] UNet LoRA config not found in checkpoint. Skipping UNet LoRA loading.")
        else:
            lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
            lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
            lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
            self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
            self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
            self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
            if "state_dict_unet" in model:
                 unet_state_dict = model["state_dict_unet"]
                 current_unet_keys = set(self.unet.state_dict().keys())
                 filtered_unet_state_dict = {k: v for k, v in unet_state_dict.items() if k in current_unet_keys and ("lora" in k or "conv_in" in k)}
                 missing_keys, unexpected_keys = self.unet.load_state_dict(filtered_unet_state_dict, strict=False)
                 if missing_keys: print(f"[UNet LoRA Load] Missing keys: {missing_keys}")
                 if unexpected_keys: print(f"[UNet LoRA Load] Unexpected keys: {unexpected_keys}")
            else: print("[LoRA Load Warning] 'state_dict_unet' not found.")
            self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
            print("UNet LoRA loaded (inference_time).")

        # --- Load VAE LoRA ---
        if "vae_lora_encoder_modules" not in model or "rank_vae" not in model:
             print("[LoRA Load Warning] VAE LoRA config not found in checkpoint. Skipping VAE LoRA loading.")
        else:
            vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
            self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
            if "state_dict_vae" in model:
                 vae_state_dict = model["state_dict_vae"]
                 current_vae_keys = set(self.vae.state_dict().keys())
                 filtered_vae_state_dict = {k: v for k, v in vae_state_dict.items() if k in current_vae_keys and "lora" in k}
                 missing_keys, unexpected_keys = self.vae.load_state_dict(filtered_vae_state_dict, strict=False)
                 if missing_keys: print(f"[VAE LoRA Load] Missing keys: {missing_keys}")
                 if unexpected_keys: print(f"[VAE LoRA Load] Unexpected keys: {unexpected_keys}")
            else: print("[LoRA Load Warning] 'state_dict_vae' not found.")
            self.vae.set_adapter(['default_encoder'])
            print("VAE LoRA loaded (inference_time).")


    def encode_prompt(self, prompt_batch):
        # --- Identical to OSEDiff_test.encode_prompt ---
        model_device = self.device
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(model_device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt):
        # --- Non-tiled forward pass ---
        device = self.device

        # Ensure input is on the correct device and dtype
        lq = lq.to(device=device, dtype=self.weight_dtype)

        prompt_embeds = self.encode_prompt([prompt]).to(self.weight_dtype) # Ensure dtype match

        # VAE encode
        vae_input_dtype = next(self.vae.parameters()).dtype
        lq_latent_dist = self.vae.encode(lq.to(vae_input_dtype))
        lq_latent = lq_latent_dist.latent_dist.sample() * self.vae.config.scaling_factor
        lq_latent = lq_latent.to(self.weight_dtype) # Match UNet dtype

        # UNet prediction
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample

        # Diffusion step
        model_pred_sched = model_pred.to(lq_latent.dtype)
        x_denoised = self.noise_scheduler.step(model_pred_sched, self.timesteps, lq_latent, return_dict=True).prev_sample

        # VAE decode
        vae_input_denoised = x_denoised.to(vae_input_dtype) / self.vae.config.scaling_factor
        output_image = (self.vae.decode(vae_input_denoised).sample).clamp(-1, 1)

        return output_image # Return full batch tensor
