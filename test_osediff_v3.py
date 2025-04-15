# This version fixes the latent size for XLA compilation cache
# It also has a warm-up phase with a dummy lq to trigger the pre-compile
# Future work: need to adjust the shape size to dynamically cache the input

import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import time
from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import traceback # Import traceback for better error logging


import gc # Import garbage collector
from ram.models.ram_lora import ram
from ram import inference_ram as inference
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

# Import the TENSOR version of the color fix function(s)
from my_utils.wavelet_color_fix_v2 import adain_color_fix_tensor, wavelet_color_fix_tensor # Adjust path if needed

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, model, device='xla:0'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    # weight type
    weight_dtype = torch.bfloat16 # Assuming bfloat16 for TPU
    lq = lq.to(dtype = weight_dtype)
    # --- RAM Model Captioning (Optional, currently commented out) ---
    # if model is not None: # Check if DAPE model was loaded
    #     try:
    #         # Ensure lq is in the range [0, 1] for RAM transforms if needed
    #         lq_for_ram = lq.clamp(0, 1) # Clamp if lq might be [-1, 1] here
    #         lq_ram = ram_transforms(lq_for_ram).to(dtype=model.dtype, device=device) # Match DAPE's dtype
    #         with torch.no_grad():
    #              captions = inference(lq_ram, model)
    #         if captions:
    #              validation_prompt = f"{captions[0]}, {args.prompt}" # Combine caption and base prompt
    #         else:
    #              validation_prompt = args.prompt # Fallback to base prompt
    #     except Exception as e:
    #         print(f"  Warning: Error during caption generation: {e}")
    #         validation_prompt = args.prompt # Fallback on error
    # else:
    #     validation_prompt = args.prompt # Use base prompt if DAPE is None
    # -----------------------------------------------------------------
    # For now, just use an empty prompt as per the original logic
    validation_prompt = ""

    return validation_prompt, lq

def inference_in_parallel(index, all_image_names):
    # initialize the chip runtime
    device = xm.xla_device()
    world_size = xr.world_size()
    rank = xr.process_index()

    # --- Define Static Target Shape ---
    # Ensure target dimensions are multiples of 8
    target_w = args.process_size - (args.process_size % 8)
    target_h = args.process_size - (args.process_size % 8)
    print(f"Rank {rank}: Using static input shape: {target_w}x{target_h}")
    # ---------------------------------

    # initialize the model (ensure it's moved to device inside OSEDiff_test)
    model = OSEDiff_test(args)
    # model.to(dtype=torch.bfloat16) # Dtype handling should be inside OSEDiff_test

    # get ram model (ensure it's moved to device)
    DAPE = None
    if args.ram_path and args.ram_ft_path: # Only load if paths are provided
        DAPE = ram(pretrained=args.ram_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
        DAPE.eval()
        DAPE.to(device) # Move DAPE model to TPU
        # Set DAPE dtype
        weight_dtype_ram = torch.bfloat16 # Or match model.weight_dtype
        DAPE = DAPE.to(dtype=weight_dtype_ram)
    else:
        print("RAM model paths not provided, skipping caption generation.")


    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)

    # --- XLA Warm-up Step ---
    print(f"Rank {rank}: Starting XLA warm-up...")
    warmup_start_time = time.time()
    try:
        # Create a dummy tensor with the static shape and model's dtype
        # Use the model's expected input dtype (often self.weight_dtype in OSEDiff_test)
        # The input range should be [-1, 1] for the model
        dummy_input = torch.randn(1, 3, target_h, target_w, # B, C, H, W
                                  dtype=model.weight_dtype,
                                  device=device)
        dummy_prompt = "warmup"

        with torch.no_grad():
            _ = model(dummy_input, prompt=dummy_prompt) # Run one forward pass
            # Optionally, run color fix warmup if used
            if args.align_method != 'nofix':
                 dummy_source = torch.randn_like(dummy_input)
                 if args.align_method == 'adain' and 'adain_color_fix_tensor' in globals():
                      _ = adain_color_fix_tensor(dummy_input, dummy_source)
                 elif args.align_method == 'wavelet' and 'wavelet_color_fix_tensor' in globals():
                      _ = wavelet_color_fix_tensor(dummy_input, dummy_source)

        xm.mark_step() # Ensure compilation finishes
        warmup_end_time = time.time()
        print(f"Rank {rank}: XLA warm-up finished in {warmup_end_time - warmup_start_time:.4f}s")
    except Exception as e:
        print(f"Rank {rank}: Error during XLA warm-up: {e}")
        # Decide if you want to continue or exit if warmup fails
        # sys.exit(1) # Example: exit if warmup fails
    # -------------------------


    # splitted image_names
    num_images_per_process = len(all_image_names) // world_size + (1 if len(all_image_names) % world_size > rank else 0)
    start_index = sum(len(all_image_names) // world_size + (1 if len(all_image_names) % world_size > i else 0) for i in range(rank))
    end_index = start_index + num_images_per_process

    my_image_names = all_image_names[start_index:end_index] # slice for this process

    print(f"Rank {rank}/{world_size}: Processing {len(my_image_names)} images from index {start_index} to {end_index-1}")


    for image_name in my_image_names:
        print("="*18*world_size) # Make separator width dynamic
        print(f"Rank {rank}: Processing {image_name}")
        input_image_pil = Image.open(image_name).convert('RGB') # Keep original PIL for resize ref
        ori_width, ori_height = input_image_pil.size

        # --- Input Image Preprocessing ---
        rscale = args.upscale
        resize_flag = False
        processed_image_pil = input_image_pil # Start with original

        # 1. Initial resize for small images (if needed)
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            small_target_w, small_target_h = int(scale*ori_width), int(scale*ori_height)
            print(f"  Resizing small input to {small_target_w}x{small_target_h}")
            processed_image_pil = processed_image_pil.resize((small_target_w, small_target_h), Image.LANCZOS)
            resize_flag = True # Flag that original was small

        # 2. Upscale before passing to model
        upscaled_w, upscaled_h = processed_image_pil.size[0]*rscale, processed_image_pil.size[1]*rscale
        processed_image_pil = processed_image_pil.resize((upscaled_w, upscaled_h), Image.LANCZOS)

        # 3. Resize to fit within static target shape (maintaining aspect ratio)
        current_w, current_h = processed_image_pil.size
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        scale = min(scale_w, scale_h) # Scale factor to fit inside target
        fit_w = int(current_w * scale)
        fit_h = int(current_h * scale)
        # Ensure fit dimensions are multiples of 8 (optional but good practice)
        fit_w = fit_w - (fit_w % 8)
        fit_h = fit_h - (fit_h % 8)

        if fit_w != current_w or fit_h != current_h:
            print(f"  Resizing to fit target: {fit_w}x{fit_h}")
            processed_image_pil = processed_image_pil.resize((fit_w, fit_h), Image.LANCZOS)
        else:
            print(f"  Image already fits target size: {fit_w}x{fit_h}")

        # 4. Pad to static target shape
        padded_image_pil = Image.new('RGB', (target_w, target_h), (0, 0, 0)) # Black background
        paste_x = (target_w - fit_w) // 2
        paste_y = (target_h - fit_h) // 2
        padded_image_pil.paste(processed_image_pil, (paste_x, paste_y))
        print(f"  Padded to static size: {target_w}x{target_h} (content at {paste_x},{paste_y})")
        # ---------------------------------

        bname = os.path.basename(image_name)

        # --- Get Caption & Prepare Input Tensor 'lq' ---
        # Pass the *padded* PIL image to get_validation_prompt
        validation_prompt, lq = get_validation_prompt(args, padded_image_pil, DAPE, device=device)
        # lq is now a TPU tensor with static shape [B, C, target_h, target_w], range [0, 1]

        # Scale lq to [-1, 1] for the model input
        lq = lq * 2.0 - 1.0

        if args.save_prompts:
            txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
            with open(txt_save_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)
                # f.close() # 'with open' handles closing

        print(f"  Tag: {validation_prompt}".encode('utf-8')[:300])

        # --- Prepare Source Tensor for Color Fix (ONCE per image) ---
        time_source_prep_start = time.time()
        # Use the *padded PIL image*
        source_tensor = tensor_transforms(padded_image_pil).unsqueeze(0).to(device) # Range [0, 1]
        # Scale source to match model output range [-1, 1] and dtype
        source_tensor_scaled = (source_tensor.to(model.weight_dtype) * 2.0 - 1.0)
        time_source_prep_end = time.time()
        print(f"  Source tensor prep time: {time_source_prep_end - time_source_prep_start:.4f}s")


        # --- Model Inference ---
        with torch.no_grad():
            time0 = time.time() # Start timing just before model call
            # Input lq has static shape [B, C, target_h, target_w]
            output_image = model(lq, prompt=validation_prompt)
            # Output is TPU tensor, shape [B, C, target_h, target_w], range [-1, 1]
            xm.mark_step() # Ensure model forward completes before timing/next step
            time1 = time.time()
            print(f"  Model inference time: {time1 - time0:.4f}s")

            # --- Color Fix on TPU ---
            time_colorfix_start = time.time()
            output_image_colorfixed = output_image # Default if no fix needed

            if args.align_method == 'adain':
                 if 'adain_color_fix_tensor' in globals():
                      print("  Applying AdaIN color fix on TPU...")
                      # Both input and source have static shape
                      output_image_colorfixed = adain_color_fix_tensor(output_image, source_tensor_scaled)
                      xm.mark_step() # Mark step after AdaIN
                 else:
                      print("  AdaIN tensor function not found, skipping color fix.")

            elif args.align_method == 'wavelet':
                 if 'wavelet_color_fix_tensor' in globals():
                      print("  Applying Wavelet color fix on TPU...")
                      # Both input and source have static shape
                      output_image_colorfixed = wavelet_color_fix_tensor(output_image, source_tensor_scaled)
                      xm.mark_step() # Mark step after Wavelet
                 else:
                      print("  Wavelet tensor function not found, skipping color fix.")

            time_colorfix_end = time.time()
            if args.align_method != 'nofix':
                 print(f"  Color fix ({args.align_method}) on TPU time: {time_colorfix_end - time_colorfix_start:.4f}s")


            # --- Post-Processing (CPU Transfer, Crop, PIL Conversion, Save) ---
            # Scale the *final* tensor back to [0, 1] on TPU
            time_final_transfer_start = time.time()
            # Use float() for scaling calculation, then clamp
            output_im_processed_final = (output_image_colorfixed.float() * 0.5 + 0.5).clamp(0, 1)

            # Transfer final result to CPU (still padded)
            output_im_cpu_padded = output_im_processed_final.to('cpu')
            time_final_transfer_end = time.time()
            print(f"  Final TPU->CPU transfer time: {time_final_transfer_end - time_final_transfer_start:.4f}s")

            # --- Crop the output tensor on CPU ---
            time_crop_start = time.time()
            # Crop region corresponds to where the content was pasted
            output_im_cpu_cropped = output_im_cpu_padded[:, :, paste_y:paste_y+fit_h, paste_x:paste_x+fit_w]
            time_crop_end = time.time()
            print(f"  CPU tensor crop time: {time_crop_end - time_crop_start:.4f}s")
            # ------------------------------------

            # Convert *cropped* tensor to PIL
            time_pil_conv_start = time.time()
            # ToPILImage expects CxHxW, range [0, 1]
            output_pil = transforms.ToPILImage()(output_im_cpu_cropped.squeeze(0)) # Remove batch dim
            time_pil_conv_end = time.time()
            print(f"  ToPILImage time: {time_pil_conv_end - time_pil_conv_start:.4f}s")

            # Resize (PIL on CPU) - Use original PIL dimensions if resize_flag is set
            # This resize operates on the *cropped* PIL image
            time_resize_start = time.time()
            if resize_flag:
                # Calculate the target size based on the *original* small image dimensions
                final_target_w = int(args.upscale * ori_width)
                final_target_h = int(args.upscale * ori_height)
                print(f"  Resizing output back to original scaled size: {final_target_w}x{final_target_h}")
                output_pil = output_pil.resize((final_target_w, final_target_h), Image.LANCZOS)
            time_resize_end = time.time()
            if resize_flag:
                print(f"  PIL resize time: {time_resize_end - time_resize_start:.4f}s")

            # Save (CPU I/O)
            time_save_start = time.time()
            save_path = os.path.join(args.output_dir, bname)
            output_pil.save(save_path)
            time_save_end = time.time()
            print(f"  Image save time: {time_save_end - time_save_start:.4f}s")
            print(f"  Saved to: {save_path}")

        # Total time for this image (model + post-processing)
        total_image_time = time.time() - time0 # time0 starts just before the real inference call
        print(f"  Total time for image {bname}: {total_image_time:.4f}s")
        # Optional: Force garbage collection between images if memory is tight
        # gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16") # Note: 'fp16' likely means bfloat16 on TPU
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--num_devices", type=int, default=4)

    args = parser.parse_args()

    # Validate mixed_precision for TPU
    if args.mixed_precision == "fp16":
        print("Warning: Using 'fp16' for mixed_precision on TPU. This typically maps to bfloat16.")
        # PyTorch/XLA generally uses bfloat16 when fp16 is requested.

    num_devices=args.num_devices
    print(f"Devices: {num_devices}")

    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    xmp.spawn(inference_in_parallel, args=(image_names,), nprocs=num_devices, start_method='fork')
