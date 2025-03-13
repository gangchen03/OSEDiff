\import os
import sys
import time
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm  # For progress bar

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

from osediff import OSEDiff_inference_time
from ram.models.ram_lora import ram
from ram import inference_ram as inference

# Define transformations
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_validation_prompt(args, lq, model, weight_dtype, device='xla:0'):
    validation_prompt = ""
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype, device=device)
    captions = inference(lq_ram, model)
    
    # Assuming captions should be used to form the validation_prompt
    if captions:
        validation_prompt = captions[0]  # Adjust based on how captions are returned
    return validation_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference Speed Test")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='SD model path')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl', help='Path to OSEDiff model')
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    parser.add_argument('--ram_ft_path', type=str, default=None, help='Lora Path to RAM finetuned model')
    # Precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'bf16', 'fp32'], default="bf16", help='Mixed precision mode')
    # Merge LoRA
    parser.add_argument("--merge_and_unload_lora", action='store_true', help='Merge LoRA weights before inference')
    # Tile settings
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224, help='VAE decoder tiled size')
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024, help='VAE encoder tiled size')
    parser.add_argument("--latent_tiled_size", type=int, default=96, help='Latent tiled size')
    parser.add_argument("--latent_tiled_overlap", type=int, default=32, help='Latent tiled overlap')
    # Additional arguments
    parser.add_argument('--device', type=str, default='xla', choices=['xla', 'cpu'], help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument("--process_size", type=int, default=512, help='Size for processing')
    parser.add_argument('--inference_iterations', type=int, default=500, help='Number of inference iterations')
    parser.add_argument('--warmup_iterations', type=int, default=5, help='Number of warm-up iterations')
    parser.add_argument("--num_devices", type=int, default=4)
    
    return parser.parse_args()

def inference_in_parallel(index, args):
    # initialize the chip runtime
    device = xm.xla_device()
    # torch.arange(0, 100, device=device)
    world_size = xr.world_size()
    rank = xr.process_index()
    
    # Weight type
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # Initialize the model
    model = OSEDiff_inference_time(args)
    model.to(device, dtype=weight_dtype)
    model.eval()
    
    # Initialize RAM model
    DAPE = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit='swin_l')
    DAPE.eval()
    DAPE.to(device, dtype=weight_dtype)
    
    # Initialize timing variables
    total_time = 0.0
    batch_size = args.batch_size
    inference_iterations = args.inference_iterations
    warmup_iterations = args.warmup_iterations
    
    # Generate random tensors for inference
    # Pre-generate all tensors
    input_tensors = torch.randn((inference_iterations, batch_size, 3, args.process_size, args.process_size), device=device, dtype=weight_dtype)

    
    # Warm-up runs
    if rank == 0:
        print(f"Running {warmup_iterations} warm-up iterations...")
    for _ in range(warmup_iterations):
        lq = input_tensors[_].clone()
        validation_prompt = get_validation_prompt(args, lq, DAPE, weight_dtype, device=device)
        with torch.no_grad():
            lq_processed = lq * 2 - 1  # normalization
            output_image = model(lq_processed, prompt=validation_prompt)

    xm.rendezvous("warmup_done")
    
    if rank == 0:
        print(f"Starting inference for {inference_iterations} iterations...")
    # Inference runs with timing
    for idx in tqdm(range(inference_iterations), desc=f"Inference - Rank {rank}", disable=(rank != 0)):
        start_time = time.time()
        lq = input_tensors[idx].clone()
        validation_prompt = get_validation_prompt(args, lq, DAPE, weight_dtype, device=device)
        xm.mark_step()

        with torch.no_grad():
            lq_processed = lq * 2 - 1  # normalization
            output_image = model(lq_processed, prompt=validation_prompt)
        
        xm.mark_step()

        end_time = time.time()
        total_time += (end_time - start_time)

    xm.rendezvous("inference_done")
    
    total_time = xm.mesh_reduce('total_time_sum', total_time, sum)
    avg_time = total_time / (inference_iterations * world_size)
    if rank == 0:
        print(f'Average inference time per iteration: {avg_time:.4f} seconds.')


def main():
    args = parse_args()
    args.merge_and_unload_lora = True

    num_devices = args.num_devices
    print(f"Devices: {num_devices}")

    xmp.spawn(inference_in_parallel, args=(args,), nprocs=num_devices, start_method='fork')


if __name__ == "__main__":
    main()
