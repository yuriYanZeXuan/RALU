import argparse
import os
import torch
from PIL import Image

from pipeline_flux_RALU import FluxPipeline_RALU
from pipeline_qwen_image_ralu import QwenImagePipelineRALU

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    if args.model_type == 'flux':
        # --- FLUX Model Logic ---
        pipe = FluxPipeline_RALU.from_pretrained(
            args.flux_model_path,
            torch_dtype=torch_dtype
        ).to(device)

        # Set RALU parameters for FLUX
        pipe.set_params(
            use_RALU_default=args.use_RALU_default,
            level=args.level,
            N=args.N,
            e=args.e,
            up_ratio=args.up_ratio,
        )

        # Generate correlated noise for FLUX
        pipe.generate_noise(device, args.height, args.width)
        
        torch.cuda.empty_cache()

        print("Generating image with FLUX model...")
        image = pipe(
            args.prompt,
            guidance_scale=args.guidance_scale,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            height=args.height,
            width=args.width,
        ).images[0]

    elif args.model_type == 'qwen':
        # --- QwenImage Model Logic (with diffusers) ---
        pipe = QwenImagePipelineRALU.from_pretrained(
            args.qwen_edit_path,
            torch_dtype=torch_dtype
        ).to(device)

        # Load the QwenEdit weights if a path is provided
        # Note: The base model might already be the edited one, but this allows overriding or fine-tuning.
        if args.qwen_edit_path and os.path.exists(args.qwen_edit_path):
            print(f"Loading custom QwenEdit LoRA weights from {args.qwen_edit_path}")
            pipe.load_lora_weights(args.qwen_edit_path)

        # Configure RALU for Qwen
        if args.use_ralu:
            pipe.set_ralu_params(level=args.level, num_inference_steps=args.num_inference_steps)
        else:
            pipe.set_ralu_params(level=None) # Disable RALU

        # Load the image to be edited
        edit_image = Image.open(args.edit_image_path).convert("RGB")

        print("Generating image with QwenImage model...")
        image = pipe(
            prompt=args.prompt,
            image=edit_image,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            true_cfg_scale=args.guidance_scale,
            generator=torch.Generator(device).manual_seed(args.seed)
        ).images[0]

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Save the output image
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "output.png")
    image.save(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RALU accelerated inference with either FLUX or QwenImage model.")

    # Common arguments
    parser.add_argument('--model_type', type=str, default='flux', choices=['flux', 'qwen'], help='Model to use for inference.')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for image generation or editing.')
    parser.add_argument('--output_dir', type=str, default='./outputs_ralu', help='Directory to save the generated image.')
    parser.add_argument('--height', type=int, default=1024, help='Image height.')
    parser.add_argument('--width', type=int, default=1024, help='Image width.')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='Guidance scale (CFG).')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation.')
    parser.add_argument('--flux_model_path', type=str, default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux', help='Path to the FLUX model.')
    parser.add_argument('--qwen_edit_path', type=str, default='/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit', help='Path to the QwenEdit model.')

    # RALU arguments
    parser.add_argument('--use_ralu', action='store_true', help='Enable RALU acceleration.')
    parser.add_argument('--level', type=int, default=4, choices=[4, 7], help='RALU speedup level (4x or 7x).')

    # FLUX-specific arguments
    parser.add_argument('--use_RALU_default', action='store_true', help='(FLUX only) Use RALU default setting.')
    parser.add_argument('--N', type=int, default=None, nargs='+', help='(FLUX only) Number of steps for each stage.')
    parser.add_argument('--e', type=float, default=None, nargs='+', help='(FLUX only) End timestep for each stage.')
    parser.add_argument('--up_ratio', type=float, default=0.3, help='(FLUX only) Upsampling ratio.')

    # Qwen-specific arguments
    parser.add_argument('--edit_image_path', type=str, help='(Qwen only) Path to the input image to be edited.')


    args = parser.parse_args()
    
    if args.model_type == 'qwen' and not args.edit_image_path:
        parser.error("--edit_image_path is required when --model_type is 'qwen'")

    main(args)