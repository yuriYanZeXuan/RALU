import argparse
import os
import torch
import sys
from PIL import Image

# Add VLM-Lora-Moe to path to import diffsynth
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLM-Lora-Moe')))

from pipeline_qwen_image_ralu import QwenImagePipelineRALU
from diffsynth.utils.model_config import ModelConfig


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    # These configs assume the base Qwen-Image model is available from Hugging Face.
    # The `from_pretrained` method will download them if not cached.
    model_configs = [
        ModelConfig(model_id="Qwen/Qwen-Image", file_pattern="text_encoder.safetensors", model_name="qwen_image_text_encoder"),
        ModelConfig(model_id="Qwen/Qwen-Image", file_pattern="dit.safetensors", model_name="qwen_image_dit"),
        ModelConfig(model_id="Qwen/Qwen-Image", file_pattern="vae.safetensors", model_name="qwen_image_vae"),
    ]

    pipe = QwenImagePipelineRALU.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs
    )

    # Load the QwenEdit weights.
    # The user provided a path to the weights.
    if args.qwen_edit_path:
        if os.path.exists(args.qwen_edit_path):
            print(f"Loading QwenEdit weights from {args.qwen_edit_path}")
            pipe.load_moe_lora(pipe.dit, args.qwen_edit_path, alpha=1.0)
        else:
            print(f"Warning: QwenEdit weights path not found: {args.qwen_edit_path}")

    # Configure RALU based on command-line arguments
    if args.use_ralu:
        pipe.set_ralu_params(level=args.level, num_inference_steps=args.num_inference_steps)
    else:
        pipe.set_ralu_params(level=None)

    # Load the image to be edited
    if not args.edit_image_path or not os.path.exists(args.edit_image_path):
        raise ValueError(f"Input image for editing not found at: {args.edit_image_path}")
    edit_image = Image.open(args.edit_image_path)

    # Generate the image
    print("Generating image...")
    image = pipe(
        prompt=args.prompt,
        edit_image=edit_image,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        cfg_scale=args.guidance_scale,
        seed=args.seed
    )
    
    # Save the output image
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "output.png")
    image.save(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RALU accelerated inference for Qwen-Image-Edit")

    parser.add_argument('--prompt', type=str, required=True, help='The editing prompt.')
    parser.add_argument('--edit_image_path', type=str, required=True, help='Path to the input image to be edited.')
    parser.add_argument('--output_dir', type=str, default='./outputs_qwen_ralu', help='Directory to save the generated image.')
    parser.add_argument('--qwen_edit_path', type=str, default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/QwenEdit2509", help='Path to the QwenEdit LoRA weights.')
    
    parser.add_argument('--height', type=int, default=1024, help='Image height.')
    parser.add_argument('--width', type=int, default=1024, help='Image width.')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='Guidance scale (CFG).')
    parser.add_argument('--num_inference_steps', type=int, default=30, help='Number of denoising steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation.')

    parser.add_argument('--use_ralu', action='store_true', help='Enable RALU acceleration.')
    parser.add_argument('--level', type=int, default=4, choices=[4, 7], help='RALU speedup level (4x or 7x).')

    args = parser.parse_args()
    main(args)
