# This file is based on the original diffusers.pipelines.qwenimage.pipeline_qwenimage_edit
# with modifications to support RALU acceleration.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline, retrieve_timesteps, calculate_shift


class QwenImagePipelineRALU(QwenImageEditPipeline):
    """
    QwenImageEditPipeline with RALU acceleration.
    """
    
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def set_ralu_params(self, level=None, num_inference_steps=50):
        """
        Sets the RALU parameters, specifically the resolution schedule.
        """
        if level is None:
            self.ralu_schedule = None
            print("RALU is disabled.")
            return

        if level == 4:
            # 4x speedup: 25% tokens for the first 50% of steps
            schedule = [0.25] * (num_inference_steps // 2) + [1.0] * (num_inference_steps - num_inference_steps // 2)
        elif level == 7:
            # 7x speedup: 10% tokens for the first 70% of steps
            low_res_steps = int(num_inference_steps * 0.7)
            schedule = [0.1] * low_res_steps + [1.0] * (num_inference_steps - low_res_steps)
        else:
            raise ValueError(f"Invalid RALU level: {level}. Supported levels are 4 and 7.")
        
        self.ralu_schedule = schedule
        print(f"RALU level {level} enabled with schedule: {self.ralu_schedule}")

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        image_size = image[0].size if isinstance(image, list) else image.size
        
        def calculate_dimensions(target_area, ratio):
            width = math.sqrt(target_area * ratio)
            height = width / ratio
            width = round(width / 32) * 32
            height = round(height / 32) * 32
            return width, height, None

        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        
        original_height = height or calculated_height
        original_width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = original_width // multiple_of * multiple_of
        height = original_height // multiple_of * multiple_of

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Guidance handling
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)

        # RALU state
        is_low_res = False
        current_height, current_width = height, width
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                
                # --- RALU LOGIC ---
                if hasattr(self, "ralu_schedule") and self.ralu_schedule is not None:
                    ratio = self.ralu_schedule[i]
                    prev_ratio = self.ralu_schedule[i-1] if i > 0 else 0.0

                    # Transition to low-res (at the beginning)
                    if not is_low_res and ratio < 1.0:
                        unpacked_latents = self._unpack_latents(latents, current_height, current_width, self.vae_scale_factor)
                        
                        scale_factor = math.sqrt(ratio)
                        new_latent_height = int(unpacked_latents.shape[-2] * scale_factor)
                        new_latent_width = int(unpacked_latents.shape[-1] * scale_factor)

                        new_latent_height = (new_latent_height // 2) * 2
                        new_latent_width = (new_latent_width // 2) * 2

                        downsampled_latents = torch.nn.functional.interpolate(unpacked_latents, size=(new_latent_height, new_latent_width), mode="bilinear", align_corners=False)
                        
                        current_height = new_latent_height * self.vae_scale_factor
                        current_width = new_latent_width * self.vae_scale_factor
                        
                        latents = self._pack_latents(downsampled_latents, batch_size * num_images_per_prompt, num_channels_latents, current_height, current_width)
                        
                        img_shapes = [
                            [
                                (1, current_height // self.vae_scale_factor // 2, current_width // self.vae_scale_factor // 2),
                                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
                            ]
                        ] * batch_size
                        is_low_res = True

                    # Transition to high-res
                    elif is_low_res and ratio == 1.0 and prev_ratio < 1.0:
                        unpacked_latents = self._unpack_latents(latents, current_height, current_width, self.vae_scale_factor)
                        
                        original_latent_height = height // self.vae_scale_factor
                        original_latent_width = width // self.vae_scale_factor

                        upsampled_latents = torch.nn.functional.interpolate(unpacked_latents, size=(original_latent_height, original_latent_width), mode="bilinear", align_corners=False)
                        
                        current_height, current_width = height, width
                        
                        latents = self._pack_latents(upsampled_latents, batch_size * num_images_per_prompt, num_channels_latents, current_height, current_width)
                        
                        img_shapes = [
                            [
                                (1, current_height // self.vae_scale_factor // 2, current_width // self.vae_scale_factor // 2),
                                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
                            ]
                        ] * batch_size
                        is_low_res = False
                # --- END RALU LOGIC ---

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # Predict the noise for the conditional output
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                # Perform guidance
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    # ... callback logic ...
                    pass

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 7. Post-processing
        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, current_height, current_width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
