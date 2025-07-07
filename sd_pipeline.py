import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
import os
from PIL import Image
import logging
import numpy as np

from laplacian import LaplacianPyramidTorch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predicted_original_from_noise(latent, noise_pred, timestep, alphas_cumprod):
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha 
    predicted_original_sample = (latent - beta.sqrt() * noise_pred) / alpha.sqrt()
    return predicted_original_sample
    
def noise_from_predicted_original(latent, predicted_original, timestep, alphas_cumprod):
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha
    noise = (latent - alpha.sqrt() * predicted_original) / beta.sqrt()
    return noise

def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

class StableDiffusionGenerator:
    """
    A comprehensive Stable Diffusion pipeline for image generation using diffusers library.
    """

    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base", device="auto"):
        """
        Initialize the Stable Diffusion pipeline with individual components.

        Args:
            model_id (str): Hugging Face model ID for Stable Diffusion
            device (str): Device to run the model on ("auto", "cuda", "cpu")
        """
        self.model_id = model_id
        self.device = self._get_device(device)
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self._load_components()

    def _get_device(self, device):
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_components(self):
        """Load individual Stable Diffusion components from pipeline."""
        try:
            logger.info(f"Loading Stable Diffusion pipeline from: {self.model_id}")
            logger.info(f"Using device: {self.device}")

            # Determine the correct dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Load the full pipeline first from cache
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True  # Use cached models only
            )

            # Extract components from the pipeline
            self.unet = pipeline.unet.to(self.device)
            self.vae = pipeline.vae.to(self.device)
            self.text_encoder = pipeline.text_encoder.to(self.device)
            self.tokenizer = pipeline.tokenizer
            self.scheduler = pipeline.scheduler

            # Store alphas_cumprod for FDG
            self.alphas_cumprod = self.scheduler.alphas_cumprod
            self.pyramid = LaplacianPyramidTorch(levels=2, dtype=dtype)

            # Set components to evaluation mode
            self.unet.eval()
            self.vae.eval()
            self.text_encoder.eval()

            # Clean up the pipeline
            del pipeline

            logger.info("All components loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise

    def _encode_prompt(self, prompt, negative_prompt=""):
        """Encode text prompts using CLIP text encoder."""
        # Tokenize prompts
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        negative_text_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        negative_text_input = {k: v.to(self.device) for k, v in negative_text_input.items()}

        # Encode text
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input["input_ids"])[0]
            negative_text_embeddings = self.text_encoder(negative_text_input["input_ids"])[0]

        # Concatenate negative and positive embeddings
        text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])

        return text_embeddings

    def _prepare_latents(self, batch_size, height, width, generator=None):
        """Prepare initial latents for diffusion process."""
        # Calculate latent dimensions
        latents_height = height // 8
        latents_width = width // 8

        # Determine the correct dtype for latents
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Generate random latents
        latents = torch.randn(
            (batch_size, 4, latents_height, latents_width),
            generator=generator,
            device=self.device,
            dtype=dtype
        )

        # Scale latents
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _denoise_step(self, latents, timesteps, text_embeddings, guidance_scale):
        """Perform one denoising step."""
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Predict noise residual
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous sample
        latents = self.scheduler.step(noise_pred, timesteps, latents).prev_sample

        return latents

    def _decode_latents(self, latents):
        """Decode latents to images using VAE."""
        # Scale latents
        latents = 1 / 0.18215 * latents

        # Decode latents
        with torch.no_grad():
            images = self.vae.decode(latents).sample

        # Convert to PIL images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(image) for image in images]

        return images

    def generate_image_grid(
        self,
        prompts,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        rows=2,
        cols=2,
        seed=None,
        use_fdg=False,
    ):
        """
        Generate a grid of images from multiple prompts.

        Args:
            prompts (list): List of text prompts
            negative_prompt (str): Text prompt for what to avoid
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            seed (int): Random seed for reproducibility

        Returns:
            PIL.Image: Grid image
        """
        all_images = []

        for i, prompt in enumerate(prompts):
            if i >= rows * cols:
                break

            # Use different seed for each image if no specific seed provided
            current_seed = seed + i if seed is not None else None

            images = self.__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images=1,
                seed=current_seed,
                use_fdg=use_fdg,
            )
            all_images.extend(images)

        # Create image grid
        grid_image = make_image_grid(all_images, rows=rows, cols=cols)
        return grid_image

    def save_images(self, images, output_dir="generated_images", prefix="sd_image"):
        """
        Save generated images to disk.

        Args:
            images (list): List of PIL Image objects
            output_dir (str): Directory to save images
            prefix (str): Prefix for image filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, image in enumerate(images):
            filename = f"{prefix}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            logger.info(f"Saved image: {filepath}")

    def __del__(self):
        """Clean up resources."""
        components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
        for comp in components:
            if hasattr(self, comp) and getattr(self, comp) is not None:
                delattr(self, comp)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_image_cfg(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        num_images=1,
        seed=None
    ):
        """
        Main inference method - implements the complete Stable Diffusion pipeline.

        Args:
            prompt (str): Text prompt describing the desired image
            negative_prompt (str): Text prompt for what to avoid in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            num_images (int): Number of images to generate
            seed (int): Random seed for reproducibility

        Returns:
            list: List of PIL Image objects
        """
        if any(comp is None for comp in [self.unet, self.vae, self.text_encoder, self.tokenizer, self.scheduler]):
            raise RuntimeError("Components not loaded. Call _load_components() first.")

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        try:
            logger.info(f"Generating image with prompt: '{prompt}'")

            # Set scheduler timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            # Encode text prompts
            text_embeddings = self._encode_prompt(prompt, negative_prompt)

            # Prepare latents
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
            latents = self._prepare_latents(num_images, height, width, generator)

            # Denoising loop
            for i, timestep in enumerate(timesteps):
                logger.info(f"Denoising step {i+1}/{len(timesteps)}")

                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)

                # Predict noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings
                    ).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous sample
                latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

            # Decode latents to images
            images = self._decode_latents(latents)

            logger.info(f"Generated {len(images)} image(s) successfully!")
            return images

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

    def generate_image_fdg(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        num_images=1,
        seed=None
    ):
        """
        The complete Stable Diffusion pipeline using FDG method from
        "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales."

        Args:
            prompt (str): Text prompt describing the desired image
            negative_prompt (str): Text prompt for what to avoid in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            num_images (int): Number of images to generate
            seed (int): Random seed for reproducibility

        Returns:
            list: List of PIL Image objects
        """
        if any(comp is None for comp in [self.unet, self.vae, self.text_encoder, self.tokenizer, self.scheduler]):
            raise RuntimeError("Components not loaded. Call _load_components() first.")

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        try:
            logger.info(f"Generating image with prompt: '{prompt}'")

            guidance_scale_high = guidance_scale
            guidance_scale_low = guidance_scale * 0.5

            # Set scheduler timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            # Encode text prompts
            text_embeddings = self._encode_prompt(prompt, negative_prompt)

            # Prepare latents
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
            latents = self._prepare_latents(num_images, height, width, generator)

            # Denoising loop
            for i, timestep in enumerate(timesteps):
                logger.info(f"Denoising step {i+1}/{len(timesteps)}")

                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)

                # Predict noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings
                    ).sample


                # Predicted original
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                predicted_x0_uncond = predicted_original_from_noise(latents, noise_pred_uncond, timestep, self.scheduler.alphas_cumprod)
                predicted_x0_cond = predicted_original_from_noise(latents, noise_pred_text, timestep, self.scheduler.alphas_cumprod)
                # Construct pyramid
                pred_uncond_pyramid = self.pyramid.build_laplacian_pyramid(predicted_x0_uncond) # high to low
                pred_text_pyramid = self.pyramid.build_laplacian_pyramid(predicted_x0_cond) # high to low
                pyramid_list = []
                for d_c, d_u, scale in zip(pred_text_pyramid, pred_uncond_pyramid, [guidance_scale_high, guidance_scale_low]):
                    diff = d_c - d_u
                    diff_par, diff_ort = project(diff, d_c)
                    diff = diff_par + diff_ort
                    p_guided = d_c + scale * diff
                    pyramid_list.append(p_guided)

                predicted_x0 = self.pyramid.reconstruct_from_laplacian(pyramid_list)
                noise_pred = noise_from_predicted_original(latents, predicted_x0, timestep, self.scheduler.alphas_cumprod)

                # Compute previous sample
                latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

            # Decode latents to images
            images = self._decode_latents(latents)

            logger.info(f"Generated {len(images)} image(s) successfully!")
            return images

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

    def __call__(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        num_images=1,
        seed=None,
        use_fdg=False
    ):
        """
        Generate images using Stable Diffusion (alias for __call__).

        Args:
            prompt (str): Text prompt describing the desired image
            negative_prompt (str): Text prompt for what to avoid in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            num_images (int): Number of images to generate
            seed (int): Random seed for reproducibility

        Returns:
            list: List of PIL Image objects
        """
        if use_fdg is False:
            result = self.generate_image_cfg(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed
            )
        else:
            result = self.generate_image_fdg(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed,
        )
        return result