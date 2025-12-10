import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Suppress warnings
logging.set_verbosity_error()

try:
    from ..config import *
except ImportError:
    # Fallback for when running directly or when relative import fails
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import *


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    """Generate overlapping views for MultiDiffusion."""
    panorama_height = int(panorama_height / 8)
    panorama_width = int(panorama_width / 8)
    num_blocks_height = int((panorama_height - window_size) // stride + 1)
    num_blocks_width = int((panorama_width - window_size) // stride + 1)
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusionGenerator(nn.Module):
    """
    MultiDiffusion implementation for region-based ad generation.
    Based on official implementation from the paper.
    """
    
    def __init__(self, model_name: str = DIFFUSION_MODEL, device: str = "cuda"):
        super().__init__()
        
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise SystemExit("CUDA not available; stopping.")
        
        print(f'[INFO] Loading Stable Diffusion {model_name}...')
        
        # Determine model version
        if "3.5-large" in model_name:
            model_key = "stabilityai/stable-diffusion-3.5-large"
        elif "2-1" in model_name or "2.1" in model_name:
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif "2-base" in model_name or "2.0" in model_name or "2-base" in model_name:
            model_key = "stabilityai/stable-diffusion-2-base"
        elif "1-5" in model_name or "1.5" in model_name:
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            model_key = model_name
        
        # Load models
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] Stable Diffusion loaded successfully!')
    
    @torch.no_grad()
    def get_random_background(self, n_samples):
        backgrounds = torch.rand(
            n_samples, 3, device=self.device
        )[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])
    
    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        """Get text embeddings for prompts."""
        # Handle single prompt or list
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        
        # Tokenize and embed
        text_input = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        
        # Unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode images to latent space."""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to images."""
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    
    def create_mask_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        height: int,
        width: int,
        feather: int = 10
    ) -> torch.Tensor:
        """
        Create a mask from bounding box with optional feathering.
        """
        # Create PIL image for drawing
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Draw filled rectangle
        draw.rectangle(bbox, fill=255)
        
        # Convert to numpy and normalize
        mask = np.array(mask_img).astype(np.float32) / 255.0
        
        # Apply Gaussian blur for feathering
        if feather > 0:
            import cv2
            kernel_size = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather / 3)
        
        # Convert to tensor and resize to latent dimensions
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(height // 8, width // 8),
            mode='nearest'
        )
        
        return mask
    
    @torch.no_grad()
    def generate_ad(
        self,
        main_prompt: str,
        region_prompts: List[Dict],
        logo_image: Optional[Image.Image] = None,
        logo_position: Tuple[int, int] = (600, 50),
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
        num_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        seed: Optional[int] = None,
        bootstrapping: int = 20,
        negative_prompt: str = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, ugly, deformed"
    ) -> Image.Image:
        """
        Generate ad using MultiDiffusion with region control.
        """
        if seed is not None:
            seed_everything(seed)
        
        # Prepare prompts and masks
        prompts = [main_prompt]
        masks = []
        
        # Sort regions by importance
        importance_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        sorted_regions = sorted(
            region_prompts,
            key=lambda x: importance_order.get(x.get('importance', 'medium'), 1)
        )
        
        # Create masks for each region
        for region in sorted_regions:
            if 'bbox' in region and region['bbox']:
                bbox = region['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    mask = self.create_mask_from_bbox(bbox, height, width)
                    masks.append(mask)
                    prompts.append(region['prompt'])
        
        # If no regions specified, use full image
        if len(masks) == 0:
            print("[WARNING] No valid regions specified, using single prompt generation")
            return self._generate_simple(
                main_prompt, width, height, num_steps, guidance_scale, negative_prompt
            )
        
        # Stack masks and create background mask
        fg_masks = torch.cat(masks, dim=0)
        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
        bg_mask = torch.clamp(bg_mask, 0, 1)
        all_masks = torch.cat([bg_mask, fg_masks], dim=0)
        
        print(f"[INFO] Generating with {len(prompts)} prompts (1 background + {len(prompts)-1} regions)")
        
        # Generate with MultiDiffusion
        img = self._multidiffusion_generate(
            all_masks,
            prompts,
            negative_prompt,
            height,
            width,
            num_steps,
            guidance_scale,
            bootstrapping
        )
        
        # Composite logo if provided
        if logo_image is not None:
            img = self._composite_logo(img, logo_image, logo_position)
        
        return img
    
    @torch.no_grad()
    def _multidiffusion_generate(
        self,
        masks: torch.Tensor,
        prompts: List[str],
        negative_prompts: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        bootstrapping: int
    ) -> Image.Image:
        """Core MultiDiffusion generation algorithm."""
        
        # Get bootstrapping backgrounds
        bootstrapping_backgrounds = self.get_random_background(bootstrapping)
        
        # Get text embeddings
        neg_prompts = [negative_prompts] * len(prompts)
        text_embeds = self.get_text_embeds(prompts, neg_prompts)
        
        # Initialize latent
        latent = torch.randn(
            (1, self.unet.in_channels, height // 8, width // 8),
            device=self.device
        )
        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        
        # Get views for overlapping generation
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        print(f"[INFO] Running {num_inference_steps} denoising steps...")
        
        with torch.autocast('cuda' if self.device == 'cuda' else 'cpu'):
            for i, t in enumerate(self.scheduler.timesteps):
                if (i + 1) % 10 == 0:
                    print(f"  Step {i+1}/{num_inference_steps}")
                
                count.zero_()
                value.zero_()
                
                # Process each view
                for h_start, h_end, w_start, w_end in views:
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(
                        len(prompts), 1, 1, 1
                    )
                    
                    # Bootstrapping
                    if i < bootstrapping:
                        bg_indices = torch.randint(
                            0, bootstrapping, (len(prompts) - 1,)
                        )
                        bg = bootstrapping_backgrounds[bg_indices]
                        bg = self.scheduler.add_noise(
                            bg, noise[:, :, h_start:h_end, w_start:w_end], t
                        )
                        latent_view[1:] = (
                            latent_view[1:] * masks_view[1:] + 
                            bg * (1 - masks_view[1:])
                        )
                    
                    # Classifier-free guidance
                    latent_model_input = torch.cat([latent_view] * 2)
                    
                    # Predict noise
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeds
                    )['sample']
                    
                    # Apply guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond + 
                        guidance_scale * (noise_pred_text - noise_pred_uncond)
                    )
                    
                    # Denoise
                    latents_view_denoised = self.scheduler.step(
                        noise_pred, t, latent_view
                    )['prev_sample']
                    
                    # Accumulate
                    value[:, :, h_start:h_end, w_start:w_end] += (
                        latents_view_denoised * masks_view
                    ).sum(dim=0, keepdims=True)
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(
                        dim=0, keepdims=True
                    )
                
                # MultiDiffusion step
                latent = torch.where(count > 0, value / count, value)
        
        # Decode to image
        imgs = self.decode_latents(latent)
        img = T.ToPILImage()(imgs[0].cpu())
        
        return img
    
    @torch.no_grad()
    def _generate_simple(
        self,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        negative_prompt: str
    ) -> Image.Image:
        """Simple single-prompt generation (fallback)."""
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image
    
    def _composite_logo(
        self,
        image: Image.Image,
        logo: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """Composite logo onto generated image."""
        result = image.copy()
        logo_resized = logo.resize(LOGO_REGION_SIZE, Image.Resampling.LANCZOS)
        
        if logo_resized.mode == 'RGBA':
            result.paste(logo_resized, position, logo_resized)
        else:
            result.paste(logo_resized, position)
        
        return result


if __name__ == "__main__":
    # Test
    generator = MultiDiffusionGenerator()
    
    test_regions = [
        {
            "name": "Product",
            "prompt": "sleek laptop on clean surface, product photography",
            "bbox": (200, 150, 550, 400),
            "importance": "critical"
        }
    ]
    
    img = generator.generate_ad(
        main_prompt="modern tech advertisement background, gradient blue",
        region_prompts=test_regions,
        seed=42
    )
    img.save("test_multidiffusion.png")
    print("Test complete!")