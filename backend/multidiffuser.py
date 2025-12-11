import os
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
        
        # Handle device specification (cuda, cuda:0, cuda:1, cpu)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise SystemExit("CUDA not available; stopping.")
            # If device is just "cuda", use default (cuda:0)
            # Otherwise use the specified device (cuda:0, cuda:1, etc.)
            if device == "cuda":
                self.device = torch.device("cuda:0")
            else:
                # Extract GPU index from "cuda:X"
                self.device = torch.device(device)
            # Verify the device exists
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise SystemExit(f"GPU {self.device.index} not available. Only {torch.cuda.device_count()} GPU(s) available.")
        else:
            self.device = torch.device(device)
        
        print(f'[INFO] Using device: {self.device}')
        
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
    
    def load_mask_from_image(
        self,
        mask_path: str,
        height: int,
        width: int,
        feather: int = 0
    ) -> torch.Tensor:
        """
        Load mask from image file and convert to tensor.
        
        Args:
            mask_path: Path to mask image (grayscale, white pixels = region)
            height: Target height
            width: Target width
            feather: Optional feathering amount (0 = no feathering)
        
        Returns:
            Mask tensor ready for MultiDiffusion
        """
        import cv2
        
        # Load mask image
        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize((width, height), Image.LANCZOS)
        
        # Convert to numpy and normalize
        mask = np.array(mask_img).astype(np.float32) / 255.0
        
        # Apply Gaussian blur for feathering if requested
        if feather > 0:
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
    
    def get_position_from_mask(
        self,
        mask_path: str,
        height: int,
        width: int,
        method: str = "centroid"
    ) -> Tuple[int, int]:
        """
        Calculate logo position from mask image.
        
        Args:
            mask_path: Path to mask image (grayscale, white pixels = region)
            height: Target image height
            width: Target image width
            method: "centroid" (center of mass) or "bbox_center" (center of bounding box)
        
        Returns:
            (x, y) position tuple
        """
        import cv2
        import numpy as np
        
        # Load mask image
        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize((width, height), Image.LANCZOS)
        mask_array = np.array(mask_img)
        
        # Find non-zero pixels
        coords = np.column_stack(np.where(mask_array > 128))  # Threshold at 128
        
        if len(coords) == 0:
            # No mask region found, return center
            return (width // 2, height // 2)
        
        if method == "centroid":
            # Calculate centroid (center of mass)
            y_center = int(np.mean(coords[:, 0]))
            x_center = int(np.mean(coords[:, 1]))
        elif method == "bbox_center":
            # Calculate bounding box center
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
        else:
            raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'bbox_center'")
        
        return (x_center, y_center)
    
    @torch.no_grad()
    def generate_ad(
        self,
        main_prompt: str,
        region_prompts: List[Dict],
        logo_image: Optional[Image.Image] = None,
        logo_position: Optional[Tuple[int, int]] = None,
        logo_mask_path: Optional[str] = None,
        logo_size: Optional[Tuple[int, int]] = None,
        preserve_logo_aspect: bool = True,
        logo_blend_mode: str = "soft_light",
        logo_opacity: float = 0.9,
        logo_feather: bool = True,
        logo_feather_radius: int = 5,
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
        num_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        seed: Optional[int] = None,
        bootstrapping: int = 20,
        negative_prompt: str = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, ugly, deformed",
        mask_path: Optional[str] = None,
        mask_paths: Optional[List[str]] = None,
        mask_prompt: Optional[str] = None,
        mask_prompts: Optional[List[str]] = None,
        mask_feather: int = 10,
        background_prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Generate ad using MultiDiffusion with region control.
        
        Args:
            mask_path: Optional path to single mask image file (overrides region bboxes)
            mask_paths: Optional list of mask image paths (for multiple masks)
            mask_prompts: Optional list of prompts for each mask (must match mask_paths length)
            mask_feather: Feathering amount for mask image (default: 10)
            background_prompt: Optional specific prompt for background. If None, uses main_prompt.
        """
        if seed is not None:
            seed_everything(seed)
        
        # Use background_prompt if provided, otherwise use main_prompt
        bg_prompt = background_prompt if background_prompt is not None else main_prompt
        
        # Prepare prompts and masks
        prompts = [bg_prompt]  # First prompt is always background
        masks = []
        
        # If mask_paths (multiple masks) is provided, use them
        if mask_paths:
            if mask_prompts is None:
                mask_prompts = [main_prompt] * len(mask_paths)
            elif len(mask_prompts) != len(mask_paths):
                raise ValueError(f"Number of mask_prompts ({len(mask_prompts)}) must match number of mask_paths ({len(mask_paths)})")
            
            print(f"[INFO] Loading {len(mask_paths)} mask(s) from image(s)")
            for i, (mask_path_item, mask_prompt_item) in enumerate(zip(mask_paths, mask_prompts)):
                print(f"  Mask {i+1}: {mask_path_item} -> '{mask_prompt_item[:50]}...'")
                mask = self.load_mask_from_image(mask_path_item, height, width, feather=mask_feather)
                masks.append(mask)
                prompts.append(mask_prompt_item)
        
        # If single mask_path is provided, use it directly
        elif mask_path:
            print(f"[INFO] Loading mask from image: {mask_path}")
            mask = self.load_mask_from_image(mask_path, height, width, feather=mask_feather)
            masks.append(mask)
            
            # Use mask_prompt if provided, otherwise first region prompt, otherwise main prompt
            if mask_prompt:
                prompts.append(mask_prompt)
                print(f"  Using custom mask prompt: '{mask_prompt[:50]}...'")
            elif region_prompts and len(region_prompts) > 0:
                prompts.append(region_prompts[0].get('prompt', main_prompt))
            else:
                prompts.append(main_prompt)
        else:
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
        
        bg_prompt_display = bg_prompt[:50] + "..." if len(bg_prompt) > 50 else bg_prompt
        print(f"[INFO] Generating with {len(prompts)} prompts:")
        print(f"  Background: '{bg_prompt_display}'")
        print(f"  Regions: {len(prompts)-1}")
        
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
        
        # Composite logo if provided with blending
        if logo_image is not None:
            print(f"[INFO] Compositing logo onto image...")
            # Determine logo position
            final_logo_position = logo_position
            
            # If logo_mask_path is provided, calculate position from mask
            if logo_mask_path:
                if os.path.exists(logo_mask_path):
                    print(f"[INFO] Calculating logo position from mask: {logo_mask_path}")
                    final_logo_position = self.get_position_from_mask(
                        logo_mask_path, height, width, method="centroid"
                    )
                    print(f"  Logo will be placed at: {final_logo_position}")
                else:
                    print(f"[WARNING] Logo mask not found: {logo_mask_path}, using default position")
                    final_logo_position = logo_position if logo_position else (600, 50)
            elif logo_position is None:
                # Default position if neither specified
                final_logo_position = (600, 50)
            
            print(f"[INFO] Compositing logo at position {final_logo_position} with size {logo_size}, blend_mode={logo_blend_mode}, opacity={logo_opacity}")
            img = self._composite_logo(
                img, 
                logo_image, 
                final_logo_position,
                logo_size=logo_size,
                preserve_aspect=preserve_logo_aspect,
                blend_mode=logo_blend_mode,
                opacity=logo_opacity,
                feather_edges=logo_feather,
                feather_radius=logo_feather_radius
            )
            print(f"[INFO] Logo compositing complete!")
        else:
            print(f"[INFO] No logo image provided, skipping logo compositing")
        
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
        print(f"[INFO] Using negative prompt: '{negative_prompts[:100]}...'")
        print(f"[INFO] Number of prompts: {len(prompts)}")
        for i, (p, np) in enumerate(zip(prompts, neg_prompts)):
            print(f"  Prompt {i}: '{p[:50]}...'")
            print(f"  Negative {i}: '{np[:50]}...'")
        text_embeds = self.get_text_embeds(prompts, neg_prompts)
        print(f"[INFO] Text embeddings shape: {text_embeds.shape} (expected [2*{len(prompts)}, 77, 768])")
        
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
        
        with torch.autocast('cuda' if self.device.type == 'cuda' else 'cpu', device_type=str(self.device.type)):
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
        position: Tuple[int, int],
        logo_size: Optional[Tuple[int, int]] = None,
        preserve_aspect: bool = True,
        blend_mode: str = "normal",
        opacity: float = 1.0,
        feather_edges: bool = True,
        feather_radius: int = 5
    ) -> Image.Image:
        """
        Composite logo onto generated image with blending options.
        
        Args:
            image: Generated image to composite onto
            logo: Logo image to composite
            position: (x, y) position to place logo
            logo_size: Optional (width, height) to resize logo. If None, uses original size.
            preserve_aspect: If True and logo_size provided, preserves aspect ratio
            blend_mode: Blending mode - "normal", "multiply", "screen", "overlay", "soft_light"
            opacity: Opacity of logo (0.0 to 1.0)
            feather_edges: Whether to feather/soften logo edges
            feather_radius: Radius for edge feathering
        """
        import cv2
        import numpy as np
        
        result = image.copy()
        
        # Use logo as-is if no size specified
        if logo_size is None:
            logo_to_use = logo.copy()
        elif preserve_aspect:
            # Preserve aspect ratio
            logo_w, logo_h = logo.size
            target_w, target_h = logo_size
            
            # Calculate scaling to fit within target size while preserving aspect
            scale = min(target_w / logo_w, target_h / logo_h)
            new_w = int(logo_w * scale)
            new_h = int(logo_h * scale)
            
            logo_to_use = logo.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            # Resize to exact size (may distort)
            logo_to_use = logo.resize(logo_size, Image.Resampling.LANCZOS)
        
        # Ensure logo has alpha channel
        if logo_to_use.mode != 'RGBA':
            logo_to_use = logo_to_use.convert('RGBA')
        
        # Extract alpha channel
        logo_array = np.array(logo_to_use)
        logo_rgb = logo_array[:, :, :3]
        logo_alpha = logo_array[:, :, 3:4].astype(np.float32) / 255.0
        
        # Apply opacity
        logo_alpha = logo_alpha * opacity
        
        # Feather edges if requested
        if feather_edges and feather_radius > 0:
            # Create feathered alpha mask
            alpha_mask = (logo_alpha[:, :, 0] * 255).astype(np.uint8)
            kernel_size = feather_radius * 2 + 1
            feathered_alpha = cv2.GaussianBlur(alpha_mask, (kernel_size, kernel_size), feather_radius / 2)
            feathered_alpha = feathered_alpha.astype(np.float32) / 255.0
            logo_alpha = feathered_alpha[:, :, np.newaxis]
        
        # Get the region from the base image
        x, y = position
        logo_w, logo_h = logo_to_use.size[0], logo_to_use.size[1]
        
        # Ensure we don't go out of bounds
        img_w, img_h = result.size
        
        # Calculate crop region
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(x + logo_w, img_w)
        y_end = min(y + logo_h, img_h)
        
        # Check if logo is completely outside image
        if x_end <= 0 or y_end <= 0 or x_start >= img_w or y_start >= img_h:
            return result
        
        # Calculate offsets for logo cropping
        logo_x_offset = max(0, -x)
        logo_y_offset = max(0, -y)
        logo_crop_w = x_end - x_start
        logo_crop_h = y_end - y_start
        
        # Crop logo if needed (when partially outside image)
        if logo_x_offset > 0 or logo_y_offset > 0 or logo_crop_w < logo_w or logo_crop_h < logo_h:
            logo_rgb = logo_rgb[logo_y_offset:logo_y_offset+logo_crop_h, logo_x_offset:logo_x_offset+logo_crop_w]
            logo_alpha = logo_alpha[logo_y_offset:logo_y_offset+logo_crop_h, logo_x_offset:logo_x_offset+logo_crop_w]
        
        # Get the corresponding region from the base image
        base_region = np.array(result.crop((x_start, y_start, x_end, y_end))).astype(np.float32)
        
        # Ensure base_region has alpha if needed
        if base_region.shape[2] == 3:
            base_alpha = np.ones((base_region.shape[0], base_region.shape[1], 1), dtype=np.float32)
        else:
            base_alpha = base_region[:, :, 3:4].astype(np.float32) / 255.0
            base_region = base_region[:, :, :3]
        
        # Apply blending mode
        if blend_mode == "normal":
            blended_rgb = logo_rgb
        elif blend_mode == "multiply":
            blended_rgb = (base_region * logo_rgb / 255.0).astype(np.uint8)
        elif blend_mode == "screen":
            blended_rgb = (255 - (255 - base_region) * (255 - logo_rgb) / 255.0).astype(np.uint8)
        elif blend_mode == "overlay":
            mask = base_region < 128
            blended_rgb = np.where(
                mask,
                2 * base_region * logo_rgb / 255.0,
                255 - 2 * (255 - base_region) * (255 - logo_rgb) / 255.0
            ).astype(np.uint8)
        elif blend_mode == "soft_light":
            # Soft light blending
            base_norm = base_region / 255.0
            logo_norm = logo_rgb / 255.0
            blended = np.where(
                base_norm < 0.5,
                2 * base_norm * logo_norm + base_norm * base_norm * (1 - 2 * logo_norm),
                2 * base_norm * (1 - logo_norm) + np.sqrt(base_norm) * (2 * logo_norm - 1)
            )
            blended_rgb = (blended * 255.0).astype(np.uint8)
        else:
            blended_rgb = logo_rgb
        
        # Composite with alpha blending
        final_alpha = logo_alpha + base_alpha * (1 - logo_alpha)
        final_rgb = (
            blended_rgb * logo_alpha + 
            base_region * base_alpha * (1 - logo_alpha)
        ) / np.maximum(final_alpha, 0.001)
        
        final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
        final_alpha = np.clip(final_alpha * 255, 0, 255).astype(np.uint8)
        
        # Ensure final_alpha is 2D before adding channel dimension
        if final_alpha.ndim == 3:
            final_alpha = final_alpha.squeeze(axis=2)
        
        # Combine back into RGBA
        final_image = np.concatenate([final_rgb, final_alpha[:, :, np.newaxis]], axis=2)
        final_pil = Image.fromarray(final_image, mode='RGBA')
        
        # Paste back onto result
        result.paste(final_pil, (x_start, y_start), final_pil)
        
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