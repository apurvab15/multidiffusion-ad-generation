"""
Perceptual Contrast Optimization
Enhances generated ads using saliency detection and CV techniques

Features:
- Saliency detection (Grounding DINO + Spectral Residual fallback)
- Adaptive contrast enhancement (CLAHE)
- Edge sharpening with unsharp masking
- Color vibrancy optimization
- Region-aware processing
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class PerceptualOptimizer:
    """
    Optimize ad perceptual quality using:
    1. Saliency detection (to find what draws attention)
    2. Contrast enhancement
    3. Edge sharpening
    4. Semantic region emphasis
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print("[INFO] Loading perceptual optimization models...")
        
        # Try to load Grounding DINO for semantic saliency
        self.has_grounding = False
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            )
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device)
            self.has_grounding = True
            print("  ✓ Grounding DINO loaded (semantic saliency)")
        except Exception as e:
            print(f"  ⚠ Grounding DINO not available: {e}")
            print("  → Using fast spectral residual method instead")
        
        print("[INFO] Perceptual optimizer ready!")
    
    def optimize(
        self,
        image: Image.Image,
        target_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        focus_keywords: str = "product, logo, text",
        enhancement_level: str = "medium"  # low, medium, high
    ) -> Dict:
        """
        Optimize image perceptual quality.
        
        Args:
            image: Generated ad image
            target_regions: List of region bboxes that should draw attention
            focus_keywords: What to emphasize (for saliency)
            enhancement_level: How aggressively to enhance (low/medium/high)
            
        Returns:
            Dictionary with optimized image and analysis
        """
        print("\n[Perceptual Optimization] Starting...")
        
        results = {
            'original': image,
            'analysis': {}
        }
        
        # Get enhancement parameters
        params = self._get_enhancement_params(enhancement_level)
        
        # Step 1: Saliency Analysis
        print("  [1/4] Analyzing visual saliency...")
        saliency_map = self._compute_saliency(image, focus_keywords)
        results['saliency_map'] = saliency_map
        results['analysis']['saliency_score'] = self._score_saliency(
            saliency_map, target_regions
        )
        print(f"    → Saliency score: {results['analysis']['saliency_score']:.3f}")
        
        # Step 2: Contrast Enhancement
        print("  [2/4] Enhancing contrast...")
        enhanced = self._enhance_contrast(image, saliency_map, params)
        results['analysis']['contrast_improvement'] = self._measure_contrast(
            image, enhanced
        )
        print(f"    → Contrast improved by {results['analysis']['contrast_improvement']:.1%}")
        
        # Step 3: Edge Sharpening
        print("  [3/4] Sharpening edges...")
        enhanced = self._sharpen_edges(enhanced, saliency_map, params)
        results['analysis']['sharpness_improvement'] = self._measure_sharpness(
            image, enhanced
        )
        print(f"    → Sharpness improved by {results['analysis']['sharpness_improvement']:.1%}")
        
        # Step 4: Color Vibrancy (in important regions)
        print("  [4/4] Optimizing color vibrancy...")
        enhanced = self._enhance_vibrancy(enhanced, saliency_map, params)
        
        results['optimized'] = enhanced
        
        print("[Perceptual Optimization] Complete!")
        
        return results
    
    def _get_enhancement_params(self, level: str) -> Dict:
        """Get enhancement parameters based on level."""
        params = {
            'low': {
                'contrast_strength': 1.2,
                'sharpen_strength': 0.3,
                'vibrancy_boost': 1.1,
                'clahe_clip': 1.5
            },
            'medium': {
                'contrast_strength': 1.5,
                'sharpen_strength': 0.5,
                'vibrancy_boost': 1.2,
                'clahe_clip': 2.0
            },
            'high': {
                'contrast_strength': 2.0,
                'sharpen_strength': 0.8,
                'vibrancy_boost': 1.3,
                'clahe_clip': 3.0
            }
        }
        return params.get(level, params['medium'])
    
    def _compute_saliency(
        self,
        image: Image.Image,
        focus_keywords: str
    ) -> np.ndarray:
        """
        Compute saliency map highlighting important regions.
        Uses Grounding DINO if available, otherwise spectral residual.
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        if self.has_grounding and focus_keywords:
            # Use Grounding DINO for semantic saliency
            try:
                saliency = self._grounding_dino_saliency(image, focus_keywords)
                if saliency is not None:
                    return saliency
            except Exception as e:
                print(f"    ⚠ Grounding DINO failed: {e}, using fallback")
        
        # Fallback: Spectral Residual Saliency (fast, no ML needed)
        return self._spectral_residual_saliency(img_array)
    
    def _grounding_dino_saliency(
        self,
        image: Image.Image,
        focus_keywords: str
    ) -> Optional[np.ndarray]:
        """Use Grounding DINO for semantic saliency detection."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Process image
        inputs = self.processor(
            images=image,
            text=focus_keywords,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        # Get detection boxes and scores
        target_sizes = torch.tensor([[h, w]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.20,
            text_threshold=0.20,
            target_sizes=target_sizes
        )[0]
        
        # Create saliency map from detections
        saliency_map = np.zeros((h, w), dtype=np.float32)
        
        if len(results['boxes']) == 0:
            return None
        
        for box, score in zip(results['boxes'], results['scores']):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            score_val = score.cpu().item()
            
            # Gaussian blob around detection
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            size = max(x2 - x1, y2 - y1)
            
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[:h, :w]
            
            # Gaussian distribution centered on detection
            sigma = size / 3
            gaussian = np.exp(
                -((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2)
            )
            
            # Accumulate with score weighting
            saliency_map = np.maximum(saliency_map, gaussian * score_val)
        
        # Normalize
        if saliency_map.max() > 0:
            saliency_map /= saliency_map.max()
        
        return saliency_map
    
    def _spectral_residual_saliency(self, img: np.ndarray) -> np.ndarray:
        """
        Fast saliency detection using spectral residual method.
        Paper: "Saliency Detection: A Spectral Residual Approach" (Hou & Zhang, 2007)
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)
        
        # Apply Fourier transform
        fft = np.fft.fft2(gray)
        amplitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Compute log amplitude spectrum
        log_amplitude = np.log(amplitude + 1e-5)
        
        # Spectral residual = log amplitude - smoothed log amplitude
        residual = log_amplitude - cv2.boxFilter(log_amplitude, -1, (3, 3))
        
        # Inverse FFT to get saliency map
        saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase)))
        
        # Smooth with Gaussian
        saliency = cv2.GaussianBlur(saliency, (11, 11), 0)
        
        # Normalize to [0, 1]
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency
    
    def _score_saliency(
        self,
        saliency_map: np.ndarray,
        target_regions: Optional[List[Tuple[int, int, int, int]]]
    ) -> float:
        """
        Score how well saliency aligns with target regions.
        Returns value between 0 and 1.
        """
        if not target_regions or len(target_regions) == 0:
            # No target regions specified, return mean saliency
            return float(np.mean(saliency_map))
        
        # Calculate average saliency in target regions
        target_saliency = []
        
        for x1, y1, x2, y2 in target_regions:
            # Ensure bounds are valid
            h, w = saliency_map.shape
            x1, x2 = max(0, int(x1)), min(w, int(x2))
            y1, y2 = max(0, int(y1)), min(h, int(y2))
            
            if x2 > x1 and y2 > y1:
                region_saliency = saliency_map[y1:y2, x1:x2]
                target_saliency.append(np.mean(region_saliency))
        
        if len(target_saliency) == 0:
            return float(np.mean(saliency_map))
        
        # Return average saliency in target regions
        return float(np.mean(target_saliency))
    
    def _enhance_contrast(
        self,
        image: Image.Image,
        saliency_map: np.ndarray,
        params: Dict
    ) -> Image.Image:
        """
        Enhance contrast adaptively based on saliency.
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        img_array = np.array(image)
        
        # Convert to LAB color space (better for contrast manipulation)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=params['clahe_clip'],
            tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l_channel)
        
        # Resize saliency map to match image
        if saliency_map.shape != l_channel.shape:
            saliency_resized = cv2.resize(
                saliency_map,
                (l_channel.shape[1], l_channel.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            saliency_resized = saliency_map
        
        # Blend based on saliency (more enhancement in salient regions)
        strength = params['contrast_strength']
        blend_factor = saliency_resized * (strength - 1) + 1
        
        l_final = np.clip(
            l_channel * (1 - saliency_resized) + l_enhanced * saliency_resized * blend_factor,
            0, 255
        ).astype(np.uint8)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_final, a_channel, b_channel])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(rgb_enhanced)
    
    def _sharpen_edges(
        self,
        image: Image.Image,
        saliency_map: np.ndarray,
        params: Dict
    ) -> Image.Image:
        """
        Sharpen edges selectively based on saliency.
        Uses unsharp masking.
        """
        img_array = np.array(image).astype(np.float32)
        
        # Create blurred version
        blurred = cv2.GaussianBlur(img_array, (0, 0), 3.0)
        
        # Unsharp mask = original + (original - blurred) * strength
        sharpened = img_array + (img_array - blurred) * params['sharpen_strength']
        sharpened = np.clip(sharpened, 0, 255)
        
        # Resize saliency map
        if saliency_map.shape != img_array.shape[:2]:
            saliency_resized = cv2.resize(
                saliency_map,
                (img_array.shape[1], img_array.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            saliency_resized = saliency_map
        
        # Apply sharpening more strongly in salient regions
        saliency_3ch = saliency_resized[:, :, np.newaxis]
        result = img_array * (1 - saliency_3ch) + sharpened * saliency_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _enhance_vibrancy(
        self,
        image: Image.Image,
        saliency_map: np.ndarray,
        params: Dict
    ) -> Image.Image:
        """
        Enhance color vibrancy in salient regions.
        Increases saturation selectively.
        """
        img_array = np.array(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Resize saliency map
        if saliency_map.shape != s.shape:
            saliency_resized = cv2.resize(
                saliency_map,
                (s.shape[1], s.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            saliency_resized = saliency_map
        
        # Boost saturation in salient regions
        boost = params['vibrancy_boost']
        s_enhanced = s * (1 + (boost - 1) * saliency_resized)
        s_enhanced = np.clip(s_enhanced, 0, 255)
        
        # Merge back
        hsv_enhanced = cv2.merge([h, s_enhanced, v]).astype(np.uint8)
        rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(rgb_enhanced)
    
    def _measure_contrast(
        self,
        original: Image.Image,
        enhanced: Image.Image
    ) -> float:
        """
        Measure contrast improvement.
        Uses RMS contrast metric.
        """
        orig_array = np.array(original.convert('L')).astype(np.float32)
        enhanced_array = np.array(enhanced.convert('L')).astype(np.float32)
        
        # RMS contrast
        orig_contrast = np.std(orig_array) / (np.mean(orig_array) + 1e-5)
        enhanced_contrast = np.std(enhanced_array) / (np.mean(enhanced_array) + 1e-5)
        
        improvement = (enhanced_contrast - orig_contrast) / (orig_contrast + 1e-5)
        
        return float(np.clip(improvement, 0, 1))
    
    def _measure_sharpness(
        self,
        original: Image.Image,
        enhanced: Image.Image
    ) -> float:
        """
        Measure sharpness improvement.
        Uses Laplacian variance.
        """
        orig_array = np.array(original.convert('L'))
        enhanced_array = np.array(enhanced.convert('L'))
        
        # Laplacian operator for edge detection
        orig_laplacian = cv2.Laplacian(orig_array, cv2.CV_64F)
        enhanced_laplacian = cv2.Laplacian(enhanced_array, cv2.CV_64F)
        
        # Variance of Laplacian (higher = sharper)
        orig_sharpness = orig_laplacian.var()
        enhanced_sharpness = enhanced_laplacian.var()
        
        improvement = (enhanced_sharpness - orig_sharpness) / (orig_sharpness + 1e-5)
        
        return float(np.clip(improvement, 0, 1))


if __name__ == "__main__":
    # Test the optimizer
    print("Testing Perceptual Optimizer...")
    
    # Create test image
    from PIL import ImageDraw
    test_img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(test_img)
    
    # Draw a "product" rectangle
    draw.rectangle([150, 150, 350, 350], fill='red', outline='black', width=3)
    draw.text((200, 240), "PRODUCT", fill='white')
    
    # Initialize optimizer
    optimizer = PerceptualOptimizer()
    
    # Run optimization
    results = optimizer.optimize(
        image=test_img,
        target_regions=[(150, 150, 350, 350)],
        focus_keywords="product, red rectangle",
        enhancement_level="medium"
    )
    
    # Save results
    results['original'].save('test_original.png')
    results['optimized'].save('test_optimized.png')
    
    # Save saliency visualization
    saliency_viz = (results['saliency_map'] * 255).astype(np.uint8)
    Image.fromarray(saliency_viz).save('test_saliency.png')
    
    print("\n✓ Test complete!")
    print(f"  Saliency score: {results['analysis']['saliency_score']:.3f}")
    print(f"  Contrast improvement: {results['analysis']['contrast_improvement']:.1%}")
    print(f"  Sharpness improvement: {results['analysis']['sharpness_improvement']:.1%}")
    print("\nCheck test_*.png files for results.")