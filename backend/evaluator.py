"""
Evaluation Module
Computes metrics for generated advertisements
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2


class AdEvaluator:
    """Comprehensive evaluation metrics for generated ads."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load CLIP
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP loaded!")
    
    def evaluate_ad(
        self,
        image: Image.Image,
        prompt: str,
        logo_colors: Optional[List[str]] = None,
        original_logo: Optional[Image.Image] = None,
        logo_bbox: Optional[Tuple] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated ad.
        """
        metrics = {}
        
        # CLIP Score
        metrics['clip_score'] = self.compute_clip_score(image, prompt)
        
        # Aesthetic Score
        metrics['aesthetic_score'] = self.compute_aesthetic_score(image)
        
        # Brand Consistency
        if logo_colors:
            metrics['brand_consistency'] = self.compute_brand_consistency(
                image, logo_colors
            )
        
        # Logo Preservation
        if original_logo and logo_bbox:
            metrics['logo_preservation'] = self.compute_logo_preservation(
                image, original_logo, logo_bbox
            )
        
        # Layout Score
        metrics['layout_score'] = self.compute_layout_score(image)
        
        return metrics
    
    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity."""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        normalized_score = (similarity + 1) / 2
        return normalized_score
    
    def compute_aesthetic_score(self, image: Image.Image) -> float:
        """Compute aesthetic score using heuristics."""
        img_array = np.array(image)
        
        # Color diversity
        hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten() / hist.sum()
        color_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        color_score = min(color_entropy / 10, 1.0)
        
        # Contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = gray.std() / 128.0
        contrast_score = min(contrast, 1.0)
        
        # Composition
        h, w = gray.shape
        thirds_h = [h // 3, 2 * h // 3]
        thirds_w = [w // 3, 2 * w // 3]
        
        edges = cv2.Canny(gray, 100, 200)
        power_points_density = sum([
            edges[y-h//10:y+h//10, x-w//10:x+w//10].sum()
            for y in thirds_h for x in thirds_w
        ])
        total_edges = edges.sum()
        composition_score = power_points_density / (total_edges + 1)
        composition_score = min(composition_score * 2, 1.0)
        
        aesthetic_score = (color_score + contrast_score + composition_score) / 3
        return aesthetic_score
    
    def compute_brand_consistency(
        self,
        image: Image.Image,
        logo_colors: List[str]
    ) -> float:
        """Compute brand color consistency."""
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        gen_colors = kmeans.cluster_centers_
        
        logo_colors_rgb = [self._hex_to_rgb(c) for c in logo_colors[:5]]
        
        logo_lab = rgb2lab(np.array(logo_colors_rgb).reshape(1, -1, 3))
        gen_lab = rgb2lab(gen_colors.reshape(1, -1, 3))
        
        min_distances = []
        for gen_color in gen_lab[0]:
            distances = [deltaE_ciede2000(gen_color, logo_color) 
                        for logo_color in logo_lab[0]]
            min_distances.append(min(distances))
        
        avg_distance = np.mean(min_distances)
        consistency = 1.0 / (1.0 + avg_distance / 50.0)
        
        return consistency
    
    def compute_logo_preservation(
        self,
        image: Image.Image,
        original_logo: Image.Image,
        logo_bbox: Tuple
    ) -> float:
        """Compute logo preservation using SSIM."""
        img_array = np.array(image)
        logo_array = np.array(original_logo)
        
        x, y = logo_bbox[:2]
        h, w = logo_array.shape[:2]
        extracted_logo = img_array[y:y+h, x:x+w]
        
        if extracted_logo.shape != logo_array.shape:
            extracted_logo = cv2.resize(extracted_logo, (logo_array.shape[1], logo_array.shape[0]))
        
        score = ssim(extracted_logo, logo_array, channel_axis=2)
        return score
    
    def compute_layout_score(self, image: Image.Image) -> float:
        """Analyze layout quality."""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = gray.shape
        left = edges[:, :w//2].sum()
        right = edges[:, w//2:].sum()
        top = edges[:h//2, :].sum()
        bottom = edges[h//2:, :].sum()
        
        balance_lr = 1 - abs(left - right) / (left + right + 1)
        balance_tb = 1 - abs(top - bottom) / (top + bottom + 1)
        
        layout_score = (balance_lr + balance_tb) / 2
        return layout_score
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple:
        """Convert hex to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    evaluator = AdEvaluator()
    
    test_image = Image.new('RGB', (512, 512), color='blue')
    test_prompt = "modern tech advertisement"
    
    metrics = evaluator.evaluate_ad(test_image, test_prompt)
    print("Metrics:", metrics)