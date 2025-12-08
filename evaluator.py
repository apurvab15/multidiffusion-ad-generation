import torch
import clip
import numpy as np
from PIL import Image
from typing import Dict, List
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2


class AdEvaluator:
    """Comprehensive evaluation metrics for generated ads."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load CLIP for text-image similarity
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP loaded!")
    
    def evaluate_ad(
        self,
        image: Image.Image,
        prompt: str,
        logo_colors: List[str] = None,
        original_logo: Image.Image = None,
        logo_bbox: tuple = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated ad.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. CLIP Score (text-image alignment)
        metrics['clip_score'] = self.compute_clip_score(image, prompt)
        
        # 2. Aesthetic Score (using simple heuristics)
        metrics['aesthetic_score'] = self.compute_aesthetic_score(image)
        
        # 3. Brand Consistency (if logo colors provided)
        if logo_colors:
            metrics['brand_consistency'] = self.compute_brand_consistency(
                image, logo_colors
            )
        
        # 4. Logo Preservation (if original logo provided)
        if original_logo and logo_bbox:
            metrics['logo_preservation'] = self.compute_logo_preservation(
                image, original_logo, logo_bbox
            )
        
        # 5. Layout Score (simple composition analysis)
        metrics['layout_score'] = self.compute_layout_score(image)
        
        return metrics
    
    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP similarity between image and text."""
        
        # Preprocess image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize text
        text_input = clip.tokenize([prompt]).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        # Scale to 0-1 range (CLIP scores are typically 0.2-0.4)
        # Normalize to make more interpretable
        normalized_score = (similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        return normalized_score
    
    def compute_aesthetic_score(self, image: Image.Image) -> float:
        """
        Compute aesthetic score using simple heuristics.
        
        In a full implementation, you'd use LAION aesthetic predictor.
        This is a simplified version based on:
        - Color diversity
        - Contrast
        - Composition balance
        """
        img_array = np.array(image)
        
        # 1. Color diversity (entropy of color histogram)
        hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten() / hist.sum()
        color_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        color_score = min(color_entropy / 10, 1.0)  # Normalize
        
        # 2. Contrast (std of luminance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = gray.std() / 128.0  # Normalize
        contrast_score = min(contrast, 1.0)
        
        # 3. Composition (rule of thirds)
        h, w = gray.shape
        thirds_h = [h // 3, 2 * h // 3]
        thirds_w = [w // 3, 2 * w // 3]
        
        # Check if interesting content is at power points
        edges = cv2.Canny(gray, 100, 200)
        power_points_density = sum([
            edges[y-h//10:y+h//10, x-w//10:x+w//10].sum()
            for y in thirds_h for x in thirds_w
        ])
        total_edges = edges.sum()
        composition_score = power_points_density / (total_edges + 1)
        composition_score = min(composition_score, 1.0)

        # Combine scores
        aesthetic_score = (color_score + contrast_score + composition_score) / 3
        return aesthetic_score
    

def compute_brand_consistency(
    self,
    image: Image.Image,
    logo_colors: List[str]
) -> float:
    """
    Compute how well the image matches brand colors.
    """
    img_array = np.array(image)
    
    # Extract dominant colors from generated image
    pixels = img_array.reshape(-1, 3)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    gen_colors = kmeans.cluster_centers_
    
    # Convert logo colors to RGB
    logo_colors_rgb = [self._hex_to_rgb(c) for c in logo_colors[:5]]
    
    # Compute minimum color distance in LAB space
    logo_lab = rgb2lab(np.array(logo_colors_rgb).reshape(1, -1, 3))
    gen_lab = rgb2lab(gen_colors.reshape(1, -1, 3))
    
    min_distances = []
    for gen_color in gen_lab[0]:
        distances = [deltaE_ciede2000(gen_color, logo_color) 
                    for logo_color in logo_lab[0]]
        min_distances.append(min(distances))
    
    # Lower distance = higher consistency
    avg_distance = np.mean(min_distances)
    consistency = 1.0 / (1.0 + avg_distance / 50.0)  # Normalize
    
    return consistency

def compute_logo_preservation(
    self,
    image: Image.Image,
    original_logo: Image.Image,
    logo_bbox: tuple
) -> float:
    """Compute how well the logo is preserved using SSIM."""
    
    img_array = np.array(image)
    logo_array = np.array(original_logo)
    
    # Extract logo region from generated image
    x, y = logo_bbox[:2]
    h, w = logo_array.shape[:2]
    extracted_logo = img_array[y:y+h, x:x+w]
    
    # Resize if needed
    if extracted_logo.shape != logo_array.shape:
        extracted_logo = cv2.resize(extracted_logo, (logo_array.shape[1], logo_array.shape[0]))
    
    # Compute SSIM
    score = ssim(extracted_logo, logo_array, channel_axis=2)
    
    return score

def compute_layout_score(self, image: Image.Image) -> float:
    """
    Analyze layout quality using saliency and balance.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Simple saliency using edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Check balance (left vs right, top vs bottom)
    h, w = gray.shape
    left = edges[:, :w//2].sum()
    right = edges[:, w//2:].sum()
    top = edges[:h//2, :].sum()
    bottom = edges[h//2:, :].sum()
    
    balance_lr = 1 - abs(left - right) / (left + right + 1)
    balance_tb = 1 - abs(top - bottom) / (top + bottom + 1)
    
    layout_score = (balance_lr + balance_tb) / 2
    
    return layout_score

def _hex_to_rgb(self, hex_color: str) -> tuple:
    """Convert hex to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))