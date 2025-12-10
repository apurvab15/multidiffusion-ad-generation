"""
Logo Color Extraction Module
Extracts dominant colors and generates complementary palettes
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import colorsys


def extract_dominant_colors(image_path: str, n_colors: int = 5) -> List[str]:
    """
    Extract dominant colors from logo using K-means clustering.
    
    Args:
        image_path: Path to logo image
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of hex color codes
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Remove white background (if present)
    mask = np.all(img > 240, axis=2)
    pixels = img[~mask].reshape(-1, 3)
    
    if len(pixels) == 0:
        pixels = img.reshape(-1, 3)
    
    # Perform K-means clustering
    n_clusters = min(n_colors, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get colors sorted by frequency
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Sort by frequency
    sorted_indices = np.argsort(-counts)
    sorted_colors = colors[sorted_indices]
    
    # Convert to hex
    hex_colors = [rgb_to_hex(tuple(color)) for color in sorted_colors]
    
    return hex_colors


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_complementary_colors(hex_colors: List[str]) -> List[str]:
    """Generate complementary color palette."""
    complementary = []
    
    for hex_color in hex_colors[:3]:
        rgb = hex_to_rgb(hex_color)
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        
        # Complementary: rotate hue by 180 degrees
        comp_hsv = ((hsv[0] + 0.5) % 1.0, hsv[1], hsv[2])
        comp_rgb = colorsys.hsv_to_rgb(*comp_hsv)
        comp_rgb = tuple(int(c * 255) for c in comp_rgb)
        
        complementary.append(rgb_to_hex(comp_rgb))
    
    return complementary


def describe_color_palette(hex_colors: List[str]) -> str:
    """Generate natural language description of color palette."""
    descriptions = []
    
    for hex_color in hex_colors[:3]:
        rgb = hex_to_rgb(hex_color)
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        
        # Determine color name
        hue = hsv[0] * 360
        sat = hsv[1]
        val = hsv[2]
        
        if sat < 0.1:
            if val > 0.9:
                color_name = "white"
            elif val < 0.1:
                color_name = "black"
            else:
                color_name = "gray"
        else:
            if hue < 30 or hue >= 330:
                color_name = "red"
            elif hue < 90:
                color_name = "yellow"
            elif hue < 150:
                color_name = "green"
            elif hue < 210:
                color_name = "cyan"
            elif hue < 270:
                color_name = "blue"
            else:
                color_name = "magenta"
        
        # Add intensity descriptor
        if sat > 0.6 and val > 0.6:
            intensity = "vibrant"
        elif sat < 0.3:
            intensity = "muted"
        else:
            intensity = "soft"
        
        descriptions.append(f"{intensity} {color_name}")
    
    return ", ".join(descriptions)


def analyze_logo_colors(image_path: str) -> dict:
    """
    Comprehensive color analysis of logo.
    
    Returns:
        Dictionary with color information
    """
    dominant_colors = extract_dominant_colors(image_path, n_colors=5)
    complementary = get_complementary_colors(dominant_colors)
    
    return {
        'dominant_colors': dominant_colors,
        'primary_color': dominant_colors[0],
        'complementary_colors': complementary,
        'color_description': describe_color_palette(dominant_colors)
    }


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        test_logo = sys.argv[1]
        result = analyze_logo_colors(test_logo)
        print("Color Analysis:")
        print(f"  Primary: {result['primary_color']}")
        print(f"  Palette: {result['dominant_colors']}")
        print(f"  Description: {result['color_description']}")
    else:
        print("Usage: python color_extractor.py <logo_path>")