import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model settings
DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

# Generation settings
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 768
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Region settings
LOGO_REGION_SIZE = (128, 128)  # Width, height for logo

# Evaluation settings
CLIP_MODEL = "ViT-B/32"

# API keys
GEMINI_API_KEY = "AIzaSyDFphVdNdeh6ClLlr1532BNl081rt-FRdQ"