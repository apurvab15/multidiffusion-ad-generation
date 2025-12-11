import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model settings
DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

# Generation settings
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_INFERENCE_STEPS = 75
GUIDANCE_SCALE = 7.5

# Region settings
LOGO_REGION_SIZE = (128, 128)  # Width, height for logo

# Evaluation settings
CLIP_MODEL = "ViT-B/32"

# GPU settings
# Set to "cuda", "cuda:0", "cuda:1", etc. for specific GPUs, or "cpu"
# Can also be overridden via --device flag or CUDA_VISIBLE_DEVICES env var
DEVICE = os.getenv("DEVICE", "cuda")  # Default: cuda (uses first available GPU)

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")