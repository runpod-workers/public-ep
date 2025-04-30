import torch
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
import os

# Path where models will be saved
model_base_path = os.environ.get("MODEL_BASE_PATH", "/models")
os.makedirs(model_base_path, exist_ok=True)

print("Downloading SD3 model and components...")

# Download SD3 pipeline
model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Download tiny autoencoder
tiny_vae = AutoencoderTiny.from_pretrained('madebyollin/taesd3', torch_dtype=torch.float16)

# Save models to local directory
sd3_path = os.path.join(model_base_path, "sd3-medium")
vae_path = os.path.join(model_base_path, "taesd3")

print(f"Saving SD3 pipeline to {sd3_path}...")
pipeline.save_pretrained(sd3_path)

print(f"Saving tiny autoencoder to {vae_path}...")
tiny_vae.save_pretrained(vae_path)

print("Models successfully downloaded and saved")