import torch
from diffusers import FluxPipeline, AutoencoderTiny
import os

# Path where models will be saved
model_base_path = os.environ.get("MODEL_BASE_PATH", "/models")
os.makedirs(model_base_path, exist_ok=True)

print("Downloading flux model and components...")

# Download SD3 pipeline
model_id = 'black-forest-labs/FLUX.1-schnell'
pipeline =  FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   
)

# Download tiny autoencoder
tiny_vae = AutoencoderTiny.from_pretrained('madebyollin/taesd3', torch_dtype=torch.float16)

# Save models to local directory
flux_path = os.path.join(model_base_path, "flux-schenll")
vae_path = os.path.join(model_base_path, "taesd3")

print(f"Saving flux pipeline to {flux_path}...")
pipeline.save_pretrained(flux_path)

print(f"Saving tiny autoencoder to {vae_path}...")
tiny_vae.save_pretrained(vae_path)

print("Models successfully downloaded and saved")