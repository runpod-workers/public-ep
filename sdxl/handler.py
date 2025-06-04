import os
import io
import time

import torch
from diffusers import StableDiffusionXLPipeline


import runpod
from runpod.serverless.utils.rp_validator import validate
from utils import upload_to_r2
import uuid
from nanoid import generate
from datetime import datetime
from utils import calculate_cost
from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


class SimpleModelHandler:
    def __init__(self):
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load SDXL pipeline for text-to-image generation"""
        print("Loading SDXL pipeline...")
        start_time = time.time()
        
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
            add_watermarker=False,
        ).to("cuda")
        
        # # Enable memory efficient attention
        # self.pipeline.enable_xformers_memory_efficient_attention()
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")


MODELS = SimpleModelHandler()


@torch.inference_mode()
def generate_image(job):
    """Generate image from text prompt"""
    
    job_input = job["input"]
    img_format = job_input.get("image_format", "png")

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    
    job_input = validated_input["validated_input"]

    # Set random seed if not provided
    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Setup generator
    generator = torch.Generator("cuda").manual_seed(job_input["seed"])

    # Generate image
    try:
        start_time = time.time()
        
        images = MODELS.pipeline(
            prompt=job_input["prompt"],
            negative_prompt=job_input.get("negative_prompt", ""),
            height=job_input.get("height", 1024),
            width=job_input.get("width", 1024),
            num_inference_steps=job_input.get("num_inference_steps", 30),
            guidance_scale=job_input.get("guidance", 7.5),
            num_images_per_prompt=job_input.get("num_images", 1),
            generator=generator,
        ).images

        generation_time = time.time() - start_time
        print(f"Image generated in {generation_time:.2f} seconds")

    except RuntimeError as err:
        return {
            "error": f"Generation failed: {err}",
            "refresh_worker": True,
        }

    # Save and upload image
    try:
        img = images[0]
        buffered = io.BytesIO()
        
        # Generate filename
        now = datetime.now()
        filename = f"gen-images/{now.month}/{now.day}/{generate(size=10)}/{uuid.uuid4()}.{img_format}"
        
        # Save with format
        if img_format == "png":
            img.save(buffered, format="PNG")
            content_type = "image/png"
        elif img_format in ["jpeg", "jpg"]:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buffered, format="JPEG", quality=95)
            content_type = "image/jpeg"
        
        buffered.seek(0)
        url = upload_to_r2(buffered.getvalue(), filename, content_type)
        
        return {
            "image_url": url,
            "generation_time": f"{generation_time:.2f}s",
            "cost": calculate_cost(job_input.get("num_images", 1)),
            "seed": job_input["seed"]
        }

    except Exception as err:
        return {"error": f"Upload failed: {err}"}


runpod.serverless.start({"handler": generate_image})