import torch
import base64
import io
from diffusers import FluxPipeline, FluxTransformer2DModel
from PIL import Image
from utils import upload_to_r2
import uuid
from datetime import datetime, timedelta
from nanoid import generate

class FluxSchnellGenerator:
    def __init__(self):
        self.pipe = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the Flux-schnell model with optimizations for both memory and speed."""
        if self.initialized:
            return
        
        # Model path
        flux_model_path = "/models/flux-schnell"
        fp8_transformer_path = "/models/fp8_transformer/flux1-schnell-fp8-e4m3fn.safetensors"
        
        # Load the Flux pipeline with optimizations
        self.pipe = FluxPipeline.from_pretrained(
            flux_model_path,
            transformer=None,
            # text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        )
        
        transformer = FluxTransformer2DModel.from_single_file(
            fp8_transformer_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()
        
        self.initialized = True
        
    def generate(self, input_data):
        """Generate an image based on the input and return as base64 string."""
        if not self.initialized:
            self.initialize()
        
        # Extract parameters from input data
        prompt = input_data.get("prompt", "A photo of a dog and a cat")
        negative_prompt = input_data.get("negative_prompt", "")
        height = input_data.get("height", 768)
        width = input_data.get("width", 1360)
        steps = input_data.get("num_inference_steps", 4)  # Default for schnell is lower
        guidance_scale = input_data.get("guidance_scale", 0.0)  # Default for schnell is 0
        seed = input_data.get("seed", None)
        max_sequence_length = input_data.get("max_sequence_length", 256)  # Schnell-specific
        img_format = input_data.get("image_format", "png")
        if img_format not in ["png", "jpeg", "jpg"]:
            raise ValueError("Invalid image format. Supported formats are 'png' and 'jpg'.")
        
        # Set up generator if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        
        # Generate the image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,  # Schnell-specific parameter
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        now = datetime.now()
        filename = f"gen-images/{now.month}/{now.day}/{generate(size=10)}/{uuid.uuid4()}.{img_format}"
        if img_format == "png":
            image.save(buffered, format="PNG")
            content_type = "image/png"
        elif img_format == "jpeg" or img_format == "jpg":
            image.convert("RGB")
            # Convert to RGB for JPEG
            image.save(buffered, format="JPEG")
            content_type = "image/jpeg"
        buffered.seek(0)
        url = upload_to_r2(buffered.getvalue(), filename, content_type)
        # img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return url