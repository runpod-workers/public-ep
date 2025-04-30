import torch
import base64
import io
from diffusers import FluxPipeline
from PIL import Image

class FluxDevGenerator:
    def __init__(self):
        self.pipe = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the Flux-dev model with optimizations for both memory and speed."""
        if self.initialized:
            return
        
        # Model path
        flux_model_path = "/models/flux-dev"
        
        try:
            # Create pipeline with device_map to handle placement
            self.pipe = FluxPipeline.from_pretrained(
                flux_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"  # Let diffusers handle device placement
            )
            
            # Enable optimizations
            if hasattr(self.pipe, "vae"):
                self.pipe.enable_vae_slicing()
                self.pipe.enable_vae_tiling()
                
            # Apply memory format optimizations
            if hasattr(self.pipe, "transformer") and not getattr(self.pipe.transformer, "is_quantized", False):
                self.pipe.transformer.to(memory_format=torch.channels_last)
            
            if hasattr(self.pipe, "vae") and not getattr(self.pipe.vae, "is_quantized", False):
                self.pipe.vae.to(memory_format=torch.channels_last)
            
        except Exception as e:
            print(f"Failed to load with auto device mapping: {e}")
            print("Falling back to simpler model configuration")
            
            # Fallback approach: Basic CPU offloading
            self.pipe = FluxPipeline.from_pretrained(
                flux_model_path,
                torch_dtype=torch.bfloat16
            )
            
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        
        self.initialized = True
        
    def generate(self, input_data):
        """Generate an image based on the input and return as base64 string."""
        if not self.initialized:
            self.initialize()
        
        # Extract parameters from input data
        prompt = input_data.get("prompt", "A photo of a cat")
        negative_prompt = input_data.get("negative_prompt", "")
        height = input_data.get("height", 768)
        width = input_data.get("width", 1360)
        steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 3.5)
        seed = input_data.get("seed", None)
        
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
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str