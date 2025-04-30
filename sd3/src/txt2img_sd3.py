import torch
import base64
import io
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image

class SD3Generator:
    def __init__(self):
        self.pipe = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the SD3 model with optimizations for both memory and speed."""
        if self.initialized:
            return
        
        # Model selection
        # model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        sd3_model_path = "/models/sd3-medium"
        taesd3_path = "/models/taesd3"
        
        # Memory optimization approach 1: Use 8-bit quantization for T5 encoder
        try:
            # Setup quantization config
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Create pipeline with the quantized T5 encoder and device_map to handle placement
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                sd3_model_path,
                quantization_config={"text_encoder_3": quantization_config},
                torch_dtype=torch.float16,
                device_map="auto"  # Let diffusers handle device placement
            )
            
            # Speed optimization: Use tiny autoencoder
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd3_path, 
                torch_dtype=torch.float16
            )
            
            # Only apply memory format to non-quantized components
            if hasattr(self.pipe, "transformer") and not getattr(self.pipe.transformer, "is_quantized", False):
                self.pipe.transformer.to(memory_format=torch.channels_last)
            
            if hasattr(self.pipe, "vae") and not getattr(self.pipe.vae, "is_quantized", False):
                self.pipe.vae.to(memory_format=torch.channels_last)
            
        except Exception as e:
            print(f"Failed to load with quantization: {e}")
            print("Falling back to simpler model configuration")
            
            # Memory optimization approach 2: Use without T5 encoder
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                sd3_model_path,
                text_encoder_3=None,  # Remove T5 encoder
                tokenizer_3=None,
                torch_dtype=torch.float16
            )
            
            self.pipe.to("cuda")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd3_path, 
                torch_dtype=torch.float16
            ).to("cuda")
        
        self.initialized = True
        
    def generate(self, input_data):
        """Generate an image based on the input and return as base64 string."""
        if not self.initialized:
            self.initialize()
        
        # Extract parameters from input data
        prompt = input_data.get("prompt", "A photo of a cat")
        negative_prompt = input_data.get("negative_prompt", "")
        height = input_data.get("height", 768)
        width = input_data.get("width", 768)
        steps = input_data.get("num_inference_steps", 20)
        guidance_scale = input_data.get("guidance_scale", 5.0)
        
        # Generate the image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

# Create a singleton instance
sd3 = SD3Generator()