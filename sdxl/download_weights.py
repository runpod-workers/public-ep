import torch
from diffusers import StableDiffusionXLPipeline


def fetch_pretrained_model(model_class, model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def download_model_weights():
    """
    Downloads SDXL model weights for Docker image baking.
    """
    print("Downloading SDXL model weights...")
    
    # Download with bfloat16 precision (matching runtime usage)
    common_args = {
        "torch_dtype": torch.bfloat16,
        "use_safetensors": True,
        "add_watermarker": False,
    }

    # Download the main pipeline
    pipe = fetch_pretrained_model(
        StableDiffusionXLPipeline,
        "stabilityai/stable-diffusion-xl-base-1.0",
        **common_args,
    )
    
    print("✓ SDXL base model downloaded successfully")
    
    # Clean up to save space during build
    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("✓ Model weights cached for runtime use")


if __name__ == "__main__":
    download_model_weights()