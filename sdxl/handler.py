import os
import base64
import io

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from utils import upload_to_r2
import uuid
from nanoid import generate
from datetime import datetime

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Base Pipeline from cache using identifier
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        base_pipe.enable_xformers_memory_efficient_attention()

        return base_pipe

    def load_refiner(self):
        # Load VAE from cache using identifier
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Load Refiner Pipeline from cache using identifier
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        refiner_pipe.enable_xformers_memory_efficient_attention()

        return refiner_pipe

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()


MODELS = ModelHandler()



def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    import json, pprint

    # Log the exact structure RunPod delivers so we can see every nesting level.
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Original (strict) behaviour – assume the expected single wrapper exists.
    # -------------------------------------------------------------------------
    job_input = job["input"]
    img_format = job_input.get("image_format", "png")

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback

        print("[generate_image] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        # Re-raise so RunPod registers the failure (but logs are now visible).
        raise

    print("[generate_image] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    starting_image = job_input["image_url"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    # Assuming device_map puts model on cuda if available
    generator = torch.Generator("cuda").manual_seed(job_input["seed"])

    MODELS.base.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.base.scheduler.config
    )

    if starting_image:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input["prompt"],
            num_inference_steps=job_input["refiner_inference_steps"],
            strength=job_input["strength"],
            image=init_image,
            generator=generator,
        ).images
    else:
        # Generate latent image using pipe
        image = MODELS.base(
            prompt=job_input["prompt"],
            negative_prompt=job_input["negative_prompt"],
            height=job_input["height"],
            width=job_input["width"],
            num_inference_steps=job_input["num_inference_steps"],
            guidance_scale=job_input["guidance_scale"],
            denoising_end=job_input["high_noise_frac"],
            output_type="latent",
            num_images_per_prompt=job_input["num_images"],
            generator=generator,
        ).images

        try:
            output = MODELS.refiner(
                prompt=job_input["prompt"],
                num_inference_steps=job_input["refiner_inference_steps"],
                strength=job_input["strength"],
                image=image,
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
                "refresh_worker": True,
            }
    # Convert to base64
        buffered = io.BytesIO()
        now = datetime.now()
        filename = f"gen-images/{now.month}/{now.day}/{generate(size=10)}/{uuid.uuid4()}.{img_format}"
        if img_format == "png":
            output.save(buffered, format="PNG")
            content_type = "image/png"
        elif img_format == "jpeg" or img_format == "jpg":
            output.convert("RGB")
            # Convert to RGB for JPEG
            output.save(buffered, format="JPEG")
            content_type = "image/jpeg"
        buffered.seek(0)
    
    
    url = upload_to_r2(buffered.getvalue(), filename, content_type)


    results = {
        "status": "success",
        "message": "Image generated successfully",
        "image_url": url,
    }
    
    return results


runpod.serverless.start({"handler": generate_image})