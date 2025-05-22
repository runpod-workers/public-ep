import runpod
from txt2img_sd3 import SD3Generator

sd3 = SD3Generator()


async def handler(job):
    job_input = job.get("input")
    if not job_input:
        raise ValueError("No input provided")
    image_format = job_input.get("image_format", "png")
    if image_format not in ["png", "jpeg", "jpg"]:
        raise ValueError("Invalid image format. Supported formats are 'png' and 'jpg'.")
    
    mime_type = "image/jpeg" if image_format in ["jpeg", "jpg"] else "image/png"
    try:
        img_url = sd3.generate(job_input)
        return {
            "status": "success",
            "message": "Image generated successfully",
            "image_url": img_url,
            # "image": base64_img,
            # "data_url": f"data:{mime_type};base64,{base64_img}",
        }
    except RuntimeError as e:
        return {
            "status": "error",
            "message": str(e),
        }

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: 500,
        "return_aggregate_stream": True,
    }
)