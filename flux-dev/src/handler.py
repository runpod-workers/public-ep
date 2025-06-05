import runpod
from txt2img_flux_dev import FluxDevGenerator
from utils import calculate_cost


flux_dev = FluxDevGenerator()

async def handler(job):
    job_input = job.get("input")
    if not job_input:
        raise ValueError("No input provided")
    image_format = job_input.get("image_format", "png")
    if image_format not in ["png", "jpeg", "jpg"]:
        raise ValueError("Invalid image format. Supported formats are 'png' and 'jpg'.")
    
    mime_type = "image/jpeg" if image_format in ["jpeg", "jpg"] else "image/png"
    try:
        img_url = flux_dev.generate(job_input)
        return {
            "image_url": img_url,
            "cost": calculate_cost(
                job_input["width"], job_input["height"]
            ),
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
        "return_aggregate_stream": True,
    }
)
