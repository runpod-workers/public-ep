import runpod
from txt2img_flux_dev import FluxDevGenerator



flux_dev = FluxDevGenerator()


async def handler(job):
    job_input = job.get("input")
    if not job_input:
        raise ValueError("No input provided")
    
    try:
        base64_img = flux_dev.generate(job_input)
        return {
            "status": "success",
            "message": "Image generated successfully",
            "image": base64_img,
            "data_url": f"data:image/png;base64,{base64_img}",
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