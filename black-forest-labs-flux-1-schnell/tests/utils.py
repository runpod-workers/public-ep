
import aiohttp
import asyncio
import os
from PIL import Image
import io
import time



API_URL = f"https://api.runpod.ai/v2/{os.getenv('ENDPOINT_ID')}/run"
API_KEY = os.getenv("RUNPOD_API_KEY")  # Store API key in environment variable


# Helper function to validate image data
def is_valid_image(image_data):
    try:
        Image.open(io.BytesIO(image_data))
        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False


async def make_api_request(payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    start_time = time.time()
    print(f"Sending request to {API_URL}...")
    
    async with aiohttp.ClientSession() as session:
        # Initial request to start the job
        async with session.post(API_URL, json=payload, headers=headers, timeout=120) as response:
            initial_response = await response.json()
            
            # Check for immediate errors
            if response.status != 200:
                elapsed_time = time.time() - start_time
                print(f"Request failed with status {response.status} in {elapsed_time:.2f} seconds")
                return {
                    "status_code": response.status,
                    "elapsed_time": elapsed_time,
                    "json_data": initial_response,
                    "state": "FAILED"
                }
            
            # Get the request_id (job ID) from the response
            request_id = initial_response.get("id")
            if not request_id:
                elapsed_time = time.time() - start_time
                print(f"Request failed in {elapsed_time:.2f} seconds - no job id received")
                return {
                    "status_code": response.status,
                    "elapsed_time": elapsed_time,
                    "json_data": initial_response,
                    "state": "FAILED"
                }
            
            print(f"Job submitted with ID: {request_id}")
        
        # Poll until COMPLETED or FAILED
        status_url = f"{API_URL.replace('/run', '')}/status/{request_id}"
        
        while True:
            async with session.get(status_url, headers=headers, timeout=30) as status_response:
                status_data = await status_response.json()
                current_state = status_data.get("status", "UNKNOWN")
                
                elapsed_time = time.time() - start_time
                print(f"Current state: {current_state}, elapsed time: {elapsed_time:.2f} seconds")
                
                # Return on terminal states
                if current_state in ["COMPLETED", "FAILED", "CANCELLED"]:
                    # If completed, format the response to match what the tests expect
                    if current_state == "COMPLETED":
                        # Extract the output data from the nested structure RunPod uses
                        output = status_data.get("output", {})
                        
                        # Format to match what the tests expect
                        formatted_data = {
                            "status": "success",
                            "image": output.get("image", ""),
                            "data_url": output.get("data_url", "")
                        }
                        
                        return {
                            "status_code": status_response.status,
                            "elapsed_time": elapsed_time,
                            "json_data": formatted_data,
                            "state": current_state
                        }
                    else:
                        # For failed or cancelled states
                        return {
                            "status_code": status_response.status,
                            "elapsed_time": elapsed_time,
                            "json_data": {"status": "error", "message": status_data.get("error", "Unknown error")},
                            "state": current_state
                        }
                
                # Wait before polling again - exponential backoff might be better
                # for production but keeping simple for tests
                await asyncio.sleep(2)
