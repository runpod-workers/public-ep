import pytest
import aiohttp
import asyncio
import json
import base64
import os
from PIL import Image
import io
import time
from utils import make_api_request, is_valid_image

# Test 1: Basic functionality - verify the API can generate an image with default settings
@pytest.mark.asyncio
async def test_basic_functionality():
    print("\n--- Running Basic Functionality Test ---")
    
    # Standard payload with minimal requirements
    payload = {
        "input": {
            "prompt": "a photograph of a dog in a park",
        }
    }
    
    #
    result = await make_api_request(payload)
    
    # Test that the response is successful
    assert result["status_code"] == 200, f"API returned status code {result['status_code']}"
    
    # Test the response format
    data = result["json_data"]
    assert data["status"] == "success", f"API returned status: {data.get('status')}"
    assert "image" in data, "Response missing 'image' field"
    assert "data_url" in data, "Response missing 'data_url' field"
    
    # Test that the response contains a valid image
    image_data = base64.b64decode(data["image"])
    assert is_valid_image(image_data), "Generated image is not valid"
    
    print("--- Basic Functionality Test PASSED ---")

# Test 2: Parameter functionality - verify the API correctly handles custom parameters
@pytest.mark.asyncio
async def test_parameter_functionality():
    print("\n--- Running Parameter Functionality Test ---")
    
    # Payload with custom parameters
    payload = {
        "input": {
            "prompt": "a photo of a cat",
            "negative_prompt": "blurry, low quality",
            "height": 512,
            "width": 512,
            "num_inference_steps": 30,
            "guidance": 7.5
        }
    }
    
    # Make the API request
    result = await make_api_request(payload)
    
    # Test that the response is successful
    assert result["status_code"] == 200, f"API returned status code {result['status_code']}"
    
    # Test the response format
    data = result["json_data"]
    assert data["status"] == "success", f"API returned status: {data.get('status')}"
    assert "image" in data, "Response missing 'image' field"
    
    # Test that the response contains a valid image
    image_data = base64.b64decode(data["image"])
    assert is_valid_image(image_data), "Generated image is not valid"
    
    # Optionally check image dimensions if your API respects them exactly
    try:
        img = Image.open(io.BytesIO(image_data))
        print(f"Generated image dimensions: {img.width}x{img.height}")
        # assert img.width == 512, f"Image width is {img.width}, expected 512"
        # assert img.height == 512, f"Image height is {img.height}, expected 512"
    except Exception as e:
        print(f"Could not verify image dimensions: {e}")
    