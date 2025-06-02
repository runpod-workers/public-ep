import io
import uuid
from PIL import Image
import boto3
from botocore.client import Config
import os
from datetime import datetime, timedelta
from nanoid import generate

# === Upload function ===
def upload_to_r2(image_bytes, filename, content_type):
    session = boto3.session.Session()
    expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    client = session.client(
        's3',
        region_name='auto',
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),  
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),  
        config=Config(signature_version='s3v4')
    )

    bucket_name = os.getenv('R2_BUCKET_NAME')

    
    
    client.put_object(
        Bucket=bucket_name,
        Key=filename,
        Body=image_bytes,
        ContentType=content_type,
        Metadata={"expires-at": expires_at},
        ACL='public-read'
    )
    # Month and date
    
    now = datetime.now()
    public_url = f"{os.getenv('PUBLIC_URL')}/{filename}"
    return public_url


import os

def calculate_cost(num_images: int = 1):
    """
    Calculate flat cost based on number of images, using a per-image rate from ENV.

    Args:
        num_images (int): Number of images generated.

    Returns:
        dict: Cost per image and total cost in USD.
    """
    try:
        cost_per_image = float(os.getenv("COST_PER_IMAGE", "0.003"))
    except ValueError:
        raise ValueError("Invalid COST_PER_IMAGE value in environment.")

    total_cost = round(cost_per_image * num_images, 6)

    return total_cost


