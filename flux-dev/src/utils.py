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


def calculate_cost(width: int, height: int):
    """
    Calculate cost based on image dimensions and COST_PER_MEGAPX env variable.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        dict: Dictionary containing pixels, megapixels, and total cost in USD.
    """
    pixels = width * height
    megapixels = pixels / 1_000_000

    # Get cost from env, fallback to 0.00137 if not set
    try:
        cost_per_megapixel = float(os.getenv("COST_PER_MEGAPIXEL", "0.025"))
    except ValueError:
        raise ValueError("Invalid COST_PER_MEGAPX value in environment.")

    cost_usd = round(megapixels * cost_per_megapixel, 8)

    return cost_usd