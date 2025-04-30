FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y python3-pip git wget && \
    ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Create model directory
ARG MODEL_BASE_PATH="/models"
RUN mkdir -p ${MODEL_BASE_PATH}
ENV MODEL_BASE_PATH=${MODEL_BASE_PATH}



# Add Hugging Face token as build argument
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Log in to Hugging Face
RUN python3 -c "from huggingface_hub import HfApi; HfApi().token = '${HUGGING_FACE_HUB_TOKEN}'; print('Logged in to Hugging Face')"



# Copy and run the model download script
COPY src/download_model.py /download_model.py
RUN python3 /download_model.py

# Set environment variables to tell our code where to find the models
ENV SD3_MODEL_PATH="${MODEL_BASE_PATH}/sd3-medium"
ENV SD3_VAE_PATH="${MODEL_BASE_PATH}/taesd3"

# Copy application code
COPY src /src

# Start the handler
CMD ["python3", "/src/handler.py"]