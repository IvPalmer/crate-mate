# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install only minimal runtime system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the (slimmer) production requirements to speed up builds on Cloud Run
COPY ./requirements-prod.txt /usr/src/app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code from the host to the container at /usr/src/app
COPY . /usr/src/app

# Use Workload Identity on Cloud Run; no gcloud or key activation in the image


# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API server (Cloud Run sets $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]