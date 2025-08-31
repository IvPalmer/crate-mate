# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies and Google Cloud SDK
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /usr/src/app
COPY ./requirements.txt /usr/src/app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code from the host to the container at /usr/src/app
COPY . /usr/src/app

# Authenticate with Google Cloud
# This assumes GOOGLE_APPLICATION_CREDENTIALS is set in the environment
# and the corresponding service account JSON file is available.
RUN if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then \
      echo "Activating Google Cloud service account..." && \
      gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS; \
    else \
      echo "Skipping Google Cloud authentication. Vision API will not work without credentials."; \
    fi


# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API server (Cloud Run sets $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]