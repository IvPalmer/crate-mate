import requests
import os
import json

# The URL for the upload endpoint
url = "http://localhost/api/upload"

# Directory containing the sample images
samples_dir = "samples"

# List of image files to test
image_files = [
    "beatles.jpg",
    "led 1.jpg",
    "led 2.jpg",
    "brawther.jpg"
]

def test_image(file_path):
    """Sends a single image to the API and prints the response."""
    print(f"--- Testing {os.path.basename(file_path)} ---")
    try:
        with open(file_path, "rb") as f:
            files = {"image": (os.path.basename(file_path), f, "image/jpeg")}
            response = requests.post(url, files=files, timeout=60)

        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            try:
                # Try to print JSON error response if possible
                print(response.json())
            except json.JSONDecodeError:
                # Otherwise, print raw text
                print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    print("-" * (len(os.path.basename(file_path)) + 14))
    print("\\n")


if __name__ == "__main__":
    for image_file in image_files:
        path = os.path.join(samples_dir, image_file)
        if os.path.exists(path):
            test_image(path)
        else:
            print(f"File not found: {path}")
