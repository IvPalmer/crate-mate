#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Test specific Discogs release
release_id = "11939850"  # Brawther - Transient States
token = os.getenv("DISCOGS_TOKEN")

headers = {
    "Authorization": f"Discogs token={token}",
    "User-Agent": "CrateMate/1.0"
}

response = requests.get(
    f"https://api.discogs.com/releases/{release_id}",
    headers=headers
)

if response.status_code == 200:
    data = response.json()
    videos = data.get('videos', [])
    print(f"Release: {data.get('title')} by {data.get('artists_sort')}")
    print(f"Found {len(videos)} videos")
    for v in videos:
        print(f"- {v.get('title')} : {v.get('uri')}")
else:
    print(f"Error: {response.status_code}")
