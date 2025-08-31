"""
Handles direct interaction with the Discogs API, including reverse image search.
"""
import logging
import os
import requests
import json

logger = logging.getLogger(__name__)

DISCOGS_USER_TOKEN = os.getenv('DISCOGS_TOKEN')
DISCOGS_IMAGE_SEARCH_URL = "https://api.discogs.com/database/search?type=release&image"

class DiscogsCollector:
    """
    Performs a reverse image search using a direct request to the Discogs API.
    """
    def __init__(self, name="discogs_image_search"):
        self.name = name
        self.headers = {
            "Authorization": f"Discogs token={DISCOGS_USER_TOKEN}",
            "User-Agent": "CrateMate/1.0",
            "Content-Type": "application/octet-stream"
        }

    async def search_by_image(self, image_bytes: bytes) -> dict:
        """
        Performs a reverse image search using the Discogs API.

        Args:
            image_bytes: The byte content of the image to search for.

        Returns:
            A dictionary containing the search results or an error.
        """
        if not DISCOGS_USER_TOKEN:
            return {"success": False, "error": "Discogs client not initialized. Check DISCOGS_TOKEN."}

        try:
            response = requests.get(
                DISCOGS_IMAGE_SEARCH_URL, 
                headers=self.headers, 
                data=image_bytes
            )
            
            response.raise_for_status()

            # The response is a JSON string where the first line is '{"resp": ' and the last is '}'.
            # We need to clean it up to parse it correctly.
            # This is highly specific to this undocumented endpoint.
            if response.text and response.text.startswith('{"resp":'):
                # Find the start of the actual JSON content
                start_index = response.text.find('{"results":')
                if start_index != -1:
                    json_str = response.text[start_index:-1] # Trim the leading part and trailing '}'
                    data = json.loads(json_str)
                    results = data.get("results", [])

                    if results:
                        best_match = results[0]
                        logger.info(f"Discogs reverse image search found: {best_match.get('title')}")
                        
                        album_data = {
                            'id': best_match.get('id'),
                            'title': best_match.get('title'),
                            'artist': best_match.get('artist'),
                            'year': best_match.get('year'),
                            'cover_image': best_match.get('cover_image'),
                            'thumb': best_match.get('thumb'),
                            'format': best_match.get('format', []),
                        }
                        return {"success": True, "album": album_data}

            return {"success": False, "error": "No results from Discogs reverse image search."}
        
        except requests.exceptions.HTTPError as http_err:
             logger.error(f"HTTP error during Discogs reverse image search: {http_err} - {http_err.response.text}")
             return {"success": False, "error": str(http_err)}
        except Exception as e:
            logger.error(f"Error during Discogs reverse image search: {e}")
            return {"success": False, "error": str(e)}

    def get_name(self):
        return self.name
