"""
Contains the main FastAPI application and defines the API routes.

Each route has /api as a prefix, so the full path to the route is /api/{route}.
"""

import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import JSONResponse
from PIL import Image

from app.hybrid_search import HybridSearch


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create instances (optionally include universal search if enabled)
enable_universal = str(os.getenv("ENABLE_UNIVERSAL", "0")).lower() in ["1", "true", "yes"]
simple_searcher = None
if enable_universal:
    try:
        from app.simple_universal_search import SimpleUniversalSearch
        simple_searcher = SimpleUniversalSearch()
    except Exception as e:
        logger.warning("Universal search disabled due to import error: %s", e)
        simple_searcher = None

hybrid_searcher = HybridSearch()

app = FastAPI()

# CORS (allow Streamlit or other frontends)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()


@router.get("/")
async def read_root():
    """
    Default route to test if the API is running.
    """
    return {"message": "Backend is online."}


@router.post("/upload")
async def upload_image(image: UploadFile = File(...)) -> dict:
    """
    Identify the album cover in an image using universal search and return the result.

    Parameters:
    - image (UploadFile): The image file to analyze.

    Returns:
    - dict: The matched record from Discogs, or a message if no matches are found.

    Raises:
    - HTTPException: If the image is invalid or an error occurs during processing.
    """
    logger.info("Received upload request for file: %s", image.filename)

    try:
        image_pil = Image.open(image.file)
    except Exception as e:
        logger.error("Invalid image file: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image file.") from e

    try:
        logger.info("Starting hybrid search...")
        # Use hybrid search by default
        use_hybrid = True  # default path
        
        if use_hybrid:
            result = await hybrid_searcher.search_album(image_pil)
        else:
            if simple_searcher is None:
                raise HTTPException(status_code=503, detail="Universal search disabled on this deployment.")
            result = await simple_searcher.search_album(image_pil)

        if result and not result.get("error"):
            # Check if using hybrid search (already has all links)
            if "discogs_url" in result:
                # Hybrid search result - already formatted with all links
                # Return the full formatted response
                return result
            else:
                # Simple search result - needs formatting
                if "album_name" in result:
                    formatted_result = result
                else:
                    # Legacy format
                    formatted_result = {
                        "album_name": result.get("title", ""),
                        "artist_name": result.get("artist", ""),
                        "album_url": f"https://www.discogs.com/release/{result.get('id', '')}",
                        "release_date": result.get("year", ""),
                        "genres": result.get("format", []),
                        "album_image": result.get("cover_image", result.get("thumb", "")),
                        "method": "universal_search"
                    }
                
                # For simple search, still try to get YouTube/Spotify
                youtube_url = "unavailable"
                spotify_url = "unavailable"
                
                try:
                    from app.collectors.youtube import YouTubeCollector
                    youtube = YouTubeCollector("youtube")
                    youtube_result = await youtube.fetch_album_details(
                        formatted_result.get("artist_name", ""),
                        formatted_result.get("album_name", "")
                    )
                    if youtube_result and "youtube_url" in youtube_result:
                        youtube_url = youtube_result["youtube_url"]
                except Exception as e:
                    logger.error(f"Error getting YouTube link: {str(e)}")

                try:
                    from app.collectors.spotify import SpotifyCollector
                    spotify = SpotifyCollector("spotify")
                    spotify_result = await spotify.fetch_album_details(
                        formatted_result.get("artist_name", ""),
                        formatted_result.get("album_name", "")
                    )
                    if spotify_result and "spotify_url" in spotify_result:
                        spotify_url = spotify_result["spotify_url"]
                except Exception as e:
                    logger.error(f"Error getting Spotify link: {str(e)}")

                return {
                    "album_name": formatted_result.get("album_name", ""),
                    "artist_name": formatted_result.get("artist_name", ""),
                    "album_url": formatted_result.get("album_url", "unavailable"),
                    "release_date": formatted_result.get("release_date", ""),
                    "genres": formatted_result.get("genres", []),
                    "album_image": formatted_result.get("album_image", ""),
                    "method": formatted_result.get("method", "simple"),
                    "discogs_url": formatted_result.get("album_url", "unavailable"),
                    "youtube_url": youtube_url,
                    "spotify_url": spotify_url,
                    "alternatives": formatted_result.get("alternatives", [])
                }
        else:
            logger.warning("Universal search failed: No matches found")
            return JSONResponse(status_code=404, content={"message": "No matches found. Reason: No matches found"})

    except Exception as e:
        logger.error("Error processing image: %s", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the image.",
        ) from e

# Prefix all routes with /api
app.include_router(router, prefix="/api")
