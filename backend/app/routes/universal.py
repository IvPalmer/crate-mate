"""
Universal album identification routes that don't rely on pre-loaded database
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.universal_recognizer import UniversalVinylRecognizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/universal", tags=["universal"])

# Initialize recognizer
recognizer = UniversalVinylRecognizer()


@router.post("/identify")
async def identify_album(image: UploadFile = File(...)):
    """
    Identify a vinyl record from an uploaded image
    Works with ANY album, not limited to pre-loaded database
    """
    logger.info(f"Received image for identification: {image.filename}")
    
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Process the image
        result = await recognizer.identify_album(image)
        
        if result["success"]:
            logger.info(f"Successfully identified album: {result['album'].get('title', 'Unknown')}")
        else:
            logger.warning(f"Failed to identify album: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in album identification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/identify-batch")
async def identify_albums_batch(images: list[UploadFile] = File(...)):
    """
    Identify multiple vinyl records from uploaded images
    """
    logger.info(f"Received {len(images)} images for batch identification")
    
    results = []
    
    for idx, image in enumerate(images):
        try:
            logger.info(f"Processing image {idx + 1}/{len(images)}: {image.filename}")
            
            # Validate file type
            if not image.content_type.startswith("image/"):
                results.append({
                    "index": idx,
                    "filename": image.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Process the image
            result = await recognizer.identify_album(image)
            result["index"] = idx
            result["filename"] = image.filename
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing image {image.filename}: {str(e)}")
            results.append({
                "index": idx,
                "filename": image.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(images),
        "results": results
    }


@router.get("/status")
async def get_status():
    """
    Get status of the universal recognizer
    """
    return {
        "status": "operational",
        "features": {
            "google_vision": recognizer.vision_available,
            "discogs": True,
            "spotify": True,
            "youtube": True,
            "musicbrainz": True
        },
        "description": "Universal vinyl record identification - works with ANY album"
    }
