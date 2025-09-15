"""
Enhanced YouTube link generator
Builds smart search URLs that usually land on the first result
"""
import logging
from typing import Dict, List
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class YouTubeEnhancedSearch:
    """
    Generate enhanced YouTube URLs for better search results
    """
    
    def __init__(self):
        self.base_url = "https://www.youtube.com/results?search_query="
        logger.info("YouTube enhanced search initialized")
    
    def generate_track_links(self, artist: str, album: str, tracklist: List[Dict]) -> List[Dict]:
        """
        Generate YouTube links for each track with smart search queries
        """
        enhanced_tracks = []
        
        for track in tracklist:
            track_title = track.get("title", "")
            position = track.get("position", "")
            duration = track.get("duration", "")
            
            if track_title:
                # Build optimized search query
                # Most specific to least specific
                search_query = f'"{artist}" "{track_title}"'
                
                enhanced_track = {
                    "position": position,
                    "title": track_title,
                    "duration": duration,
                    "youtube": {
                        "url": f"{self.base_url}{quote_plus(search_query)}",
                        "query": search_query,
                        # Special URL that often goes directly to first result
                        "lucky_url": f"https://www.youtube.com/results?search_query={quote_plus(search_query)}&sp=EgIQAQ%253D%253D"
                    }
                }
                
                enhanced_tracks.append(enhanced_track)
            else:
                # Track without title
                enhanced_tracks.append({
                    "position": position,
                    "title": track_title,
                    "duration": duration,
                    "youtube": None
                })
        
        return enhanced_tracks
    
    def generate_album_link(self, artist: str, album: str) -> Dict:
        """Generate album search link"""
        query = f'"{artist}" "{album}" full album'
        return {
            "url": f"{self.base_url}{quote_plus(query)}",
            "query": query
        }

