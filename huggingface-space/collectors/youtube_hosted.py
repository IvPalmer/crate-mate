"""
YouTube collector designed for hosted environments with network restrictions.
Falls back gracefully when direct access is not available.
"""
import logging
from typing import Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class YouTubeHostedSearch:
    """
    YouTube search designed for hosted environments like HuggingFace Spaces.
    Provides smart search URLs and graceful fallbacks when direct access is blocked.
    """
    
    def __init__(self):
        self.search_url = "https://www.youtube.com/results?search_query="
        self.watch_url = "https://www.youtube.com/watch?v="
        logger.info("YouTube hosted search initialized")
    
    def generate_track_links(self, artist: str, album: str, tracklist: List[Dict]) -> List[Dict]:
        """
        Generate YouTube links optimized for hosted environments.
        Creates smart search queries that often land on the correct video.
        """
        enhanced_tracks = []
        
        for track in tracklist:
            track_title = track.get("title", "")
            position = track.get("position", "")
            duration = track.get("duration", "")
            
            if track_title:
                # Create optimized search queries
                queries = self._build_search_queries(artist, track_title, album)
                
                # Use the most specific query as primary
                primary_query = queries[0]
                search_url = f"{self.search_url}{quote_plus(primary_query)}"
                
                enhanced_track = {
                    "position": position,
                    "title": track_title,
                    "duration": duration,
                    "youtube": {
                        "url": search_url,
                        "query": primary_query,
                        "is_search": True,
                        "search_type": "optimized"
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
    
    def _build_search_queries(self, artist: str, track_title: str, album: str) -> List[str]:
        """
        Build optimized search queries that often land on the correct video.
        Returns queries ordered from most to least specific.
        """
        # Clean up common issues in track titles
        cleaned_title = self._clean_track_title(track_title)
        
        # Build queries from most to least specific
        queries = [
            f'"{artist}" "{cleaned_title}" official',  # Most specific
            f'"{artist}" "{cleaned_title}"',            # Standard
            f'{artist} {cleaned_title} official',       # Less quoted
            f'{artist} {cleaned_title}',                # Basic
            f'{artist} {cleaned_title} {album}',        # With album
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries
    
    def _clean_track_title(self, title: str) -> str:
        """
        Clean track title for better search results.
        """
        # Common patterns that can interfere with search
        import re
        
        # Remove common suffixes/prefixes that might not match YouTube titles
        patterns_to_clean = [
            r'\s*\(.*?\)\s*$',          # Remove parenthetical info at end
            r'\s*\[.*?\]\s*$',          # Remove brackets at end  
            r'^\d+\.\s*',               # Remove track numbers at start
            r'\s*-\s*$',                # Remove trailing dashes
            r'\s+',                     # Normalize whitespace
        ]
        
        cleaned = title
        for pattern in patterns_to_clean[:-1]:  # Don't apply whitespace normalization yet
            cleaned = re.sub(pattern, '', cleaned)
        
        # Apply whitespace normalization
        cleaned = re.sub(patterns_to_clean[-1], ' ', cleaned).strip()
        
        return cleaned
    
    def generate_album_link(self, artist: str, album: str) -> Dict:
        """Generate optimized album search link"""
        queries = [
            f'"{artist}" "{album}" full album',
            f'"{artist}" "{album}" complete album',  
            f'{artist} {album} full album',
            f'{artist} {album} album'
        ]
        
        primary_query = queries[0]
        return {
            "url": f"{self.search_url}{quote_plus(primary_query)}",
            "query": primary_query,
            "is_search": True,
            "search_type": "album"
        }
    
    def get_status(self) -> Dict:
        """Return status information for debugging"""
        return {
            "collector_type": "hosted_search",
            "can_direct_access": False,
            "provides_search_links": True,
            "network_restrictions": "Compatible with hosted environments"
        }
