"""
YouTube video URL extractor using yt-dlp
Gets actual video URLs from searches without API
"""
import logging
import subprocess
import json
from typing import Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class YouTubeYtdlpSearch:
    """
    Extract actual YouTube video URLs using yt-dlp
    """
    
    def __init__(self):
        logger.info("YouTube yt-dlp search initialized")
        self.ytdlp_available = self._check_ytdlp()
    
    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is available"""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"yt-dlp found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning("yt-dlp not found. Install with: pip install yt-dlp")
        return False
    
    def get_first_video_url(self, search_query: str) -> Optional[Dict]:
        """
        Get the first YouTube video URL for a search query
        Returns video info including direct URL
        """
        if not self.ytdlp_available:
            return None
        
        try:
            # Use ytsearch1: to get only the first result
            search_string = f"ytsearch1:{search_query}"
            
            # yt-dlp command to get video info without downloading
            cmd = [
                "yt-dlp",
                "--dump-json",  # Output JSON info
                "--no-playlist",  # Don't process playlists
                "--quiet",  # Suppress progress output
                "--no-warnings",  # Suppress warnings
                search_string
            ]
            
            logger.info(f"Searching YouTube for: {search_query}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            if result.returncode == 0 and result.stdout:
                # Parse the JSON output
                video_info = json.loads(result.stdout)
                
                # Extract relevant info
                return {
                    "url": f"https://www.youtube.com/watch?v={video_info['id']}",
                    "video_id": video_info["id"],
                    "title": video_info.get("title", ""),
                    "channel": video_info.get("uploader", ""),
                    "duration": video_info.get("duration_string", ""),
                    "is_search": False  # This is a direct video link
                }
            
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp search timed out")
        except json.JSONDecodeError:
            logger.error("Failed to parse yt-dlp output")
        except Exception as e:
            logger.error(f"yt-dlp error: {e}")
        
        return None
    
    def get_track_videos(self, artist: str, album: str, tracklist: List[Dict]) -> List[Dict]:
        """
        Get YouTube video URLs for each track
        Returns tracks with direct video links where found
        """
        enhanced_tracks = []
        
        for track in tracklist:
            track_title = track.get("title", "")
            position = track.get("position", "")
            duration = track.get("duration", "")
            
            enhanced_track = {
                "position": position,
                "title": track_title,
                "duration": duration,
                "youtube": None
            }
            
            if track_title and self.ytdlp_available:
                # Search for "Artist Track" 
                search_query = f"{artist} {track_title}"
                video_info = self.get_first_video_url(search_query)
                
                if video_info:
                    enhanced_track["youtube"] = video_info
                    logger.info(f"Found video for {track_title}: {video_info['url']}")
            
            enhanced_tracks.append(enhanced_track)
        
        return enhanced_tracks

