"""
YouTube direct link finder using web scraping
No API needed, finds actual video URLs
"""
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional
from urllib.parse import quote_plus
import re
import json

logger = logging.getLogger(__name__)


class YouTubeDirectSearch:
    """
    Finds direct YouTube video links without API
    """
    
    def __init__(self):
        self.search_url = "https://www.youtube.com/results?search_query="
        logger.info("YouTube direct search initialized")
    
    async def find_track_videos(self, artist: str, album: str, tracklist: List[Dict]) -> List[Dict]:
        """
        Find actual YouTube video URLs for tracks
        Returns tracks with direct video links
        """
        enhanced_tracks = []
        
        # Process tracks in parallel for speed
        tasks = []
        for track in tracklist[:10]:  # Limit to first 10 tracks
            track_title = track.get("title", "")
            position = track.get("position", "")
            
            if track_title:
                task = self._find_video_for_track(artist, track_title, album, position)
                tasks.append(task)
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results with track info
        for i, track in enumerate(tracklist):
            enhanced_track = {
                "position": track.get("position", ""),
                "title": track.get("title", ""),
                "duration": track.get("duration", ""),
                "youtube_url": None,
                "youtube_title": None,
                "youtube_channel": None,
                "youtube_video_id": None
            }
            
            # Add video info if found
            if i < len(results) and isinstance(results[i], dict):
                enhanced_track.update(results[i])
            
            enhanced_tracks.append(enhanced_track)
        
        return enhanced_tracks
    
    async def _find_video_for_track(self, artist: str, track_title: str, album: str, position: str) -> Optional[Dict]:
        """
        Find a specific track's YouTube video
        """
        # Try different search queries
        queries = [
            f"{artist} {track_title}",
            f"{artist} - {track_title}",
            f"{artist} {track_title} official",
            f"{artist} {track_title} {album}"
        ]
        
        for query in queries:
            try:
                result = await self._search_youtube(query)
                if result:
                    logger.info(f"Found video for {position}. {track_title}: {result['url']}")
                    return result
            except Exception as e:
                logger.debug(f"Error searching for {track_title}: {e}")
                continue
        
        logger.info(f"No video found for {position}. {track_title}")
        return None
    
    async def _search_youtube(self, query: str) -> Optional[Dict]:
        """
        Search YouTube and extract first video result
        """
        search_url = f"{self.search_url}{quote_plus(query)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract video data from YouTube's initial data
                        video_data = self._extract_video_from_html(html)
                        if video_data:
                            return video_data
        except Exception as e:
            logger.debug(f"Error in YouTube search: {e}")
        
        return None
    
    def _extract_video_from_html(self, html: str) -> Optional[Dict]:
        """
        Extract first video result from YouTube HTML
        """
        try:
            # Look for ytInitialData in the HTML
            match = re.search(r'var ytInitialData = ({.*?});', html)
            if not match:
                return None
            
            # Parse the JSON data
            yt_data = json.loads(match.group(1))
            
            # Navigate through the data structure to find videos
            contents = (
                yt_data.get('contents', {})
                .get('twoColumnSearchResultsRenderer', {})
                .get('primaryContents', {})
                .get('sectionListRenderer', {})
                .get('contents', [])
            )
            
            # Find first video result
            for section in contents:
                items = section.get('itemSectionRenderer', {}).get('contents', [])
                for item in items:
                    if 'videoRenderer' in item:
                        video = item['videoRenderer']
                        video_id = video.get('videoId')
                        
                        if video_id:
                            # Extract video details
                            title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                            channel = video.get('ownerText', {}).get('runs', [{}])[0].get('text', '')
                            
                            return {
                                'youtube_url': f"https://www.youtube.com/watch?v={video_id}",
                                'youtube_video_id': video_id,
                                'youtube_title': title,
                                'youtube_channel': channel
                            }
            
        except Exception as e:
            logger.debug(f"Error extracting video data: {e}")
        
        return None
    
    async def find_album_video(self, artist: str, album: str) -> Optional[Dict]:
        """
        Try to find full album video
        """
        queries = [
            f"{artist} {album} full album",
            f"{artist} - {album} (Full Album)",
            f"{artist} {album} complete"
        ]
        
        for query in queries:
            result = await self._search_youtube(query)
            if result:
                return result
        
        return None

