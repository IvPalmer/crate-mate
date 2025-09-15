"""
Simple standalone collectors for Streamlit app
"""
import os
import logging
import requests
from urllib.parse import quote
import google.generativeai as genai
from PIL import Image
import discogs_client
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)


class GeminiCollector:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def identify_album(self, image):
        """Identify album from image"""
        try:
            prompt = """
            Analyze this album cover and identify:
            1. Artist name
            2. Album title  
            3. Release year (if visible)
            4. Genre
            5. Confidence level (1-100%)
            
            Format response as:
            Artist: [name]
            Album: [title]
            Year: [year]
            Genre: [genre]  
            Confidence: [percentage]%
            """
            
            response = self.model.generate_content([prompt, image])
            text = response.text
            
            # Parse response
            result = {}
            for line in text.split('\n'):
                if line.startswith('Artist:'):
                    result['artist'] = line.replace('Artist:', '').strip()
                elif line.startswith('Album:'):
                    result['album'] = line.replace('Album:', '').strip()
                elif line.startswith('Year:'):
                    result['year'] = line.replace('Year:', '').strip()
                elif line.startswith('Genre:'):
                    result['genre'] = line.replace('Genre:', '').strip()
                elif line.startswith('Confidence:'):
                    conf_str = line.replace('Confidence:', '').strip().rstrip('%')
                    try:
                        result['confidence'] = int(conf_str)
                    except:
                        result['confidence'] = 80
            
            result['description'] = text
            return result
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None


class DiscogsCollector:
    def __init__(self, token):
        self.client = None
        if token:
            self.client = discogs_client.Client("CrateMate/1.0", user_token=token)
    
    def search_album(self, artist, album):
        """Search for album on Discogs"""
        if not self.client:
            return None
            
        try:
            query = f"{artist} {album}"
            results = self.client.search(query, type="release")
            
            if results and results.count > 0:
                release = results[0]  # Get first result
                
                data = {
                    'artist': release.artists[0].name if release.artists else artist,
                    'title': release.title,
                    'year': getattr(release, 'year', None),
                    'genre': getattr(release, 'genres', []),
                    'styles': getattr(release, 'styles', []),
                    'label': release.labels[0].name if release.labels else None,
                    'catalog_number': getattr(release, 'catno', None),
                    'discogs_url': release.url if hasattr(release, 'url') else None,
                }
                
                # Get cover image
                if hasattr(release, 'images') and release.images:
                    data['cover_image'] = release.images[0]['uri']
                
                # Get detailed tracklist
                if hasattr(release, 'tracklist') and release.tracklist:
                    tracks = []
                    for track in release.tracklist:
                        track_data = {
                            'title': track.title,
                            'position': getattr(track, 'position', ''),
                            'duration': getattr(track, 'duration', '')
                        }
                        tracks.append(track_data)
                    data['tracklist'] = tracks
                    data['tracklist_simple'] = [track.title for track in release.tracklist]
                
                return data
                
        except Exception as e:
            logger.error(f"Discogs error: {e}")
        
        return None


class SpotifyCollector:
    def __init__(self, client_id, client_secret):
        self.client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )
        )
    
    def search_album(self, artist, album):
        """Search for album on Spotify with track details"""
        try:
            query = f"artist:{artist} album:{album}"
            results = self.client.search(q=query, type='album', limit=1)
            
            if results['albums']['items']:
                album_data = results['albums']['items'][0]
                album_id = album_data['id']
                
                # Get album tracks
                tracks_result = self.client.album_tracks(album_id)
                tracks = []
                
                for i, track in enumerate(tracks_result['items'], 1):
                    tracks.append({
                        'position': i,
                        'title': track['name'],
                        'duration': self._format_duration(track['duration_ms']),
                        'spotify_url': track['external_urls']['spotify']
                    })
                
                return {
                    'spotify_url': album_data['external_urls']['spotify'],
                    'name': album_data['name'],
                    'artist': album_data['artists'][0]['name'],
                    'tracks': tracks
                }
        except Exception as e:
            logger.error(f"Spotify error: {e}")
        
        return None
    
    def _format_duration(self, duration_ms):
        """Convert milliseconds to MM:SS format"""
        if not duration_ms:
            return ""
        seconds = duration_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"


class YouTubeCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def search_album(self, artist, album):
        """Search for album on YouTube"""
        try:
            query = f"{artist} {album} full album"
            if self.api_key:
                # Use API
                url = f"https://www.googleapis.com/youtube/v3/search"
                params = {
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': 1,
                    'key': self.api_key
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data['items']:
                        video_id = data['items'][0]['id']['videoId']
                        return {'youtube_url': f"https://www.youtube.com/watch?v={video_id}"}
            else:
                # Basic search URL
                search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
                return {'youtube_url': search_url}
                
        except Exception as e:
            logger.error(f"YouTube error: {e}")
        
        return None
    
    def generate_track_links(self, artist, album, tracklist):
        """Generate direct YouTube video links for individual tracks"""
        enhanced_tracks = []
        
        for track in tracklist:
            if isinstance(track, str):
                track_title = track
                position = len(enhanced_tracks) + 1
            else:
                track_title = track.get('title', str(track))
                position = track.get('position', len(enhanced_tracks) + 1)
            
            # Get direct YouTube video link
            youtube_url = self._get_track_video_url(artist, track_title)
            
            enhanced_tracks.append({
                'position': position,
                'title': track_title,
                'youtube_url': youtube_url,
                'search_query': f"{artist} {track_title}"
            })
        
        return enhanced_tracks
    
    def _get_track_video_url(self, artist, track_title):
        """Get direct YouTube video URL for a specific track"""
        try:
            query = f"{artist} {track_title}"
            
            if self.api_key:
                # Use YouTube Data API to get direct video link
                url = f"https://www.googleapis.com/youtube/v3/search"
                params = {
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': 1,
                    'key': self.api_key
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data['items']:
                        video_id = data['items'][0]['id']['videoId']
                        return f"https://www.youtube.com/watch?v={video_id}"
            
            # Fallback to search URL if no API key or API fails
            search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
            return search_url
                
        except Exception as e:
            logger.error(f"YouTube track search error: {e}")
            # Fallback to search URL
            search_query = f"{artist} {track_title}"
            return f"https://www.youtube.com/results?search_query={quote(search_query)}"


class BandcampCollector:
    def search_album(self, artist, album):
        """Search for album on Bandcamp"""
        try:
            query = f"{artist} {album}".replace(' ', '+')
            search_url = f"https://bandcamp.com/search?q={query}"
            return {'bandcamp_url': search_url}
        except Exception as e:
            logger.error(f"Bandcamp error: {e}")
        
        return None
