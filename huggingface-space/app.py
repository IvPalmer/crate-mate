import streamlit as st
import os
import io
import asyncio
from PIL import Image, ImageOps
import hashlib
from typing import Dict, List, Optional
import sys

# Add collectors to path
sys.path.append('collectors')

st.set_page_config(
    page_title="Crate‑Mate",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .link-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 5px;
        text-decoration: none;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .spotify-link { background-color: #1DB954; }
    .youtube-link { background-color: #FF0000; }
    .discogs-link { background-color: #333333; }
    .bandcamp-link { background-color: #629aa0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_collectors():
    """Load and initialize collectors"""
    collectors = {}
    
    try:
        from gemini import GeminiCollector
        collectors['gemini'] = GeminiCollector()
    except Exception as e:
        st.sidebar.error(f"Gemini: {e}")
    
    try:
        from discogs import DiscogsCollector
        collectors['discogs'] = DiscogsCollector()
    except Exception as e:
        st.sidebar.error(f"Discogs: {e}")
    
    try:
        from spotify import SpotifyCollector
        collectors['spotify'] = SpotifyCollector()
    except Exception as e:
        st.sidebar.error(f"Spotify: {e}")
    
    try:
        from youtube_enhanced import YouTubeEnhancedSearch
        collectors['youtube'] = YouTubeEnhancedSearch()
    except Exception as e:
        st.sidebar.error(f"YouTube: {e}")
    
    try:
        from bandcamp import BandcampCollector
        collectors['bandcamp'] = BandcampCollector()
    except Exception as e:
        st.sidebar.error(f"Bandcamp: {e}")
    
    return collectors

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for recognition"""
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    image = ImageOps.exif_transpose(image)
    
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image

async def perform_search(image: Image.Image, collectors: Dict) -> Dict:
    """Perform hybrid search"""
    
    # Step 1: Gemini identification
    if 'gemini' not in collectors:
        return {"success": False, "error": "Gemini collector not available"}
    
    try:
        gemini_result = await collectors['gemini'].identify_album(image)
        if not gemini_result.get("success"):
            return gemini_result
        
        identification = gemini_result.get("result", {})
        artist = identification.get("artist", "")
        album = identification.get("album", "")
        confidence = identification.get("confidence", 0)
        
        if not artist or not album:
            return {"success": False, "error": "Could not extract artist/album"}
        
        # Step 2: Search platforms
        results = {}
        
        # Discogs
        if 'discogs' in collectors:
            try:
                discogs_results = await collectors['discogs'].search_release(f"{artist} {album}")
                if discogs_results and len(discogs_results) > 0:
                    release_id = discogs_results[0].get('id')
                    if release_id:
                        details = await collectors['discogs'].get_release_details(release_id)
                        results['discogs'] = {"success": True, "results": discogs_results, "details": details}
            except Exception as e:
                results['discogs'] = {"success": False, "error": str(e)}
        
        # Spotify
        if 'spotify' in collectors:
            try:
                spotify_result = await collectors['spotify'].fetch_album_details(artist, album)
                results['spotify'] = {"success": True, "result": spotify_result}
            except Exception as e:
                results['spotify'] = {"success": False, "error": str(e)}
        
        # Bandcamp
        if 'bandcamp' in collectors:
            try:
                bandcamp_link = collectors['bandcamp'].find_release_link(artist, album)
                results['bandcamp'] = {"success": True, "link": bandcamp_link}
            except Exception as e:
                results['bandcamp'] = {"success": False, "error": str(e)}
        
        # Build final result
        discogs_details = results.get('discogs', {}).get('details', {})
        spotify_data = results.get('spotify', {}).get('result', {})
        
        # Get tracklist and enhance with YouTube
        tracklist = discogs_details.get('tracklist', [])
        if tracklist and 'youtube' in collectors:
            try:
                enhanced_tracks = collectors['youtube'].generate_track_links(artist, album, tracklist)
                tracklist = enhanced_tracks
            except Exception:
                pass
        
        final_result = {
            "success": True,
            "identification": {
                "artist": artist,
                "album": album,
                "confidence": confidence
            },
            "album": {
                "artist": artist,
                "title": album,
                "cover_image": discogs_details.get("cover_image"),
                "year": discogs_details.get("year"),
                "genres": discogs_details.get("genres", []),
                "styles": discogs_details.get("styles", [])
            },
            "links": {
                "discogs": discogs_details.get("discogs_url"),
                "spotify": spotify_data.get("url"),
                "youtube": discogs_details.get("youtube_url"),
                "bandcamp": results.get("bandcamp", {}).get("link")
            },
            "tracks": tracklist,
            "price_info": discogs_details.get("price_info", {}),
            "market_stats": discogs_details.get("market_stats", {})
        }
        
        return final_result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_result(result: Dict):
    """Display search results"""
    
    if not result.get("success"):
        st.error(f"Search failed: {result.get('error', 'Unknown error')}")
        return
    
    identification = result.get("identification", {})
    album_info = result.get("album", {})
    links = result.get("links", {})
    tracks = result.get("tracks", [])
    market_stats = result.get("market_stats", {})
    
    # Album info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        cover_url = album_info.get("cover_image")
        if cover_url:
            try:
                st.image(cover_url, width=300, caption="Album Cover")
            except:
                st.write("🎵 Album Cover")
        else:
            st.write("🎵 No cover available")
    
    with col2:
        st.markdown(f"### {album_info.get('artist', 'Unknown Artist')}")
        st.markdown(f"**{album_info.get('title', 'Unknown Album')}**")
        
        if album_info.get("year"):
            st.write(f"📅 Year: {album_info['year']}")
        
        if album_info.get("genres"):
            st.write(f"🎭 Genres: {', '.join(album_info['genres'])}")
        
        confidence = identification.get("confidence", 0)
        if confidence >= 90:
            st.success(f"✅ High Confidence: {confidence}%")
        elif confidence >= 70:
            st.warning(f"⚠️ Medium Confidence: {confidence}%")
        else:
            st.error(f"❌ Low Confidence: {confidence}%")
    
    # Links
    st.markdown("### 🔗 Listen & Buy")
    link_cols = st.columns(4)
    
    with link_cols[0]:
        if links.get("spotify"):
            st.markdown(f'<a href="{links["spotify"]}" target="_blank" class="link-button spotify-link">🎵 Spotify</a>', unsafe_allow_html=True)
        else:
            st.write("🎵 Spotify: Unavailable")
    
    with link_cols[1]:
        if links.get("youtube"):
            st.markdown(f'<a href="{links["youtube"]}" target="_blank" class="link-button youtube-link">📺 YouTube</a>', unsafe_allow_html=True)
        else:
            st.write("📺 YouTube: Unavailable")
    
    with link_cols[2]:
        if links.get("discogs"):
            st.markdown(f'<a href="{links["discogs"]}" target="_blank" class="link-button discogs-link">💿 Discogs</a>', unsafe_allow_html=True)
        else:
            st.write("💿 Discogs: Unavailable")
    
    with link_cols[3]:
        if links.get("bandcamp"):
            st.markdown(f'<a href="{links["bandcamp"]}" target="_blank" class="link-button bandcamp-link">🎪 Bandcamp</a>', unsafe_allow_html=True)
        else:
            st.write("🎪 Bandcamp: Unavailable")
    
    # Market info
    if market_stats:
        st.markdown("### 💰 Market Information")
        market_cols = st.columns(2)
        
        with market_cols[0]:
            if market_stats.get("lowest_price"):
                st.metric("Lowest Price", f"${market_stats['lowest_price']}")
        
        with market_cols[1]:
            if market_stats.get("num_for_sale"):
                st.metric("For Sale", f"{market_stats['num_for_sale']} copies")
    
    # Tracklist
    if tracks:
        st.markdown("### 🎵 Tracklist")
        
        for i, track in enumerate(tracks, 1):
            track_title = track.get("title", f"Track {i}")
            duration = track.get("duration", "")
            
            track_col1, track_col2 = st.columns([3, 1])
            
            with track_col1:
                st.write(f"**{i}. {track_title}**")
                if duration:
                    st.write(f"⏱️ {duration}")
            
            with track_col2:
                track_links = []
                
                if track.get("spotify_url"):
                    track_links.append(f'<a href="{track["spotify_url"]}" target="_blank" class="link-button spotify-link" style="font-size: 0.8em; padding: 0.25rem 0.5rem;">🎵</a>')
                
                if track.get("youtube_url"):
                    track_links.append(f'<a href="{track["youtube_url"]}" target="_blank" class="link-button youtube-link" style="font-size: 0.8em; padding: 0.25rem 0.5rem;">📺</a>')
                elif track.get("youtube_search"):
                    track_links.append(f'<a href="{track["youtube_search"]}" target="_blank" class="link-button youtube-link" style="font-size: 0.8em; padding: 0.25rem 0.5rem;">🔍</a>')
                
                if track_links:
                    st.markdown(" ".join(track_links), unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🎵 Crate‑Mate</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-powered album recognition and music discovery</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        gemini_key = st.text_input("Gemini API Key", type="password", help="Required for album identification")
        discogs_token = st.text_input("Discogs Token", type="password", help="Required for release information")
        spotify_client_id = st.text_input("Spotify Client ID", type="password", help="Required for Spotify links")
        spotify_client_secret = st.text_input("Spotify Client Secret", type="password", help="Required for Spotify links")
        
        # Set environment variables
        if gemini_key:
            os.environ['GEMINI_API_KEY'] = gemini_key
        if discogs_token:
            os.environ['DISCOGS_TOKEN'] = discogs_token
        if spotify_client_id:
            os.environ['SPOTIFY_CLIENT_ID'] = spotify_client_id
        if spotify_client_secret:
            os.environ['SPOTIFY_CLIENT_SECRET'] = spotify_client_secret
        
        st.markdown("---")
        st.markdown("### ℹ️ How to use")
        st.markdown("""
        1. Enter your API keys above
        2. Upload an album cover image
        3. Wait for AI analysis
        4. Explore the results!
        """)
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to use the app.")
        st.stop()
    
    # Main content
    st.markdown("### 📸 Upload Album Cover")
    
    uploaded_file = st.file_uploader(
        "Choose an album cover image...",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear image of an album cover for identification"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Identify Album", type="primary", use_container_width=True):
            processed_image = preprocess_image(image)
            
            with st.spinner("Loading AI models..."):
                collectors = load_collectors()
            
            if not collectors:
                st.error("Failed to load collectors. Please check your API keys.")
                st.stop()
            
            with st.spinner("Analyzing album cover..."):
                try:
                    result = asyncio.run(perform_search(processed_image, collectors))
                    display_result(result)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.8em;">'
        'Powered by Gemini Vision, Discogs, Spotify, and YouTube'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
