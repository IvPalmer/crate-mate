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
    
    .upload-section {
        border: 2px dashed #1DB954;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f8f9fa;
    }
    
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .service-status {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 1rem 0;
    }
    
    .status-item {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .status-available {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-unavailable {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Import collectors
try:
    from gemini import GeminiVisionCollector
    from discogs import DiscogsCollector
    from spotify import SpotifyCollector
    from youtube import YouTubeCollector
    from bandcamp import BandcampCollector
except ImportError as e:
    st.error(f"Error importing collectors: {e}")
    st.stop()

# Session state initialization
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result' not in st.session_state:
    st.session_state.result = None

# Header
st.markdown('<h1 class="main-header">🎵 Crate‑Mate</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI-powered album recognition and music discovery</p>', unsafe_allow_html=True)

# Debug: Show environment variable status prominently
if os.getenv('GEMINI_API_KEY'):
    st.success(f"🔧 DEBUG: Environment variables loaded! GEMINI_API_KEY length: {len(os.getenv('GEMINI_API_KEY'))}")
else:
    st.error("🔧 DEBUG: Environment variables NOT loaded!")

# Sidebar for API keys
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    st.markdown("Enter your API keys to enable full functionality:")
    
    # Try to get API keys from environment variables first
    gemini_key = os.getenv('GEMINI_API_KEY') or st.text_input("🤖 Gemini API Key", type="password", help="Required for AI album identification")
    discogs_token = os.getenv('DISCOGS_TOKEN') or st.text_input("💿 Discogs Token", type="password", help="For enhanced album data and pricing")
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID') or st.text_input("🎵 Spotify Client ID", type="password", help="For Spotify track links")
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET') or st.text_input("🎵 Spotify Client Secret", type="password", help="For Spotify track links")
    
    st.markdown("---")
    
    # Service status
    services_status = []
    if gemini_key:
        env_indicator = " (env)" if os.getenv('GEMINI_API_KEY') else ""
        services_status.append(f"✅ Gemini AI{env_indicator}")
    else:
        services_status.append("❌ Gemini AI")
    
    if discogs_token:
        env_indicator = " (env)" if os.getenv('DISCOGS_TOKEN') else ""
        services_status.append(f"✅ Discogs{env_indicator}")
    else:
        services_status.append("❌ Discogs")
    
    if spotify_client_id and spotify_client_secret:
        env_indicator = " (env)" if os.getenv('SPOTIFY_CLIENT_ID') else ""
        services_status.append(f"✅ Spotify{env_indicator}")
    else:
        services_status.append("❌ Spotify")
    
    # YouTube and Bandcamp don't need API keys
    services_status.append("✅ YouTube")
    services_status.append("✅ Bandcamp")
    
    st.markdown("### 📊 Status")
    for status in services_status:
        st.markdown(status)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Album Cover")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of an album cover for AI recognition"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Album Cover", use_column_width=True)
        
        # Process button
        if st.button("🔍 Identify Album", disabled=st.session_state.processing):
            if not gemini_key:
                st.error("❌ Gemini API key is required for album identification!")
            else:
                st.session_state.processing = True
                st.rerun()

with col2:
    st.markdown("### 🎯 Results")
    
    if st.session_state.processing and uploaded_file is not None:
        with st.spinner("🤖 Analyzing album cover with AI..."):
            try:
                # Initialize collectors
                collectors = {}
                
                if gemini_key:
                    collectors['gemini'] = GeminiVisionCollector(gemini_key)
                
                if discogs_token:
                    collectors['discogs'] = DiscogsCollector(discogs_token)
                
                if spotify_client_id and spotify_client_secret:
                    collectors['spotify'] = SpotifyCollector(spotify_client_id, spotify_client_secret)
                
                collectors['youtube'] = YouTubeCollector()
                collectors['bandcamp'] = BandcampCollector()
                
                # Process image with Gemini
                if 'gemini' in collectors:
                    image_bytes = uploaded_file.getvalue()
                    
                    # Get AI identification
                    gemini_result = asyncio.run(collectors['gemini'].identify_album(image_bytes))
                    
                    if gemini_result and gemini_result.get('album') and gemini_result.get('artist'):
                        album_name = gemini_result['album']
                        artist_name = gemini_result['artist']
                        confidence = gemini_result.get('confidence', 0)
                        
                        st.success(f"🎵 **Identified:** {artist_name} - {album_name}")
                        
                        # Get additional data from other services
                        results = {'gemini': gemini_result}
                        
                        # Search Discogs
                        if 'discogs' in collectors:
                            try:
                                discogs_result = asyncio.run(collectors['discogs'].search_album(artist_name, album_name))
                                if discogs_result:
                                    results['discogs'] = discogs_result
                            except Exception as e:
                                st.warning(f"Discogs search failed: {e}")
                        
                        # Search Spotify
                        if 'spotify' in collectors:
                            try:
                                spotify_result = asyncio.run(collectors['spotify'].search_album(artist_name, album_name))
                                if spotify_result:
                                    results['spotify'] = spotify_result
                            except Exception as e:
                                st.warning(f"Spotify search failed: {e}")
                        
                        # Search YouTube
                        try:
                            youtube_result = asyncio.run(collectors['youtube'].search_album(artist_name, album_name))
                            if youtube_result:
                                results['youtube'] = youtube_result
                        except Exception as e:
                            st.warning(f"YouTube search failed: {e}")
                        
                        # Search Bandcamp
                        try:
                            bandcamp_result = asyncio.run(collectors['bandcamp'].search_album(artist_name, album_name))
                            if bandcamp_result:
                                results['bandcamp'] = bandcamp_result
                        except Exception as e:
                            st.warning(f"Bandcamp search failed: {e}")
                        
                        # Display results
                        st.session_state.result = results
                        
                    else:
                        st.error("❌ Could not identify the album. Please try a clearer image.")
                        st.session_state.result = None
                else:
                    st.error("❌ Gemini API key is required for album identification!")
                    st.session_state.result = None
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.session_state.result = None
            
            finally:
                st.session_state.processing = False
                st.rerun()
    
    # Display results
    if st.session_state.result:
        results = st.session_state.result
        
        # Main result from Gemini
        if 'gemini' in results:
            gemini_data = results['gemini']
            st.markdown("#### 🤖 AI Identification")
            st.write(f"**Artist:** {gemini_data.get('artist', 'Unknown')}")
            st.write(f"**Album:** {gemini_data.get('album', 'Unknown')}")
            st.write(f"**Year:** {gemini_data.get('year', 'Unknown')}")
            st.write(f"**Genre:** {gemini_data.get('genre', 'Unknown')}")
            
            confidence = gemini_data.get('confidence', 0)
            if confidence >= 90:
                st.markdown(f'<span class="confidence-high">Confidence: {confidence}%</span>', unsafe_allow_html=True)
            elif confidence >= 70:
                st.markdown(f'<span class="confidence-medium">Confidence: {confidence}%</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="confidence-low">Confidence: {confidence}%</span>', unsafe_allow_html=True)
        
        # Additional service results
        st.markdown("#### 🔗 Links & Additional Info")
        
        # Discogs
        if 'discogs' in results:
            discogs_data = results['discogs']
            st.markdown("**💿 Discogs:**")
            if discogs_data.get('url'):
                st.markdown(f"[View on Discogs]({discogs_data['url']})")
            if discogs_data.get('price_info'):
                st.write(f"Price info: {discogs_data['price_info']}")
        
        # Spotify
        if 'spotify' in results:
            spotify_data = results['spotify']
            st.markdown("**🎵 Spotify:**")
            if spotify_data.get('url'):
                st.markdown(f"[Listen on Spotify]({spotify_data['url']})")
            if spotify_data.get('tracks'):
                with st.expander("View Tracklist"):
                    for i, track in enumerate(spotify_data['tracks'][:10], 1):
                        st.write(f"{i}. {track}")
        
        # YouTube
        if 'youtube' in results:
            youtube_data = results['youtube']
            st.markdown("**📺 YouTube:**")
            if youtube_data.get('url'):
                st.markdown(f"[Watch on YouTube]({youtube_data['url']})")
        
        # Bandcamp
        if 'bandcamp' in results:
            bandcamp_data = results['bandcamp']
            st.markdown("**🎶 Bandcamp:**")
            if bandcamp_data.get('url'):
                st.markdown(f"[Buy on Bandcamp]({bandcamp_data['url']})")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Powered by AI • Made with ❤️ for music lovers</p>',
    unsafe_allow_html=True
)