import streamlit as st
import os
import io
import asyncio
from PIL import Image, ImageOps
import hashlib
from typing import Dict, List, Optional
import sys
import logging
import json
from datetime import datetime

# Add collectors to path
sys.path.append('collectors')

# Import collectors
from collectors.gemini import GeminiCollector
from collectors.discogs import DiscogsCollector
from collectors.spotify import SpotifyCollector
from collectors.youtube_enhanced import YouTubeEnhancedSearch
from collectors.bandcamp import BandcampCollector

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
        padding: 1rem 0;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
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
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #1DB954;
    }
    
    .album-cover {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 300px;
        width: 100%;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .link-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .spotify-link {
        background-color: #1DB954;
        color: white;
    }
    
    .youtube-link {
        background-color: #FF0000;
        color: white;
    }
    
    .discogs-link {
        background-color: #333;
        color: white;
    }
    
    .bandcamp-link {
        background-color: #629aa0;
        color: white;
    }
    
    .track-item {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1DB954;
    }
    
    .track-links {
        margin-top: 0.5rem;
    }
    
    .track-links a {
        font-size: 0.9rem;
        margin-right: 1rem;
    }
    
    .price-info {
        background-color: #e8f5e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alternative-match {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result' not in st.session_state:
    st.session_state.result = None

# Header
st.markdown('<h1 class="main-header">🎵 Crate‑Mate</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI-powered album recognition and music discovery</p>', unsafe_allow_html=True)

# Sidebar for API keys
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    st.markdown("Enter your API keys to enable full functionality:")
    
    gemini_key = st.text_input("🤖 Gemini API Key", type="password", help="Required for AI album identification")
    discogs_token = st.text_input("💿 Discogs Token", type="password", help="For enhanced album data and pricing")
    spotify_client_id = st.text_input("🎵 Spotify Client ID", type="password", help="For Spotify track links")
    spotify_client_secret = st.text_input("🎵 Spotify Client Secret", type="password", help="For Spotify track links")
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    # Check which services are configured
    services_status = []
    if gemini_key:
        services_status.append("✅ Gemini AI")
    else:
        services_status.append("❌ Gemini AI")
    
    if discogs_token:
        services_status.append("✅ Discogs")
    else:
        services_status.append("❌ Discogs")
    
    if spotify_client_id and spotify_client_secret:
        services_status.append("✅ Spotify")
    else:
        services_status.append("❌ Spotify")
    
    services_status.append("✅ YouTube")  # Always available
    services_status.append("✅ Bandcamp")  # Always available
    
    for status in services_status:
        st.markdown(status)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📸 Upload Album Cover")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of an album cover"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("🔍 Identify Album", type="primary", disabled=st.session_state.processing):
            if not gemini_key:
                st.error("⚠️ Please enter your Gemini API key to identify albums!")
            else:
                st.session_state.processing = True
                st.rerun()

with col2:
    if st.session_state.processing and uploaded_file is not None:
        st.markdown("### 🔍 Processing...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        async def process_image(image_data, api_keys):
            """Process the uploaded image and return results"""
            try:
                # Initialize collectors
                collectors = {}
                
                status_text.text("🤖 Initializing AI services...")
                progress_bar.progress(10)
                
                if api_keys['gemini']:
                    collectors['gemini'] = GeminiCollector(api_keys['gemini'])
                
                if api_keys['discogs']:
                    collectors['discogs'] = DiscogsCollector(api_keys['discogs'])
                
                if api_keys['spotify_id'] and api_keys['spotify_secret']:
                    collectors['spotify'] = SpotifyCollector(
                        api_keys['spotify_id'], 
                        api_keys['spotify_secret']
                    )
                
                collectors['youtube'] = YouTubeEnhancedSearch()
                collectors['bandcamp'] = BandcampCollector()
                
                # Step 1: Gemini identification
                status_text.text("🎵 Identifying album with AI...")
                progress_bar.progress(30)
                
                gemini_result = await collectors['gemini'].identify_album(image_data)
                
                if not gemini_result or gemini_result.get('confidence', 0) < 30:
                    return {
                        'error': 'Could not identify the album. Please try with a clearer image.',
                        'confidence': gemini_result.get('confidence', 0) if gemini_result else 0
                    }
                
                # Step 2: Search Discogs
                status_text.text("💿 Searching Discogs database...")
                progress_bar.progress(50)
                
                discogs_results = []
                if 'discogs' in collectors:
                    artist = gemini_result.get('artist', '')
                    album = gemini_result.get('album', '')
                    if artist and album:
                        discogs_results = await collectors['discogs'].search_release(f"{artist} {album}")
                
                # Step 3: Get detailed info for best match
                best_match = None
                if discogs_results:
                    best_match = discogs_results[0]  # Take the first/best match
                    
                    status_text.text("📊 Getting detailed album information...")
                    progress_bar.progress(70)
                    
                    # Get full release details
                    if 'discogs' in collectors and best_match.get('id'):
                        detailed_info = await collectors['discogs'].get_release_details(best_match['id'])
                        if detailed_info:
                            best_match.update(detailed_info)
                
                # Step 4: Get Spotify info
                spotify_info = None
                if 'spotify' in collectors and gemini_result.get('artist') and gemini_result.get('album'):
                    status_text.text("🎵 Finding Spotify tracks...")
                    progress_bar.progress(80)
                    
                    spotify_info = await collectors['spotify'].fetch_album_details(
                        gemini_result['artist'], 
                        gemini_result['album']
                    )
                
                # Step 5: Generate YouTube links
                status_text.text("📺 Generating YouTube links...")
                progress_bar.progress(90)
                
                youtube_links = []
                if best_match and best_match.get('tracklist'):
                    youtube_links = collectors['youtube'].generate_track_links(
                        gemini_result.get('artist', ''),
                        gemini_result.get('album', ''),
                        best_match['tracklist']
                    )
                
                # Step 6: Get Bandcamp link
                bandcamp_link = None
                if gemini_result.get('artist') and gemini_result.get('album'):
                    bandcamp_link = await collectors['bandcamp'].find_release_link(
                        gemini_result['artist'],
                        gemini_result['album']
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Compile final result
                result = {
                    'identification': gemini_result,
                    'discogs': best_match,
                    'spotify': spotify_info,
                    'youtube_tracks': youtube_links,
                    'bandcamp_link': bandcamp_link,
                    'alternatives': discogs_results[1:4] if len(discogs_results) > 1 else []
                }
                
                return result
                
            except Exception as e:
                return {'error': f'Processing error: {str(e)}'}
        
        # Run the async processing
        api_keys = {
            'gemini': gemini_key,
            'discogs': discogs_token,
            'spotify_id': spotify_client_id,
            'spotify_secret': spotify_client_secret
        }
        
        # Convert image to bytes
        image = Image.open(uploaded_file)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Process the image
        result = asyncio.run(process_image(img_byte_arr, api_keys))
        
        # Store result and stop processing
        st.session_state.result = result
        st.session_state.processing = False
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()
    
    elif st.session_state.result:
        st.markdown("### 🎵 Results")
        
        result = st.session_state.result
        
        if 'error' in result:
            st.error(f"❌ {result['error']}")
            if st.button("🔄 Try Again"):
                st.session_state.result = None
                st.rerun()
        else:
            # Display identification results
            identification = result.get('identification', {})
            discogs_info = result.get('discogs', {})
            spotify_info = result.get('spotify', {})
            
            # Album cover and basic info
            col_cover, col_info = st.columns([1, 2])
            
            with col_cover:
                if discogs_info.get('cover_image'):
                    st.markdown(f'<img src="{discogs_info["cover_image"]}" class="album-cover">', unsafe_allow_html=True)
                elif identification.get('cover_url'):
                    st.markdown(f'<img src="{identification["cover_url"]}" class="album-cover">', unsafe_allow_html=True)
            
            with col_info:
                st.markdown(f"### {identification.get('artist', 'Unknown Artist')}")
                st.markdown(f"**{identification.get('album', 'Unknown Album')}**")
                
                # Confidence indicator
                confidence = identification.get('confidence', 0)
                if confidence >= 80:
                    conf_class = "confidence-high"
                    conf_icon = "🟢"
                elif confidence >= 60:
                    conf_class = "confidence-medium"
                    conf_icon = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_icon = "🔴"
                
                st.markdown(f'{conf_icon} <span class="{conf_class}">Confidence: {confidence}%</span>', unsafe_allow_html=True)
                
                # Genre and year
                if identification.get('genre'):
                    st.markdown(f"**Genre:** {identification['genre']}")
                if discogs_info.get('year'):
                    st.markdown(f"**Year:** {discogs_info['year']}")
                elif identification.get('era'):
                    st.markdown(f"**Era:** {identification['era']}")
            
            # Links section
            st.markdown("### 🔗 Listen & Buy")
            link_cols = st.columns(4)
            
            with link_cols[0]:
                if discogs_info.get('discogs_url'):
                    st.markdown(f'<a href="{discogs_info["discogs_url"]}" target="_blank" class="link-button discogs-link">💿 Discogs</a>', unsafe_allow_html=True)
            
            with link_cols[1]:
                if spotify_info and spotify_info.get('url'):
                    st.markdown(f'<a href="{spotify_info["url"]}" target="_blank" class="link-button spotify-link">🎵 Spotify</a>', unsafe_allow_html=True)
            
            with link_cols[2]:
                if discogs_info.get('videos') and len(discogs_info['videos']) > 0:
                    youtube_url = discogs_info['videos'][0].get('uri', '')
                    if youtube_url:
                        st.markdown(f'<a href="{youtube_url}" target="_blank" class="link-button youtube-link">📺 YouTube</a>', unsafe_allow_html=True)
            
            with link_cols[3]:
                if result.get('bandcamp_link'):
                    st.markdown(f'<a href="{result["bandcamp_link"]}" target="_blank" class="link-button bandcamp-link">🎸 Bandcamp</a>', unsafe_allow_html=True)
            
            # Price information
            if discogs_info.get('lowest_price') or discogs_info.get('num_for_sale'):
                st.markdown("### 💰 Market Information")
                price_col1, price_col2 = st.columns(2)
                
                with price_col1:
                    if discogs_info.get('lowest_price'):
                        st.markdown(f"**Lowest Price:** ${discogs_info['lowest_price']}")
                
                with price_col2:
                    if discogs_info.get('num_for_sale'):
                        st.markdown(f"**Copies Available:** {discogs_info['num_for_sale']}")
            
            # Tracklist
            if discogs_info.get('tracklist') or spotify_info.get('tracks'):
                st.markdown("### 🎵 Tracklist")
                
                # Combine tracklist data
                tracks_to_display = []
                
                if discogs_info.get('tracklist'):
                    for i, track in enumerate(discogs_info['tracklist']):
                        track_info = {
                            'position': track.get('position', str(i+1)),
                            'title': track.get('title', 'Unknown Track'),
                            'duration': track.get('duration', ''),
                            'youtube_link': None,
                            'spotify_link': None
                        }
                        
                        # Add YouTube link if available
                        youtube_tracks = result.get('youtube_tracks', [])
                        if i < len(youtube_tracks):
                            track_info['youtube_link'] = youtube_tracks[i].get('youtube_url')
                        
                        # Add Spotify link if available
                        if spotify_info and spotify_info.get('tracks') and i < len(spotify_info['tracks']):
                            spotify_track = spotify_info['tracks'][i]
                            track_info['spotify_link'] = spotify_track.get('url')
                        
                        tracks_to_display.append(track_info)
                
                # Display tracks
                for track in tracks_to_display:
                    with st.container():
                        st.markdown(f"""
                        <div class="track-item">
                            <strong>{track['position']}. {track['title']}</strong>
                            {f"<span style='color: #666; margin-left: 1rem;'>{track['duration']}</span>" if track['duration'] else ""}
                            <div class="track-links">
                        """, unsafe_allow_html=True)
                        
                        # Track links
                        track_link_cols = st.columns([1, 1, 3])
                        
                        with track_link_cols[0]:
                            if track['spotify_link']:
                                st.markdown(f'<a href="{track["spotify_link"]}" target="_blank" style="color: #1DB954;">🎵 Spotify</a>', unsafe_allow_html=True)
                        
                        with track_link_cols[1]:
                            if track['youtube_link']:
                                st.markdown(f'<a href="{track["youtube_link"]}" target="_blank" style="color: #FF0000;">📺 YouTube</a>', unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Alternative matches (if confidence < 90%)
            if confidence < 90 and result.get('alternatives'):
                st.markdown("### 🤔 Other Possible Matches")
                st.markdown("*Since confidence is below 90%, here are other potential matches:*")
                
                for alt in result['alternatives'][:3]:
                    with st.expander(f"🎵 {alt.get('artist', 'Unknown')} - {alt.get('title', 'Unknown')}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if alt.get('thumb'):
                                st.image(alt['thumb'], width=150)
                        
                        with col2:
                            st.markdown(f"**Artist:** {alt.get('artist', 'Unknown')}")
                            st.markdown(f"**Album:** {alt.get('title', 'Unknown')}")
                            if alt.get('year'):
                                st.markdown(f"**Year:** {alt['year']}")
                            if alt.get('format'):
                                st.markdown(f"**Format:** {', '.join(alt['format'])}")
                            if alt.get('discogs_url'):
                                st.markdown(f'<a href="{alt["discogs_url"]}" target="_blank" class="link-button discogs-link">View on Discogs</a>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Identify Another Album"):
                st.session_state.result = None
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9em;">'
    '🎵 Powered by Gemini Vision AI, Discogs, Spotify, YouTube & Bandcamp<br>'
    'Made with ❤️ for music lovers | '
    '<a href="https://github.com/ivpalmer42/crate-mate" target="_blank">GitHub</a>'
    '</p>',
    unsafe_allow_html=True
)
