import streamlit as st
import os
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import collectors
from collectors.simple_collectors import GeminiCollector, DiscogsCollector, SpotifyCollector, YouTubeCollector, BandcampCollector

# Configure Streamlit page
st.set_page_config(
    page_title="Crate-Mate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Main title
st.title("Crate-Mate")
st.write("Upload an album cover image to identify and get detailed information")

# Sidebar for API status
with st.sidebar:
    st.write("### API Status")
    
    # Initialize collectors
    collectors = {}
    
    # Gemini AI
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        try:
            collectors['gemini'] = GeminiCollector()
            st.write("✓ Gemini AI (env)")
        except Exception as e:
            st.write("✗ Gemini AI Error")
    else:
        st.write("✗ Gemini AI")
    
    # Discogs
    discogs_token = os.getenv('DISCOGS_TOKEN')
    if discogs_token:
        try:
            collectors['discogs'] = DiscogsCollector(discogs_token)
            st.write("✓ Discogs (env)")
        except Exception as e:
            st.write("✗ Discogs Error")
    else:
        st.write("✗ Discogs")
    
    # Spotify
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    if spotify_client_id and spotify_client_secret:
        try:
            collectors['spotify'] = SpotifyCollector(spotify_client_id, spotify_client_secret)
            st.write("✓ Spotify (env)")
        except Exception as e:
            st.write("✗ Spotify Error")
    else:
        st.write("✗ Spotify")
    
    # YouTube
    youtube_key = os.getenv('YOUTUBE_API_KEY')
    if youtube_key:
        try:
            collectors['youtube'] = YouTubeCollector(youtube_key)
            st.write("✓ YouTube (env)")
        except Exception as e:
            st.write("✗ YouTube Error")
    else:
        collectors['youtube'] = YouTubeCollector()  # Works without API key
        st.write("⚠ YouTube (basic)")
    
    # Bandcamp
    collectors['bandcamp'] = BandcampCollector()
    st.write("✓ Bandcamp")

# File upload
uploaded_file = st.file_uploader("Choose an album cover image", type=['png', 'jpg', 'jpeg'])

# URL input as alternative
st.write("**Or paste an image URL:**")
image_url = st.text_input("Image URL", placeholder="https://example.com/album-cover.jpg")

# Process image
image = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

elif image_url:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.session_state.uploaded_image = image
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")

# Use session state image if available
if st.session_state.uploaded_image is not None:
    image = st.session_state.uploaded_image

if image:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Album Cover", width=300)
    
    with col2:
        if st.button("Identify Album", type="primary"):
            try:
                # Resize image if too large
                processed_image = image
                if image.size[0] > 1024 or image.size[1] > 1024:
                    processed_image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                # AI identification
                gemini_result = None
                if 'gemini' in collectors:
                    with st.spinner("Analyzing album cover..."):
                        gemini_result = collectors['gemini'].identify_album(processed_image)
                
                if gemini_result and gemini_result.get('confidence', 0) > 50:
                    artist = gemini_result.get('artist', 'Unknown')
                    album = gemini_result.get('album', 'Unknown')
                    
                    st.success("Album identified successfully")
                    
                    # Basic info
                    st.write("### Album Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Artist:** {artist}")
                        st.write(f"**Album:** {album}")
                    with col2:
                        st.write(f"**Year:** {gemini_result.get('year', 'Unknown')}")
                        st.write(f"**Confidence:** {gemini_result.get('confidence', 0)}%")
                    
                    # Get Discogs data
                    discogs_result = None
                    if 'discogs' in collectors and artist != 'Unknown' and album != 'Unknown':
                        with st.spinner("Getting detailed information..."):
                            discogs_result = collectors['discogs'].search_album(artist, album)
                    
                    if discogs_result:
                        # Album details
                        st.write("### Details")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if discogs_result.get('cover_image'):
                                st.image(discogs_result['cover_image'], width=200)
                        
                        with col2:
                            if discogs_result.get('year'):
                                st.write(f"**Year:** {discogs_result['year']}")
                            if discogs_result.get('genre'):
                                st.write(f"**Genre:** {', '.join(discogs_result['genre'])}")
                            if discogs_result.get('label'):
                                st.write(f"**Label:** {discogs_result['label']}")
                            if discogs_result.get('catalog_number'):
                                st.write(f"**Catalog:** {discogs_result['catalog_number']}")
                        
                        # Tracklist - SINGLE SECTION ONLY
                        if discogs_result.get('tracklist'):
                            st.write("### Tracklist")
                            
                            # Get track data
                            spotify_tracks = []
                            youtube_tracks = []
                            
                            if 'spotify' in collectors:
                                spotify_album = collectors['spotify'].search_album(artist, album)
                                if spotify_album and spotify_album.get('tracks'):
                                    spotify_tracks = spotify_album['tracks']
                            
                            if 'youtube' in collectors:
                                track_titles = []
                                for track in discogs_result['tracklist']:
                                    if isinstance(track, dict):
                                        track_titles.append(track.get('title', ''))
                                    else:
                                        track_titles.append(str(track))
                                
                                youtube_tracks = collectors['youtube'].generate_track_links(
                                    artist, album, track_titles
                                )
                            
                            # Display tracks
                            for i, track in enumerate(discogs_result['tracklist'], 1):
                                if isinstance(track, dict):
                                    track_title = track.get('title', track.get('name', ''))
                                    track_duration = track.get('duration', '')
                                    track_position = track.get('position', str(i))
                                else:
                                    track_title = str(track)
                                    track_duration = ''
                                    track_position = str(i)
                                
                                # Track row
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.write(f"**{track_position}. {track_title}**")
                                    if track_duration:
                                        st.caption(f"Duration: {track_duration}")
                                    elif i <= len(spotify_tracks) and spotify_tracks[i-1].get('duration'):
                                        st.caption(f"Duration: {spotify_tracks[i-1]['duration']}")
                                
                                with col2:
                                    # Spotify link
                                    if i <= len(spotify_tracks) and spotify_tracks[i-1].get('spotify_url'):
                                        st.link_button("Spotify", spotify_tracks[i-1]['spotify_url'])
                                    else:
                                        st.button("Spotify", disabled=True, key=f"spotify_disabled_{i}")
                                
                                with col3:
                                    # YouTube link
                                    if i <= len(youtube_tracks) and youtube_tracks[i-1].get('youtube_url'):
                                        st.link_button("YouTube", youtube_tracks[i-1]['youtube_url'])
                                    else:
                                        st.button("YouTube", disabled=True, key=f"youtube_disabled_{i}")
                        
                        # Price info
                        if discogs_result.get('price_info'):
                            price_info = discogs_result['price_info']
                            st.write("### Market Information")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if price_info.get('lowest_price'):
                                    st.metric("Lowest Price", f"${price_info['lowest_price']}")
                            with col2:
                                if price_info.get('median_price'):
                                    st.metric("Median Price", f"${price_info['median_price']}")
                            with col3:
                                if price_info.get('num_for_sale'):
                                    st.metric("Available Copies", price_info['num_for_sale'])
                        
                        # Platform links
                        st.write("### Listen & Buy")
                        link_cols = st.columns(4)
                        
                        # Discogs
                        with link_cols[0]:
                            if discogs_result.get('discogs_url'):
                                st.link_button("Discogs", discogs_result['discogs_url'])
                        
                        # Spotify
                        with link_cols[1]:
                            if 'spotify' in collectors:
                                spotify_result = collectors['spotify'].search_album(artist, album)
                                if spotify_result and spotify_result.get('spotify_url'):
                                    st.link_button("Spotify", spotify_result['spotify_url'])
                        
                        # YouTube
                        with link_cols[2]:
                            if 'youtube' in collectors:
                                youtube_result = collectors['youtube'].search_album(artist, album)
                                if youtube_result and youtube_result.get('youtube_url'):
                                    st.link_button("YouTube", youtube_result['youtube_url'])
                        
                        # Bandcamp
                        with link_cols[3]:
                            if 'bandcamp' in collectors:
                                bandcamp_result = collectors['bandcamp'].search_album(artist, album)
                                if bandcamp_result and bandcamp_result.get('bandcamp_url'):
                                    st.link_button("Bandcamp", bandcamp_result['bandcamp_url'])
                    
                    else:
                        st.info("Basic identification successful. Enhanced data not available.")
                
                else:
                    st.error("Could not identify the album. Please try a different image.")
                
            except Exception as e:
                st.error(f"Error during identification: {str(e)}")

# Clear button
if st.session_state.uploaded_image is not None:
    if st.button("Clear Image"):
        st.session_state.uploaded_image = None
        st.rerun()