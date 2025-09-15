---
title: Crate-Mate
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🎵 Crate‑Mate

AI-powered album recognition and music discovery tool that identifies album covers and provides comprehensive information about releases.

## Features

- 🤖 **AI Album Recognition** - Uses Google Gemini Vision to identify album covers
- 🎵 **Multi-Platform Search** - Finds links on Spotify, YouTube, Discogs, and Bandcamp  
- 💿 **Release Information** - Detailed tracklist, genres, year, and market data
- 💰 **Market Insights** - Current pricing and availability from Discogs
- 🔗 **Direct Track Links** - Individual track links for Spotify and YouTube

## How to Use

1. **Enter API Keys** - Add your API keys in the sidebar:
   - Gemini API Key (required)
   - Discogs Token (optional, for enhanced data)
   - Spotify Client ID & Secret (optional, for Spotify links)

2. **Upload Image** - Upload a clear photo of an album cover

3. **Get Results** - The AI will identify the album and provide:
   - Artist and album information
   - Confidence score
   - Links to streaming platforms
   - Complete tracklist with individual track links
   - Market pricing information

## API Keys Setup

### Gemini API Key (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into the sidebar

### Discogs Token (Optional)
1. Go to [Discogs Developer Settings](https://www.discogs.com/settings/developers)
2. Generate a new token
3. Copy and paste into the sidebar

### Spotify Credentials (Optional)
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy Client ID and Client Secret into the sidebar

## Technology Stack

- **Frontend**: Streamlit
- **AI Vision**: Google Gemini Vision API
- **Music Data**: Discogs API, Spotify API
- **Video Links**: YouTube search integration
- **Additional Sources**: Bandcamp web scraping

## Privacy

- No data is stored permanently
- API keys are only used for the current session
- Images are processed temporarily and not saved

---

*Built with ❤️ for music lovers and vinyl collectors*