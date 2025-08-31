#!/bin/bash

echo "🎵 Crate-Mate Setup Script"
echo "========================="
echo ""
echo "Enhanced vinyl record recognition with multi-platform listening links!"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .sample.env .env
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is installed and running"
echo ""

# Prompt for API keys
echo "=== API Key Configuration ==="
echo "Please provide your API keys (or press Enter to skip):"
echo ""

# Discogs
current_discogs=$(grep "^DISCOGS_TOKEN=" .env | cut -d'=' -f2)
if [ -z "$current_discogs" ]; then
    read -p "Discogs Token: " discogs_token
    if [ ! -z "$discogs_token" ]; then
        sed -i '' "s/DISCOGS_TOKEN=.*/DISCOGS_TOKEN=$discogs_token/" .env
        echo "✅ Discogs token added"
    fi
else
    echo "✅ Discogs token already configured"
fi

# Spotify
current_spotify_id=$(grep "^SPOTIFY_CLIENT_ID=" .env | cut -d'=' -f2)
if [ -z "$current_spotify_id" ]; then
    read -p "Spotify Client ID: " spotify_id
    if [ ! -z "$spotify_id" ]; then
        sed -i '' "s/SPOTIFY_CLIENT_ID=.*/SPOTIFY_CLIENT_ID=$spotify_id/" .env
        echo "✅ Spotify Client ID added"
    fi
    
    read -p "Spotify Client Secret: " spotify_secret
    if [ ! -z "$spotify_secret" ]; then
        sed -i '' "s/SPOTIFY_CLIENT_SECRET=.*/SPOTIFY_CLIENT_SECRET=$spotify_secret/" .env
        echo "✅ Spotify Client Secret added"
    fi
else
    echo "✅ Spotify credentials already configured"
fi

# YouTube (NEW!)
current_youtube=$(grep "^YOUTUBE_API_KEY=" .env | cut -d'=' -f2)
if [ -z "$current_youtube" ]; then
    echo ""
    echo "🆕 YouTube API (for listening links not on Spotify!)"
    echo "   Get your API key from: https://console.cloud.google.com/apis/credentials"
    echo "   Enable: YouTube Data API v3"
    read -p "YouTube API Key: " youtube_key
    if [ ! -z "$youtube_key" ]; then
        sed -i '' "s/YOUTUBE_API_KEY=.*/YOUTUBE_API_KEY=$youtube_key/" .env
        echo "✅ YouTube API key added"
    fi
else
    echo "✅ YouTube API already configured"
fi

echo ""
echo "=== Starting Crate-Mate ==="
echo ""

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker compose down 2>/dev/null

# Build and start the containers
echo "Building and starting services..."
docker compose up -d --build

echo ""
echo "⏳ Waiting for services to start (this may take a minute)..."
sleep 15

# Check if services are running
if docker compose ps | grep -q "running"; then
    echo ""
    echo "✨ Crate-Mate is running! ✨"
    echo ""
    echo "🌐 Access the services:"
    echo "   📸 Frontend (Upload vinyl photos): http://localhost/"
    echo "   🔧 Backend API: http://localhost/api/"
    echo "   🗄️ pgAdmin (Database): http://localhost/pga/"
    echo "       Username: admin@vinyl.com"
    echo "       Password: admin"
    echo ""
    echo "🎵 Features:"
    echo "   ✅ AI-powered vinyl recognition"
    echo "   ✅ Multi-platform listening links (Spotify + YouTube)"
    echo "   ✅ Background removal for clean album covers"
    echo "   ✅ Metadata from multiple sources"
    echo ""
    echo "📝 Useful commands:"
    echo "   View logs: docker compose logs -f"
    echo "   Stop: docker compose down"
    echo "   Restart: docker compose restart"
else
    echo ""
    echo "⚠️ Something went wrong. Check the logs with:"
    echo "   docker compose logs"
    echo ""
    echo "Common issues:"
    echo "   - Port 80 already in use (stop other web servers)"
    echo "   - Docker Desktop not running"
    echo "   - Not enough disk space"
fi
