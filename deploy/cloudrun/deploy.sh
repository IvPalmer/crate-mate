#!/usr/bin/env bash
set -euo pipefail

# Deploy Crate‑Mate backend to Google Cloud Run using gcloud + Cloud Build.
# Prereqs: gcloud installed and authed (gcloud auth login), project selected or passed via PROJECT_ID.
# Usage:
#   PROJECT_ID=your-project REGION=us-central1 ./deploy/cloudrun/deploy.sh

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-crate-mate-api}
REPO=${REPO:-crate-mate}
IMAGE=${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Export PROJECT_ID or pass inline: PROJECT_ID=... ./deploy.sh" >&2
  exit 1
fi

echo "Using project: ${PROJECT_ID}, region: ${REGION}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "Enabling required services..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com >/dev/null

echo "Creating Artifact Registry repo (if missing)..."
gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Crate‑Mate Docker images"

echo "Reading env vars from .env (if present)..."
DISCOGS_TOKEN=$(grep -E '^DISCOGS_TOKEN=' .env 2>/dev/null | cut -d'=' -f2- || true)
SPOTIFY_CLIENT_ID=$(grep -E '^SPOTIFY_CLIENT_ID=' .env 2>/dev/null | cut -d'=' -f2- || true)
SPOTIFY_CLIENT_SECRET=$(grep -E '^SPOTIFY_CLIENT_SECRET=' .env 2>/dev/null | cut -d'=' -f2- || true)
GEMINI_API_KEY=$(grep -E '^GEMINI_API_KEY=' .env 2>/dev/null | cut -d'=' -f2- || true)
YOUTUBE_API_KEY=$(grep -E '^YOUTUBE_API_KEY=' .env 2>/dev/null | cut -d'=' -f2- || true)
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-https://crate-mate.streamlit.app}

echo "Building and pushing image with Cloud Build..."
gcloud builds submit ./backend --tag "${IMAGE}"

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8000

echo "Applying environment variables..."
gcloud run services update "${SERVICE}" \
  --region "${REGION}" \
  --update-env-vars DISCOGS_TOKEN="${DISCOGS_TOKEN}",SPOTIFY_CLIENT_ID="${SPOTIFY_CLIENT_ID}",SPOTIFY_CLIENT_SECRET="${SPOTIFY_CLIENT_SECRET}",GEMINI_API_KEY="${GEMINI_API_KEY}",YOUTUBE_API_KEY="${YOUTUBE_API_KEY}",ALLOWED_ORIGINS="${ALLOWED_ORIGINS}"

URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
echo "\nDeployed: ${URL}"
echo "Set this in Streamlit Secrets as API_BASE_URL"


