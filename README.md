# 🎵 Crate-Mate: Enhanced Vinyl Record Recognition

An AI-powered companion tool for vinyl diggers and DJs. Snap a photo of any record cover and instantly get listening links across multiple platforms, detailed metadata, and more!

Based on a previous fork but now substantially rebuilt with key enhancements:
- 🆕 **YouTube integration** - Find albums not on Spotify!
- 🎯 **Multi-platform listening links** - Spotify, YouTube, and more coming soon
- 📸 **Custom-trained AI model** - ~90% accuracy on vinyl recognition
- 🎨 **Background removal** - Clean album art extraction
- 🚀 **Production-ready** - Docker-based deployment

## Project Structure

The project is containerized with Docker to increase compatibility across platforms and to simplify setup and deployment. The key services include:

- **Frontend**: React app for user interaction.
- **Backend**: Python (FastAPI) service for API endpoints, image recognition and metadata aggregation.
- **Database**: PostgreSQL with `pgvector` for vectorized queries. pgAdmin is available for browsing the database in a web browser.
- **Nginx**: Reverse proxy for routing traffic between services.

Architecture and sequence diagrams available in `diagrams/png`.

## Getting Started

### 1) Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for local dev) and Node 20 (optional)

### 2) Environment variables
Copy `.env.example` to `.env` and fill in the values you have:

- `DISCOGS_TOKEN` (required)
- `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` (optional)
- `GEMINI_API_KEY` (optional but recommended)
- `GOOGLE_APPLICATION_CREDENTIALS` (optional)
  - IMPORTANT: keep your JSON service account key OUTSIDE this repo. Put it under a secure path (e.g., `~/keys/crate-mate-gcp.json`) and point the var to that absolute path. The repo’s `.gitignore` already excludes common secret locations.
- `YOUTUBE_API_KEY` (optional; scraping fallback is used if absent)

### 3) Run with Docker
```bash
docker compose up -d --build
```
Open `http://localhost`.

### 4) Local dev (optional)
- Backend: FastAPI under `backend/`
- Frontend: React under `frontend/`

## Security notes
- Do not commit secrets. `.env` is ignored; use `.env.example` for documentation.
- The GCP JSON key file must not live inside the repo. Rotate any keys that may have been committed in the past.

## Development

### Debugging

To debug a specific container, you can view its logs using:

```bash
docker logs -f vinyl-backend
```

Note that you need to use the *container* name instead of the *service* name. Service names come from `docker compose.yml` to refer to a component of the app's architecture. *Container names* are actual instances of that service. Thus, to view service logs, we want to peek into the *container* actually running the service.

It's also important to note that `-f` attaches the logs to your terminal window and will update in real-time. Detach using `CTRL-C`. If `-f` is ommitted, then you will see the logs up to the point of the command being run (and nothing after).

### Frontend (React)

1. Navigate to the frontend directory:

    ```sh
    cd frontend
    ```

2. Install dependencies:

    ```sh
    npm install
    ```

3. Rebuild the frontend in Docker (if needed):

    For instances where you modify the Dockerfile for the frontend or update any configuration in docker-compose.yml that impacts the frontend service (e.g., ports, environment variables, volumes, or build context).

    If you update `package.json` (e.g., add, remove, or update npm packages), Docker needs to re-install dependencies, which requires rebuilding the container.

    ```sh
    cd frontend
    npm install
    docker compose up -d --build frontend
    ```

### Backend

Changes made in `/backend` are automatically reflected in the running backend service (after detecting a change in any backend file - remember to save!). If you need to rebuild the backend service for any reason:

```bash
docker compose up -d --build backend
```

### Executing commands inside a running container

To access a running container, run the following, replacing `vinyl-backend` with the name of the container (found in `docker compose.yml` or by running `docker ps`).

```bash
docker exec -it vinyl-backend /bin/bash
```

### Database

To modify the database schema:

1. Edit `database/init.sql`

2. Drop the local `postgres_data` Docker volume:

    ```sh
    docker compose down
    docker volume rm crate-mate_postgres_data
    ```

3. Rebuild the database container:

    ```sh
    docker compose up -d --build database
    ```

### Update Nginx config without rebuilding

To update the Nginx config, make the necessary changes locally and then run:

```bash
docker exec vinyl-nginx nginx -s reload
```

## Contributing Guidelins

1. Always create feature branches for new work:

    ```sh
    git checkout -b feature/new-feature-name
    ```

2. Commit changes with descriptive messages:

    ```sh
    git commit -m "Fix: Add health check for backend service"
    ```

3. Submit pull requests for review. Include:

      - A clear summary of changes
      - Testing instructions

4. Run code formatters (e.g., black for Python, Prettier for JS) before committing.