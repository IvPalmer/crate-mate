# Suggested Commit Message

```
feat: Add YouTube integration and enhance setup process

- Add YouTube collector for multi-platform listening links
- Create automated setup script (setup_crate_mate.sh)
- Update README with crate-mate branding and enhancements
- Add YouTube API key to .sample.env
- Implement confidence scoring for YouTube matches
- Prioritize official channels and full album uploads

Many vinyl releases aren't available on Spotify, so YouTube
integration provides an important fallback for listening links.
```

## Files Changed:
- `backend/app/collectors/youtube.py` (new)
- `backend/app/metadata_orchestrator.py` (modified)
- `.sample.env` (modified)
- `README.md` (modified)
- `setup_crate_mate.sh` (new)
- `ENHANCEMENT_PLAN.md` (new)

## Next Steps:
1. Test the YouTube integration
2. Add more streaming platforms (Apple Music, Bandcamp)
3. Enhance UI to display multi-platform links
4. Add caching for API responses
