# 🚨 Quick Fix for Crate-Mate

## The Problem (legacy approach)
The legacy fork only searched a tiny pre-loaded database (~1000 albums), so many records weren't found.

## The Solution
We need to bypass the vector search and directly search the entire Discogs catalog.

## Implementation Plan

### Option 1: Quick Hack (5 minutes)
1. Keep the existing UI and infrastructure
2. Replace the vector search with direct Discogs API search
3. Use the image recognition for text extraction only
4. Search the ENTIRE Discogs database

### Option 2: Proper Integration (1-2 hours)
1. Add new `/api/universal/identify` endpoint
2. Use Google Vision API for text/logo extraction
3. Smart Discogs search with multiple strategies
4. Keep all the good UI/infrastructure

### Option 3: Hybrid Approach (Best of both)
1. Try vector search first (for speed if album is in DB)
2. If no match, fall back to universal search
3. Add successful matches to the vector DB for future

## Let's Do Option 1 First!

Here's what we'll change:

```python
# Instead of this (legacy way):
vector = vectorize_image(album_cover)
matches = search_in_database(vector)  # Only ~1000 albums!

# We'll do this (crate-mate way):
info = extract_text_and_features(album_cover)
matches = search_entire_discogs(info)  # Millions of albums!
```

## Files to Modify

1. `backend/app/main.py` - Add fallback to Discogs search
2. `backend/app/collectors/discogs.py` - Enhance search capabilities
3. Create `backend/app/universal_search.py` - New search logic

## The Key Insight

A small pre-loaded "phone book" can’t cover the world; call the directory (Discogs API) instead.
