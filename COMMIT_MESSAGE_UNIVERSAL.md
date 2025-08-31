# Commit Message

```
feat: Add universal search fallback for vinyl identification

BREAKING CHANGE: Crate-mate now searches the ENTIRE Discogs catalog
instead of being limited to pre-loaded database

- Add universal search when vector matching fails
- Implement OCR text extraction with pytesseract  
- Search entire Discogs catalog (millions of records)
- Add YouTube link integration to results
- Smart query building (artist/album, catalog numbers)
- Keep fast vector search as primary, universal as fallback

This fixes the major limitation where a previous fork could only
identify ~1000 pre-loaded albums. Now ANY vinyl record in
Discogs can be identified!

Files changed:
- backend/app/main.py - Add universal search fallback
- backend/app/simple_universal_search.py - New universal search module
- backend/app/collectors/youtube.py - YouTube integration
- backend/requirements.txt - Add pytesseract
- backend/Dockerfile - Add tesseract-ocr system package
```

## To test:
1. `docker compose down`
2. `docker compose up -d --build`
3. Upload vinyl images at http://localhost/
4. Records that previously failed will now be found!
