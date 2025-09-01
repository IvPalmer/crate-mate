import os
import requests
import streamlit as st
from io import BytesIO
from PIL import Image, ImageOps
import cv2
import numpy as np


st.set_page_config(page_title="Crate‑Mate", page_icon="🎚️", layout="wide")

# Backend base URL
API_BASE_URL = (
    st.secrets.get("API_BASE_URL")
    or os.getenv("API_BASE_URL")
    or os.getenv("REACT_APP_API_BASE_URL")
    or "/api"
)

st.title("Crate‑Mate")
st.caption("Scan a record cover and get links + tracklist")

# Minimal CSS tweaks for spacing/link badges
st.markdown(
    """
    <style>
      /* General */
      .small-muted { color: #9aa0a6; font-size: 0.9rem; }

      /* Top links grid */
      .links-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 6px 0 16px; }
      .links-grid a { display: block; text-align: center; padding: 8px 10px; border-radius: 8px; background: #1f2937; color: #e5e7eb !important; text-decoration: none; }
      .links-grid a:hover { background: #374151; }

      /* Tracklist rows */
      .track-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; }
      .track-pos { width: 2.6rem; flex: 0 0 auto; font-weight: 600; opacity: 0.85; }
      .track-main { flex: 1 1 auto; min-width: 0; }
      .track-actions { flex: 0 0 auto; display: flex; gap: 8px; }
      .track-actions a { padding: 4px 8px; border-radius: 6px; background: #1f2937; color: #e5e7eb !important; text-decoration: none; white-space: nowrap; }
      .track-actions a:hover { background: #374151; }

      /* Mobile tweaks */
      @media (max-width: 640px) {
        .links-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .track-pos { width: 2.2rem; }
        .track-actions a { padding: 4px 6px; font-size: 0.9rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
📸 **How to use:**
1. Take a photo of your record with your phone's camera
2. Upload the photo below
3. We'll automatically detect and crop the record cover
""")

uploaded = st.file_uploader(
    "Upload a photo of your record", 
    type=["jpg", "jpeg", "png", "webp"],
    help="Take a photo with your camera app, then upload it here"
)

selected_file = uploaded

def detect_record_cover(img):
    """Detect and extract the record cover from an image using OpenCV."""
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    original = img_cv.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edges = cv2.Canny(gray, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Look for a square/rectangular contour
    record_contour = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # If the approximated contour has 4 points, it's likely a rectangle
        if len(approx) == 4:
            # Check if it's roughly square (aspect ratio close to 1)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3:  # Allow some tolerance for perspective
                record_contour = approx
                break
    
    if record_contour is not None:
        # Order the points for perspective transform
        pts = record_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Find the top-left, top-right, bottom-right, and bottom-left points
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        # Compute the width and height of the new image
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Make it square
        max_dim = max(max_width, max_height)
        
        # Destination points for the transform
        dst = np.array([
            [0, 0],
            [max_dim - 1, 0],
            [max_dim - 1, max_dim - 1],
            [0, max_dim - 1]], dtype="float32")
        
        # Apply perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original, M, (max_dim, max_dim))
        
        # Convert back to PIL Image
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(warped_rgb)
    
    # If no record detected, return None
    return None


def _prepare_image_bytes(file_like) -> tuple[bytes, str]:
    """Open, orient, detect record cover, square-crop, downscale and JPEG-encode the image.
    Returns (bytes, filename).
    """
    try:
        img = Image.open(file_like)
        # Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # Try to detect and extract the record cover
        detected_record = detect_record_cover(img)
        
        if detected_record is not None:
            img = detected_record
            st.success("✅ Record cover detected and extracted!")
        else:
            # Fallback to center square crop if detection fails
            st.info("📷 Using center crop (tip: try to fill the frame with the record)")
            width, height = img.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            img = img.crop((left, top, left + side, top + side))

        # Downscale large images to reduce upload size (helps mobile)
        max_dim = 1280
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        payload = buf.getvalue()
        filename = getattr(file_like, "name", "capture.jpg") or "capture.jpg"
        return payload, filename
    except Exception as e:
        st.warning(f"Image processing error: {str(e)}. Using original image.")
        # Fallback to pass-through bytes
        file_like.seek(0) if hasattr(file_like, 'seek') else None
        raw = file_like.getvalue() if hasattr(file_like, "getvalue") else file_like.read()
        filename = getattr(file_like, "name", "upload.jpg") or "upload.jpg"
        return raw, filename

if selected_file is not None:
    # Process and show preview
    with st.spinner("Processing image..."):
        try:
            # Process the image
            payload, fname = _prepare_image_bytes(selected_file)
            
            # Show preview of what will be sent
            st.write("### Preview")
            preview_img = Image.open(BytesIO(payload))
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(preview_img, caption="Processed image (square cropped)", use_container_width=True)
            
            files = {"image": (fname, payload, "image/jpeg")}
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            files = {"image": (getattr(selected_file, "name", "upload.jpg"), selected_file.getvalue(), getattr(selected_file, "type", None) or "image/jpeg")}

    with st.spinner("Identifying…"):
        try:
            resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.success("Match found")

                # Basic fields
                album = data.get("album") or {}
                cover = album.get("image")
                artist_name = data.get('artist_name') or album.get('artist') or ""
                album_name = data.get('album_name') or album.get('name') or ""

                left, right = st.columns([1, 2])
                with left:
                    if cover:
                        st.image(cover, caption=None, use_container_width=True)
                with right:
                    st.subheader(f"{artist_name} - {album_name}")

                    # Links
                    links = data.get("links") or {}
                    discogs = links.get("discogs") or data.get("discogs_url")
                    spotify = links.get("spotify") or data.get("spotify_url")
                    youtube = links.get("youtube") or data.get("youtube_url")
                    bandcamp = links.get("bandcamp") or data.get("bandcamp_url")

                    grid_html_parts = ["<div class='links-grid'>"]
                    if discogs and discogs != "unavailable":
                        grid_html_parts.append(f"<a href='{discogs}' target='_blank' rel='noopener'>View on Discogs</a>")
                    if youtube and youtube != "unavailable":
                        grid_html_parts.append(f"<a href='{youtube}' target='_blank' rel='noopener'>YouTube</a>")
                    if spotify and spotify != "unavailable":
                        grid_html_parts.append(f"<a href='{spotify}' target='_blank' rel='noopener'>Spotify</a>")
                    if bandcamp:
                        grid_html_parts.append(f"<a href='{bandcamp}' target='_blank' rel='noopener'>Bandcamp</a>")
                    grid_html_parts.append("</div>")
                    st.markdown("".join(grid_html_parts), unsafe_allow_html=True)

                    # Confidence display (if provided by backend)
                    conf = data.get("confidence")
                    if conf is not None:
                        try:
                            conf_pct = f"{float(conf) * 100:.0f}%"
                        except Exception:
                            conf_pct = str(conf)
                        st.markdown(f"**Confidence:** {conf_pct}")

                # Market
                low = data.get("lowest_price")
                copies = data.get("num_for_sale")
                currency = data.get("price_currency") or data.get("market_stats", {}).get("currency") or ""
                if low or copies:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("For sale (Discogs)", value=str(copies or 0))
                    with m2:
                        if low is not None:
                            st.metric("Lowest price", value=f"{currency}{low}")

                # Tracklist
                tracks = (data.get("tracks") or {}).get("tracklist") or []
                if tracks:
                    st.write("### Tracklist")
                    from urllib.parse import quote_plus
                    rows_html = []
                    for t in tracks:
                        pos = t.get("position", "")
                        name = t.get("title", "")
                        dur = t.get("duration", "")
                        yt = (t.get("youtube") or {}).get("url") if t.get("youtube") else None
                        sp = (t.get("spotify") or {}).get("url") if t.get("spotify") else None

                        # Fallback search links when direct links are missing
                        if not yt and (artist_name or album_name) and name:
                            q = quote_plus(f"{artist_name} {album_name} {name}")
                            yt = f"https://www.youtube.com/results?search_query={q}"
                        if not sp and artist_name and name:
                            q = quote_plus(f"{artist_name} {name}")
                            sp = f"https://open.spotify.com/search/{q}"

                        title_html = name + (f"  <span class='small-muted'>— {dur}</span>" if dur else "")
                        actions_html = []
                        if yt:
                            actions_html.append(f"<a href='{yt}' target='_blank' rel='noopener'>YouTube</a>")
                        if sp:
                            actions_html.append(f"<a href='{sp}' target='_blank' rel='noopener'>Spotify</a>")

                        row_html = f"""
                        <div class='track-row'>
                            <div class='track-pos'><strong>{pos}</strong></div>
                            <div class='track-main'>{title_html}</div>
                            <div class='track-actions'>{''.join(actions_html)}</div>
                        </div>
                        """
                        rows_html.append(row_html)

                    st.markdown("\n".join(rows_html), unsafe_allow_html=True)

                # Alternatives if low confidence (expander for mobile visibility)
                try:
                    conf_val = float(data.get("confidence")) if data.get("confidence") is not None else None
                except Exception:
                    conf_val = None
                if conf_val is not None and conf_val < 0.9:
                    alts = data.get("alternatives") or []
                    if alts:
                        with st.expander("Other possible matches"):
                            for alt in alts:
                                a_title = f"{alt.get('artist','')} - {alt.get('title','')}"
                                if alt.get("discogs"):
                                    st.markdown(f"- [{a_title}]({alt['discogs']})")
                                else:
                                    st.write(f"- {a_title}")
            else:
                try:
                    msg = resp.json()
                except Exception:
                    msg = resp.text
                st.error(f"API error {resp.status_code}: {msg}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.sidebar.header("Settings")
st.sidebar.write("API base URL used:")
st.sidebar.code(API_BASE_URL)

