import os
import requests
import streamlit as st
from io import BytesIO
from PIL import Image, ImageOps


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

      /* Square camera preview container */
      [data-testid="stCameraInput"] {
        max-width: 500px;
        margin: 0 auto;
        position: relative;
      }
      
      /* Create a wrapper for the video that's square */
      [data-testid="stCameraInput"] > div:nth-child(2) {
        position: relative !important;
        width: 100% !important;
        padding-bottom: 100% !important;
        height: 0 !important;
        overflow: visible !important;
        border-radius: 12px !important;
        background: #000 !important;
      }
      
      /* Video element - absolute within square container */
      [data-testid="stCameraInput"] video,
      [data-testid="stCameraInput"] canvas {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        border-radius: 12px !important;
      }
      
      /* Take Photo button - move it way down with absolute positioning */
      [data-testid="stCameraInput"] button:not([aria-label*="Switch"]) {
        position: absolute !important;
        top: calc(100% + 16px) !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        z-index: 10 !important;
      }
      
      /* Add padding to camera input to accommodate button */
      [data-testid="stCameraInput"] {
        padding-bottom: 60px !important;
      }
      
      /* Hide tooltip text */
      [data-testid="stTooltipHoverTarget"] span {
        display: none !important;
      }
      
      /* Switch camera button */
      [data-testid="stCameraInput"] button[aria-label*="Switch"] {
        position: absolute !important;
        top: 12px !important;
        right: 12px !important;
        z-index: 100 !important;
        background: rgba(0, 0, 0, 0.6) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        padding: 0 !important;
      }
      
      /* Force button text to be visible */
      [data-testid="stCameraInput"] button:not([aria-label*="Switch"]) p {
        margin: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

camera_supported = hasattr(st, "camera_input")

if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if camera_supported and st.button("Use camera", use_container_width=True):
        st.session_state.show_camera = True
with btn_col2:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], label_visibility="visible")

camera_img = None
if camera_supported and st.session_state.show_camera:
    # Render camera inside a container to avoid layout glitches on iOS
    cam_container = st.container()
    with cam_container:
        # JavaScript to force back camera
        st.markdown(
            """
            <script>
              // Aggressive approach to force back camera
              (function() {
                let attemptCount = 0;
                const maxAttempts = 10;
                
                // Function to switch to back camera
                function switchToBackCamera() {
                  attemptCount++;
                  
                  // Find all video elements
                  const videos = document.querySelectorAll('[data-testid="stCameraInput"] video');
                  
                  if (videos.length > 0) {
                    videos.forEach(video => {
                      // Check if video has a stream
                      if (video.srcObject) {
                        // Get current video track settings
                        const tracks = video.srcObject.getVideoTracks();
                        if (tracks.length > 0) {
                          const settings = tracks[0].getSettings();
                          
                          // Check if using front camera (facingMode might be 'user')
                          if (!settings.facingMode || settings.facingMode === 'user') {
                            // Stop current tracks
                            tracks.forEach(track => track.stop());
                            
                            // Request back camera
                            navigator.mediaDevices.getUserMedia({ 
                              video: { 
                                facingMode: { exact: 'environment' },
                                width: { ideal: 1280 },
                                height: { ideal: 1280 }
                              } 
                            }).then(stream => {
                              video.srcObject = stream;
                              console.log('Switched to back camera');
                            }).catch(err => {
                              // Try without exact constraint
                              navigator.mediaDevices.getUserMedia({ 
                                video: { 
                                  facingMode: 'environment',
                                  width: { ideal: 1280 },
                                  height: { ideal: 1280 }
                                } 
                              }).then(stream => {
                                video.srcObject = stream;
                                console.log('Switched to back camera (fallback)');
                              }).catch(err2 => {
                                console.log('Could not switch to back camera:', err2);
                                // As last resort, click the switch button
                                const switchBtn = document.querySelector('[data-testid="stCameraInput"] button[aria-label*="Switch"]');
                                if (switchBtn && attemptCount === 1) {
                                  setTimeout(() => switchBtn.click(), 100);
                                }
                              });
                            });
                          }
                        }
                      } else if (attemptCount < maxAttempts) {
                        // Video not ready, try again
                        setTimeout(switchToBackCamera, 200);
                      }
                    });
                  } else if (attemptCount < maxAttempts) {
                    // No video elements yet, try again
                    setTimeout(switchToBackCamera, 200);
                  }
                }
                
                // Start checking immediately and after delays
                switchToBackCamera();
                setTimeout(switchToBackCamera, 500);
                setTimeout(switchToBackCamera, 1000);
                
                // Also override getUserMedia for future calls
                if (!window._backCameraOverride) {
                  window._backCameraOverride = true;
                  const original = navigator.mediaDevices.getUserMedia;
                  navigator.mediaDevices.getUserMedia = function(constraints) {
                    if (constraints && constraints.video) {
                      if (typeof constraints.video === 'object') {
                        constraints.video.facingMode = 'environment';
                      } else {
                        constraints.video = { facingMode: 'environment' };
                      }
                    }
                    return original.call(this, constraints);
                  };
                }
              })();
              
              // Debug and fix button placement
              function debugCameraStructure() {
                const cameraInput = document.querySelector('[data-testid="stCameraInput"]');
                if (!cameraInput) return;
                
                console.log('Camera structure:', cameraInput);
                
                // Find all buttons
                const buttons = cameraInput.querySelectorAll('button');
                buttons.forEach((btn, idx) => {
                  console.log(`Button ${idx}:`, btn.textContent, btn.getAttribute('aria-label'), btn);
                  
                  // If this is the Take Photo button
                  if (btn.textContent?.includes('Take Photo') && !btn.getAttribute('aria-label')?.includes('Switch')) {
                    // Force absolute positioning via style attribute (highest specificity)
                    btn.style.cssText = `
                      position: absolute !important;
                      top: calc(100% + 16px) !important;
                      left: 0 !important;
                      right: 0 !important;
                      width: 100% !important;
                      margin: 0 !important;
                      padding: 12px !important;
                      font-size: 16px !important;
                      font-weight: 500 !important;
                      z-index: 10 !important;
                      background-color: #0068c9 !important;
                      color: white !important;
                      border: none !important;
                      border-radius: 4px !important;
                      cursor: pointer !important;
                    `;
                  }
                });
                
                // Ensure camera input has proper height
                cameraInput.style.paddingBottom = '60px';
              }
              
              // Run multiple times as DOM updates
              debugCameraStructure();
              setTimeout(debugCameraStructure, 100);
              setTimeout(debugCameraStructure, 300);
              setTimeout(debugCameraStructure, 500);
              setTimeout(debugCameraStructure, 1000);
              
              // Watch for any changes
              const observer = new MutationObserver(() => {
                debugCameraStructure();
              });
              
              setTimeout(() => {
                const cameraContainer = document.querySelector('[data-testid="stCameraInput"]');
                if (cameraContainer) {
                  observer.observe(cameraContainer, { 
                    childList: true, 
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['style']
                  });
                }
              }, 500);
            </script>
            """,
            unsafe_allow_html=True,
        )
        camera_img = st.camera_input(
            "Take a photo of the cover",
            help="We will auto-crop a square around the center to focus on the cover",
            label_visibility="collapsed"  # Hide the label to reduce clutter
        )
    # Keep camera visible until a capture is made or user switches actions

selected_file = camera_img or uploaded

def _prepare_image_bytes(file_like) -> tuple[bytes, str]:
    """Open, orient, optional square-crop, downscale and JPEG-encode the image.
    Returns (bytes, filename).
    """
    try:
        img = Image.open(file_like)
        # Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # Always center square crop to minimize background noise
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
            img = img.resize(new_size)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        payload = buf.getvalue()
        filename = getattr(file_like, "name", "capture.jpg") or "capture.jpg"
        return payload, filename
    except Exception:
        # Fallback to pass-through bytes
        raw = file_like.getvalue() if hasattr(file_like, "getvalue") else file_like.read()
        filename = getattr(file_like, "name", "upload.jpg") or "upload.jpg"
        return raw, filename

if selected_file is not None:
    # Show preview
    # Downscale large images client-side to reduce upload size (helps mobile)
    try:
        payload, fname = _prepare_image_bytes(selected_file)
        # Show what will be uploaded (already square-cropped if enabled)
        # No explicit preview to avoid large scroll on mobile
        files = {"image": (fname, payload, "image/jpeg")}
    except Exception:
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

