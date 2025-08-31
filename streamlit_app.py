import os
import requests
import streamlit as st
from io import BytesIO
from PIL import Image


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
      .small-muted { color: #9aa0a6; font-size: 0.9rem; }
      .link-badge a { padding: 4px 8px; border-radius: 6px; background: #1f2937; color: #e5e7eb !important; text-decoration: none; }
      .link-badge a:hover { background: #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload album cover image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # Show preview
    # Downscale large images client-side to reduce upload size (helps mobile)
    try:
        img = Image.open(uploaded)
        img = img.convert("RGB")
        max_dim = 1280
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        payload = buf.getvalue()
        files = {"image": (uploaded.name or "upload.jpg", payload, "image/jpeg")}
    except Exception:
        # Fallback to raw upload if processing fails
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}

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

                    link_cols = st.columns(4)
                    with link_cols[0]:
                        if discogs and discogs != "unavailable":
                            st.markdown(f"[View on Discogs]({discogs})")
                    with link_cols[1]:
                        if youtube and youtube != "unavailable":
                            st.markdown(f"[YouTube]({youtube})")
                    with link_cols[2]:
                        if spotify and spotify != "unavailable":
                            st.markdown(f"[Spotify]({spotify})")
                    with link_cols[3]:
                        if bandcamp:
                            st.markdown(f"[Bandcamp]({bandcamp})")

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
                    for t in tracks:
                        pos = t.get("position", "")
                        name = t.get("title", "")
                        dur = t.get("duration", "")
                        yt = (t.get("youtube") or {}).get("url") if t.get("youtube") else None
                        sp = (t.get("spotify") or {}).get("url") if t.get("spotify") else None

                        # Fallback search links when direct links are missing
                        if not yt and (artist_name or album_name) and name:
                            from urllib.parse import quote_plus
                            q = quote_plus(f"{artist_name} {album_name} {name}")
                            yt = f"https://www.youtube.com/results?search_query={q}"
                        if not sp and (artist_name) and name:
                            from urllib.parse import quote_plus
                            q = quote_plus(f"{artist_name} {name}")
                            sp = f"https://open.spotify.com/search/{q}"

                        c1, c2, c3 = st.columns([1,6,3])
                        with c1:
                            st.markdown(f"**{pos}**")
                        with c2:
                            main = f"{name}" + (f"  <span class='small-muted'>— {dur}</span>" if dur else "")
                            st.markdown(main, unsafe_allow_html=True)
                        with c3:
                            links_html = []
                            if yt:
                                links_html.append(f"<span class='link-badge'><a href='{yt}' target='_blank'>YouTube</a></span>")
                            if sp:
                                links_html.append(f"<span class='link-badge' style='margin-left:6px'><a href='{sp}' target='_blank'>Spotify</a></span>")
                            if links_html:
                                st.markdown("".join(links_html), unsafe_allow_html=True)

                # Alternatives if low confidence
                if data.get("confidence") and float(data["confidence"]) < 0.9:
                    alts = data.get("alternatives") or []
                    if alts:
                        st.write("### Other possible matches")
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

