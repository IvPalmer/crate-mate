import os
import requests
import streamlit as st
from io import BytesIO
from PIL import Image


st.set_page_config(page_title="Crate‑Mate", page_icon="🎚️", layout="centered")

# Backend base URL
API_BASE_URL = (
    st.secrets.get("API_BASE_URL")
    or os.getenv("API_BASE_URL")
    or os.getenv("REACT_APP_API_BASE_URL")
    or "/api"
)

st.title("Crate‑Mate")
st.caption("Scan a record cover and get links + tracklist")

uploaded = st.file_uploader("Upload album cover image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # Show preview
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
    except Exception:
        st.warning("Couldn't preview image, but will still try to scan it.")

    with st.spinner("Identifying…"):
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
        try:
            resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.success("Match found")

                # Title
                title = f"{data.get('artist_name') or data.get('album', {}).get('artist', '')} - " \
                        f"{data.get('album_name') or data.get('album', {}).get('name', '')}"
                st.subheader(title)

                # Links
                links = data.get("links") or {}
                discogs = links.get("discogs") or data.get("discogs_url")
                spotify = links.get("spotify") or data.get("spotify_url")
                youtube = links.get("youtube") or data.get("youtube_url")
                bandcamp = links.get("bandcamp") or data.get("bandcamp_url")

                cols = st.columns(2)
                with cols[0]:
                    if discogs and discogs != "unavailable":
                        st.markdown(f"[View on Discogs]({discogs})")
                    if spotify and spotify != "unavailable":
                        st.markdown(f"[Listen on Spotify]({spotify})")
                with cols[1]:
                    if youtube and youtube != "unavailable":
                        st.markdown(f"[YouTube]({youtube})")
                    if bandcamp:
                        st.markdown(f"[Bandcamp]({bandcamp})")

                # Market
                low = data.get("lowest_price")
                copies = data.get("num_for_sale")
                currency = data.get("price_currency") or data.get("market_stats", {}).get("currency")
                if low or copies:
                    st.info(
                        f"For sale: {copies or 0} copies" + (f" • Low: {currency or ''}{low}" if low else "")
                    )

                # Tracklist
                tracks = (data.get("tracks") or {}).get("tracklist") or []
                if tracks:
                    st.write("### Tracklist")
                    for t in tracks:
                        pos = t.get("position", "")
                        name = t.get("title", "")
                        dur = t.get("duration", "")
                        line = f"{pos} {name}" + (f" — {dur}" if dur else "")
                        st.write(line)

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

