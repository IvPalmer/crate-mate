import streamlit as st

st.set_page_config(
    page_title="Crate‑Mate Test",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Crate‑Mate - Test Version")
st.write("If you can see this, the basic Streamlit app is working!")

st.sidebar.write("### Status")
st.sidebar.write("✅ Streamlit is running")
st.sidebar.write("✅ Basic functionality works")

if st.button("Test Button"):
    st.success("Button works! The app is functional.")
    st.balloons()

st.write("---")
st.write("This is a minimal test version to verify the deployment works.")

