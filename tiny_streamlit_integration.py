import os
import streamlit as st
from PIL import Image
from tiny_integration_pipeline import full_inference

# ✅ Streamlit Setup
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="🧠 Image Captioning & Segmentation", layout="centered")
st.title("🧠 Tiny Captioning + Segmentation")
st.write("Upload an image to generate a caption and optional segmentation mask.")

# ✅ Image Upload
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption="Uploaded Image", use_container_width=True)

    # 🧠 Run inference
    with st.spinner("Generating caption and mask..."):
        try:
            caption, mask = full_inference(temp_path)
            st.subheader("📝 Caption:")
            st.write(caption)

            if mask is not None:
                st.subheader("🖼️ Segmentation Mask:")
                st.image(mask, caption="Predicted Mask", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
