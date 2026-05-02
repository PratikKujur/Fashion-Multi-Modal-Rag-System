import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Fashion Multi-Modal RAG", page_icon="👗", layout="wide")

def check_backend():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return res.status_code == 200
    except:
        return False

def display_result_item(item, show_description=False):
    col1, col2 = st.columns([1, 3])
    with col1:
        if item.get("image_path") and os.path.exists(item["image_path"]):
            img = Image.open(item["image_path"])
            st.image(img, width=150)
        else:
            st.write("🖼️ No image")
    with col2:
        st.subheader(item.get("name", item.get("item_id", "Item")))
        st.write(f"Category: {item.get('category', 'N/A')}")
        st.write(f"Score: {item.get('score', 0):.4f}")
        if show_description and item.get("description"):
            st.write(item["description"])

if not check_backend():
    st.error("⚠️ Backend not reachable. Start FastAPI first: `python main.py`")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose", ["Text Search", "Image Search", "Recommendations", "Chat"])

if page == "Text Search":
    st.title("🔍 Text Search")
    query = st.text_input("Search query", placeholder="red summer dress")
    category = st.selectbox("Category", ["", "shirt", "pants", "dress", "shoes", "jacket"])
    limit = st.slider("Results", 1, 20, 5)

    if st.button("Search"):
        params = {"query": query, "limit": limit}
        if category:
            params["category"] = category
        res = requests.get(f"{BACKEND_URL}/api/search", params=params)
        if res.status_code == 200:
            data = res.json()
            st.write(f"Found {data.get('count', 0)} results for: {data.get('query', '')}")
            for item in data.get("results", []):
                display_result_item(item, show_description=True)
        else:
            st.error("Search failed")

elif page == "Image Search":
    st.title("📸 Image Search")
    img_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    limit = st.slider("Results", 1, 20, 5)

    if img_file and st.button("Search Similar"):
        files = {"image": img_file.getvalue()}
        params = {"limit": limit}
        res = requests.post(f"{BACKEND_URL}/api/search/image", files=files, params=params)
        if res.status_code == 200:
            data = res.json()
            st.write(f"Found {data.get('count', 0)} results")
            for item in data.get("results", []):
                display_result_item(item)
        else:
            st.error("Image search failed")

elif page == "Recommendations":
    st.title("👗 Recommendations")
    item_id = st.text_input("Item ID")
    limit = st.slider("Recommendations", 1, 20, 5)

    if st.button("Get Recommendations"):
        res = requests.get(f"{BACKEND_URL}/api/recommend", params={"item_id": item_id, "limit": limit})
        if res.status_code == 200:
            data = res.json()
            st.write(f"Recommendations for {data.get('item_id', '')}")
            for item in data.get("recommendations", []):
                display_result_item(item)
        else:
            st.error("Failed to get recommendations")

elif page == "Chat":
    st.title("💬 Fashion Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "retrieved_items" in msg:
                for item in msg["retrieved_items"]:
                    display_result_item(item)

    if prompt := st.chat_input("Ask about fashion..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        res = requests.get(f"{BACKEND_URL}/api/chat", params={"message": prompt})
        if res.status_code == 200:
            data = res.json()
            reply = data["response"]
            retrieved = data.get("retrieved_items", [])
            st.session_state.messages.append({"role": "assistant", "content": reply, "retrieved_items": retrieved})
            st.rerun()
