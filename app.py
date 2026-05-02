import gradio as gr
import requests
import base64
from PIL import Image
import io
import os

BACKEND_URL = "http://localhost:7860"  # Adjust for Spaces

def search_text(query, category, limit):
    params = {"query": query, "limit": int(limit)}
    if category:
        params["category"] = category
    try:
        res = requests.get(f"{BACKEND_URL}/api/search", params=params)
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"Status {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Add more functions for image search, etc.

with gr.Blocks() as demo:
    gr.Markdown("# Fashion Multi-Modal RAG")
    with gr.Tab("Text Search"):
        query = gr.Textbox(label="Query")
        category = gr.Dropdown(["", "shirt", "pants", "dress", "shoes", "jacket"], label="Category")
        limit = gr.Slider(1, 20, value=5, label="Limit")
        output = gr.JSON()
        btn = gr.Button("Search")
        btn.click(search_text, inputs=[query, category, limit], outputs=output)

demo.launch()