import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_text(text, max_length=512):
    return text[:max_length] if len(text) > max_length else text
