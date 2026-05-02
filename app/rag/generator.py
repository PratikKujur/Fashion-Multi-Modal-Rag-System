from app.core.config import get_settings
from app.core.logging import logger

class LLMGenerator:
    def __init__(self):
        settings = get_settings()
        self.use_groq = bool(settings.GROQ_API_KEY)
        if self.use_groq:
            try:
                from groq import Groq
                self.client = Groq(api_key=settings.GROQ_API_KEY)
                self.model = settings.GROQ_MODEL
                logger.info(f"Using Groq LLM: {self.model}")
            except Exception as e:
                logger.error(f"Error initializing Groq: {e}")
                self.use_groq = False
        if not self.use_groq:
            self.hf_token = settings.HF_API_TOKEN
            self.model = settings.HF_LLM_MODEL
            logger.info(f"Using HF API LLM: {self.model}")

    def generate(self, prompt, context):
        full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
        if self.use_groq:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq error: {e}")
                return "Error: Could not generate response."
        return "LLM response placeholder"
