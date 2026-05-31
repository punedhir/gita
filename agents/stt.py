import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")


def transcribe_audio(filepath: str) -> str:
    """Transcribe a recorded or uploaded audio file via Groq Whisper."""
    if not filepath:
        return ""
    with open(filepath, "rb") as file:
        transcript = groq.audio.transcriptions.create(
            model="whisper-large-v3",
            temperature=0,
            file=file,
        )
    return (transcript.text or "").strip()
