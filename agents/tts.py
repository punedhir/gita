from pathlib import Path

from kokoro_onnx import Kokoro
import sounddevice as sd
import numpy as np 
from gtts import gTTS 
import pygame, tempfile, os

from agents.sanskrit_translation import iast_to_plain_english

KOKORO_DIR = Path(__file__).resolve().parent / ".kokoro"


def plain_english_for_tts(text: str) -> str:
    """Normalize IAST names/diacritics to ASCII before English TTS."""
    return iast_to_plain_english(text)


def speak_english(text) -> str | None:
    text = plain_english_for_tts(text)
    kokoro = Kokoro(
        str(KOKORO_DIR / "kokoro-v1.0.onnx"),
        str(KOKORO_DIR / "voices-v1.0.bin"),
    )

    print("Streaming Kokoro Playback")
    samples, sample_rate = kokoro.create(text, voice="af_heart", speed=0.7)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    arr = np.asarray(samples, dtype=np.float32)
    arr = np.clip(arr, -1, 1)
    arr = (arr * 32767).astype(np.int16)

    import scipy.io.wavfile as wav
    wav.write(tmp.name, sample_rate, arr)
#    sd.play(samples,sample_rate)
#    sd.wait()
    print(f"Streaming Done {tmp.name}")
    return tmp.name



def speak_sanskrit(text) -> str | None:
    tts = gTTS(text= text, lang="hi", slow=True,tld="co.in")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3",delete=False)
    tmp.close()
    tts.save(tmp.name)

    print(f"Google Playback {tmp}")
    return tmp.name
#    pygame.mixer.init()
#    pygame.mixer.music.load(tmp)
#    pygame.mixer.music.play()
#    while pygame.mixer.music.get_busy():
#        pygame.time.wait(100)
