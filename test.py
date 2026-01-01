import os
import random
import requests
import numpy as np
import sounddevice as sd
import torch
import openai
import webbrowser
import scipy.io.wavfile as wavfile
from datetime import datetime
from vosk import Model, KaldiRecognizer
import json
import re
from dotenv import load_dotenv

# ===================== INIT =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

SAMPLE_RATE = 16000
DEVICE = "cpu"
city = "Ижевск"

DIALOG_HISTORY = []
MAX_HISTORY = 5

VOSK_MODEL_PATH = "vosk-model-small-ru"

SYSTEM_PROMPT = (
    "Ты Джарвис. Отвечай кратко, уверенно, с иронией. "
    "Всегда обращайся «сэр»."
)

MUSIC_LINKS = [
    "https://www.youtube.com/watch?v=ZYAPgPH9hsI",
    "https://www.youtube.com/watch?v=BN1WwnEDWAM",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]

WEATHER_REPLACE = {
    "-": "минус ", "°C": " градусов", "C": " градусов",
    "0": " ноль", "1": " один", "2": " два", "3": " три",
    "4": " четыре", "5": " пять", "6": " шесть",
    "7": " семь", "8": " восемь", "9": " девять"
}

HOURS = {
    0: "полночь", 1: "час", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь",
    9: "девять", 10: "десять", 11: "одиннадцать",
    12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
    15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать",
    18: "восемнадцать", 19: "девятнадцать", 20: "двадцать",
    21: "двадцать один", 22: "двадцать два", 23: "двадцать три"
}

# ===================== LOAD MODELS =====================
print("🔊 Загружаю Vosk...")
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError("Скачай vosk-model-small-ru")

vosk_model = Model(VOSK_MODEL_PATH)

print("🗣️ Загружаю Silero TTS...")
tts_model, _ = torch.hub.load(
    'snakers4/silero-models',
    'silero_tts',
    language='ru',
    speaker='v4_ru'
)
tts_model.to(DEVICE)

# ===================== AUDIO =====================
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def speak(text: str):
    if not text.strip():
        return

    text = text.replace("Джарвис", "Дж+арвис").replace("сэр", "с+эр")

    for k, v in WEATHER_REPLACE.items():
        text = text.replace(k, v)

    for sentence in split_sentences(text):
        audio = tts_model.apply_tts(
            text=sentence,
            speaker='eugene',
            sample_rate=48000,
            put_accent=True,
            put_yo=True
        )
        sd.play(audio, samplerate=48000)
        sd.wait()

def record_vad(
    max_seconds=6,
    silence_threshold=500,
    silence_duration=0.8
):
    print("🎤 Говорите...")
    chunk_duration = 0.1
    chunk_size = int(SAMPLE_RATE * chunk_duration)

    audio = []
    silence_chunks = 0
    max_chunks = int(max_seconds / chunk_duration)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    stream.start()

    for _ in range(max_chunks):
        chunk, _ = stream.read(chunk_size)
        audio.append(chunk)

        volume = np.abs(chunk).mean()
        silence_chunks = silence_chunks + 1 if volume < silence_threshold else 0

        if silence_chunks * chunk_duration > silence_duration:
            break

    stream.stop()
    stream.close()

    return np.concatenate(audio).tobytes()

def stt(audio_bytes):
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.AcceptWaveform(audio_bytes)
    return json.loads(rec.Result()).get("text", "").strip()

# ===================== INFO =====================
def get_weather():
    try:
        r = requests.get(f"https://wttr.in/{city}?format=%C+%t&lang=ru", timeout=6)
        return f"Погода в {city}: {r.text.strip()}, сэр."
    except:
        return "Погода недоступна, сэр."

def get_time():
    h, m = datetime.now().hour, datetime.now().minute
    return f"Сейчас {HOURS[h]} {m} минут, сэр."

# ===================== GPT =====================
def gpt_query(text):
    global DIALOG_HISTORY

    DIALOG_HISTORY.append({"role": "user", "content": text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *DIALOG_HISTORY[-MAX_HISTORY * 2:]
    ]

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=120,
        temperature=0.7,
        timeout=10
    )

    answer = resp.choices[0].message.content
    DIALOG_HISTORY.append({"role": "assistant", "content": answer})
    return answer

# ===================== MAIN =====================
if __name__ == "__main__":
    speak("Джарвис онлайн. Готов к вашим распоряжениям, сэр.")

    exit_cmds = ["выход", "стоп", "пока"]
    STOP = False

    while True:
        audio = record_vad()
        text = stt(audio)

        if not text:
            continue

        print("👤:", text)
        lower = text.lower()

        if any(x in lower for x in exit_cmds):
            speak("Отключаюсь. Всего доброго, сэр.")
            break

        if "музык" in lower:
            webbrowser.open(random.choice(MUSIC_LINKS))
            speak("Музыка запущена, сэр.")
            STOP = True
            continue

        if "погода" in lower:
            speak(get_weather())
            continue

        if "время" in lower or "который час" in lower:
            speak(get_time())
            continue

        answer = gpt_query(text)
        print("🤖:", answer)
        speak(answer)
