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
import dotenv

from dotenv import load_dotenv

load_dotenv()
# ===================== НАСТРОЙКИ =====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
DEVICE = "cpu"
city = "Ижевск"
DIALOG_HISTORY = []
MAX_HISTORY = 5

# Путь к модели Vosk (измени, если папка называется иначе)
VOSK_MODEL_PATH = "vosk-model-small-ru"

openai.api_key = OPENAI_API_KEY

# Ссылки на музыку
MUSIC_LINKS = [
    "https://www.youtube.com/watch?v=ZYAPgPH9hsI",
    "https://www.youtube.com/watch?v=BN1WwnEDWAM",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]

# Замены для температуры и цифр
WEATHER_REPLACE = {
    "10": " десять", "11": " одиннадцать", "12": " двенадцать",
    "13": " тринадцать", "14": " четырнадцать", "15": " пятнадцать",
    "16": " шестнадцать", "17": " семнадцать", "18": " восемнадцать",
    "19": " девятнадцать", "20": " двадцать", "21": " двадцать один",
    "22": " двадцать два", "23": " двадцать три", "24": " двадцать четыре",
    "25": " двадцать пять", "30": " тридцать", "35": " тридцать пять", "40": " сорок",
    "-": "минус ", "C": " градусов", "°C": " градусов",
    "0": " ноль", "1": " один", "2": " два", "3": " три", "4": " четыре",
    "5": " пять", "6": " шесть", "7": " семь", "8": " восемь", "9": " девять"
}

HOURS = {
    0: "полночь", 1: "час", 2: "два", 3: "три", 4: "четыре", 5: "пять",
    6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
    11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
    15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
    19: "девятнадцать", 20: "двадцать", 21: "двадцать один", 22: "двадцать два", 23: "двадцать три"
}

# ===================== ЗАГРУЗКА МОДЕЛЕЙ =====================
print("Загружаю Vosk модель для распознавания речи...")
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"ОШИБКА: Папка с моделью не найдена: {VOSK_MODEL_PATH}")
    print("Скачай модель с https://alphacephei.com/vosk/models и положи рядом")
    exit(1)

vosk_model = Model(VOSK_MODEL_PATH)

print("Загружаю Silero TTS...")
tts_model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v4_ru')
tts_model.to(DEVICE)


# ===================== УТИЛИТЫ =====================
def play_wav(filepath: str, fallback: str = None):
    if os.path.exists(filepath):
        try:
            rate, data = wavfile.read(filepath)
            data = data.astype(np.float32) / (32768 if data.dtype == np.int16 else 2147483648)
            sd.play(data, samplerate=rate)
            sd.wait()
            return
        except Exception as e:
            print(f"[Ошибка воспроизведения]: {e}")
    if fallback:
        speak(fallback)


def speak(text: str):
    if not text.strip():
        return
    text = text.replace("Джарвис", "Дж+арвис") \
        .replace("сэр", "с+эр") \
        .replace("Сэр", "С+эр")

    for k, v in WEATHER_REPLACE.items():
        text = text.replace(k, v)

    text = " ".join(text.split())

    audio = tts_model.apply_tts(text=text, speaker='eugene', sample_rate=48000, put_accent=True, put_yo=True)
    sd.play(audio, samplerate=48000)
    sd.wait()


def record() -> bytes:
    print("🎤 Говорите...")
    rec = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return rec.tobytes()


def stt(audio_bytes: bytes) -> str:
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.AcceptWaveform(audio_bytes)
    result = json.loads(rec.Result())
    return result.get("text", "").strip()


def get_weather(city: str = "Москва") -> str:
    try:
        r = requests.get(f"https://wttr.in/{city}?format=%C+%t+%w&lang=ru&T", timeout=8)
        if r.status_code != 200:
            return "Погода недоступна, сэр."
        raw = r.text.strip()
        raw = raw.replace("☁️", "облачно").replace("☀️", "ясно") \
            .replace("🌧️", "дождь").replace("❄️", "снег").replace("🌫", "туман")
        raw = raw.replace("м/с", "метров в секунду")
        return f"Погода в городе {city}: {raw}, сэр."
    except:
        return "Нет интернета, сэр."


def get_time() -> str:
    h, m = datetime.now().hour, datetime.now().minute
    hour_str = HOURS.get(h, str(h))
    if m == 0:
        return f"Сейчас {hour_str} часов ровно, сэр."
    if m == 30:
        return f"Сейчас {hour_str} половина, сэр."
    if m == 1:
        return f"Сейчас {hour_str} час и одна минута, сэр."
    suffix = "минут" if m % 10 in (0, 5, 6, 7, 8, 9) or 11 <= m <= 19 else \
        "минута" if m % 10 == 1 else "минуты"
    return f"Сейчас {hour_str} часов и {m} {suffix}, сэр."


def gpt_query(text: str) -> str:
    global DIALOG_HISTORY
    try:
        # добавляем реплику пользователя
        DIALOG_HISTORY.append({"role": "user", "content": text})

        # берём только последние 5 сообщений
        messages = [
            {"role": "system",
             "content": "Ты Джарвис — остроумный ИИ Тони Старка из фильма железный человек. Ты - самый умный Искусственный интеллект в мире. Всегда обращайся «сэр». Стиль: уверенный, с иронией, кратко. Я - твой хозяин. "},
            *DIALOG_HISTORY[-MAX_HISTORY * 2:]  # user + assistant
        ]

        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            timeout=15
        )

        answer = resp.choices[0].message.content

        # сохраняем ответ ассистента
        DIALOG_HISTORY.append({"role": "assistant", "content": answer})

        return answer

    except Exception as e:
        return "Связь с сервером потеряна, сэр."

# ===================== MAIN =====================
if __name__ == "__main__":
    play_wav('jarvis_sounds/greet2.wav', "Дж+арвис онл+айн. Гот+ов к в+ашим распоряж+ениям, с+эр.")
    print("Джарвис активирован \n")

    exit_cmds = ["выход", "стоп", "пока", "отключайся", "выключись", "до свидания"]
    pause_cmds = ["джарвис", "продолжаем", "возобнови", "вернись"]

    STOP = False

    while True:
        audio_bytes = record()
        text = stt(audio_bytes)

        if not text:
            continue

        print("Вы сказали:", text)
        lower = text.lower()

        # Проверка на возобновление после паузы
        if STOP:
            if any(cmd in lower for cmd in pause_cmds):
                STOP = False
                play_wav('jarvis_sounds/restart.wav', "Продолжаем, сэр.")
            continue

        # Команды выхода
        if any(cmd in lower for cmd in exit_cmds):
            play_wav('jarvis_sounds/off.wav', "Отключ+аюсь. Всег+о хор+ошего, с+эр.")
            break

        # Музыка — с паузой
        if any(w in lower for w in ["включи музыку", "музыку", "включить музыку", "играй музыку", "музыка"]):
            link = random.choice(MUSIC_LINKS)
            webbrowser.open(link)
            speak("Музыка запущена, сэр. Приятного прослушивания.")
            STOP = True
            continue
        if "пауза" in lower:
            play_wav('jarvis_sounds/Как пожелаете .wav', "Отключ+аюсь. Всег+о хор+ошего, с+эр.")
            STOP = True
            continue

        # Погода
        if any(w in lower for w in ["погода", "погоду"]):
            weather = get_weather(city)
            print("Джарвис:", weather)
            speak(weather)
            continue

        # Время
        if any(w in lower for w in ["время", "который час", "сколько времени"]):
            time_str = get_time()
            print("Джарвис:", time_str)
            speak(time_str)
            continue

        # Всё остальное — через GPT
        answer = gpt_query(text)
        print("Джарвис:", answer)
        speak(answer)
