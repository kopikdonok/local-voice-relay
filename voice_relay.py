import io
import os
import torch
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel

# ==================== НАСТРОЙКИ ====================
VIRTUAL_CABLE_NAME = "CABLE Input"

# STT: размер модели Faster-Whisper ('tiny', 'base', 'small')
# 'base' дает оптимальный баланс точности и скорости на CPU
WHISPER_MODEL_SIZE = "base"

# TTS: голос Silero ('aidar', 'baya', 'kseniya', 'xenia', 'eugene')
SILERO_SPEAKER = "xenia"
SILERO_SAMPLE_RATE = 48000
SILERO_MODEL_FILE = "silero_v4_ru.pt"
# ===================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


def get_virtual_cable_index(name_query=VIRTUAL_CABLE_NAME):
    """Поиск индекса виртуального кабеля среди устройств вывода."""
    for idx, dev in enumerate(sd.query_devices()):
        if name_query.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return idx, dev["name"]
    return None, None


def load_silero_tts():
    """Загрузка локальной модели Silero TTS."""
    if not os.path.exists(SILERO_MODEL_FILE):
        print("Скачивание модели Silero TTS (однократно, ~100 МБ)...")
        torch.hub.download_url_to_file(
            "https://models.silero.ai/models/tts/ru/v4_ru.pt",
            SILERO_MODEL_FILE
        )
    
    model = torch.package.PackageImporter(SILERO_MODEL_FILE).load_pickle("tts_models", "model")
    model.to(torch.device(DEVICE))
    return model


def main():
    cable_id, cable_name = get_virtual_cable_index()
    if cable_id is None:
        print(f"Ошибка: Устройство '{VIRTUAL_CABLE_NAME}' не найдено.")
        print("Доступные устройства вывода:")
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                print(f"  [{idx}] {dev['name']}")
        return

    print(f"Выходное устройство: [{cable_id}] {cable_name}")
    print(f"Вычисления выполняются на: {DEVICE.upper()}")

    # 1. Инициализация STT (Faster-Whisper)
    print(f"Загрузка Faster-Whisper ({WHISPER_MODEL_SIZE})...")
    stt_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    # 2. Инициализация TTS (Silero)
    print("Загрузка Silero TTS...")
    tts_model = load_silero_tts()
    print("Все модели загружены в память.\n")

    # 3. Настройка микрофона
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.7  # Пауза тишины для отсечки фразы (сек)

    with sr.Microphone() as source:
        print("Калибровка фонового шума (помолчите 1 сек)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("\nГотово к работе. Говорите в микрофон (Ctrl+C для выхода)...\n")

    with sr.Microphone() as source:
        while True:
            try:
                # Захват аудио с микрофона
                audio_data = recognizer.listen(source, phrase_time_limit=15)
                
                # Конвертация в WAV для Whisper
                wav_stream = io.BytesIO(audio_data.get_wav_data())

                # Локальное распознавание речи
                segments, _ = stt_model.transcribe(
                    wav_stream, 
                    language="ru", 
                    beam_size=1,
                    vad_filter=True  # Отсекает шум внутри аудио
                )
                
                recognized_text = " ".join([segment.text for segment in segments]).strip()
                if not recognized_text:
                    continue

                print(f"[Распознано]: {recognized_text}")

                # Локальный синтез речи
                audio_tensor = tts_model.apply_tts(
                    text=recognized_text,
                    speaker=SILERO_SPEAKER,
                    sample_rate=SILERO_SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True
                )

                # Воспроизведение в Virtual Cable
                audio_numpy = audio_tensor.numpy()
                sd.play(audio_numpy, samplerate=SILERO_SAMPLE_RATE, device=cable_id)
                sd.wait()
                print("[Воспроизведено]\n")

            except KeyboardInterrupt:
                print("\nЗавершение работы.")
                break
            except Exception as e:
                print(f"Ошибка обработки: {e}")


if __name__ == "__main__":
    main()