import io
import os
import queue
import threading
import time
import customtkinter as ctk
import keyboard
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import torch
from faster_whisper import WhisperModel

# Настройки внешнего вида CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
SILERO_MODEL_FILE = "silero_v4_ru.pt"


class VoiceRelayApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Local Voice Relay")
        self.geometry("660x780")
        self.minsize(600, 720)

        # Модели и состояние потока
        self.stt_model = None
        self.tts_model = None
        self.current_whisper_size = None
        self.is_running = threading.Event()
        self.worker_thread = None
        self.log_queue = queue.Queue()

        self._build_ui()
        self._refresh_audio_devices()
        self.after(100, self._process_log_queue)

    def _build_ui(self):
        # 1. Устройства звука
        dev_frame = ctk.CTkFrame(self)
        dev_frame.pack(fill="x", padx=15, pady=(15, 6))

        ctk.CTkLabel(dev_frame, text="Звуковые устройства", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(6, 2))

        ctk.CTkLabel(dev_frame, text="Вывод (Virtual Cable):").pack(anchor="w", padx=10)
        self.out_device_var = ctk.StringVar()
        self.out_device_menu = ctk.CTkOptionMenu(dev_frame, variable=self.out_device_var)
        self.out_device_menu.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(dev_frame, text="Микрофон (Вход):").pack(anchor="w", padx=10)
        self.in_device_var = ctk.StringVar()
        self.in_device_menu = ctk.CTkOptionMenu(dev_frame, variable=self.in_device_var)
        self.in_device_menu.pack(fill="x", padx=10, pady=(0, 8))

        # 2. Режим активации (VOX vs Push-to-Talk)
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill="x", padx=15, pady=6)

        ctk.CTkLabel(mode_frame, text="Режим активации записи", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(6, 2))

        self.activation_mode_var = ctk.StringVar(value="Голос (VOX)")
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Голос (VOX)", "Push-to-Talk"],
            variable=self.activation_mode_var,
            command=self._on_mode_change
        )
        self.mode_selector.pack(fill="x", padx=10, pady=4)

        # Контейнер для настроек VOX (ползунок тишины)
        self.vox_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        self.vox_frame.pack(fill="x", padx=10, pady=(4, 8))
        self.pause_label = ctk.CTkLabel(self.vox_frame, text="Пауза тишины: 0.70 сек")
        self.pause_label.pack(anchor="w")
        self.pause_slider = ctk.CTkSlider(self.vox_frame, from_=0.3, to=2.0, number_of_steps=17, command=self._on_slider_change)
        self.pause_slider.set(0.7)
        self.pause_slider.pack(fill="x", pady=(2, 4))

        # Контейнер для настроек Push-to-Talk (кнопка назначения)
        self.ptt_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        self.ptt_key_var = ctk.StringVar(value="caps lock")
        self.ptt_bind_btn = ctk.CTkButton(
            self.ptt_frame,
            text=f"Клавиша PTT: [{self.ptt_key_var.get().upper()}] (Нажмите для смены)",
            fg_color="#3a3a3a",
            hover_color="#4a4a4a",
            command=self._start_key_binding
        )
        self.ptt_bind_btn.pack(fill="x", pady=(4, 8))

        # 3. Нейросети (Whisper + Silero)
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=15, pady=6)

        ctk.CTkLabel(model_frame, text="Параметры нейросетей", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(6, 2))

        ctk.CTkLabel(model_frame, text="Модель Faster-Whisper:").pack(anchor="w", padx=10)
        self.whisper_size_var = ctk.StringVar(value="base")
        whisper_menu = ctk.CTkSegmentedButton(model_frame, values=["tiny", "base", "small", "medium"], variable=self.whisper_size_var)
        whisper_menu.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(model_frame, text="Голос Silero TTS:").pack(anchor="w", padx=10)
        self.speaker_var = ctk.StringVar(value="xenia")
        speaker_menu = ctk.CTkSegmentedButton(model_frame, values=["xenia", "kseniya", "baya", "aidar", "eugene"], variable=self.speaker_var)
        speaker_menu.pack(fill="x", padx=10, pady=(0, 8))

        # 4. Главная кнопка запуска / остановки
        self.toggle_btn = ctk.CTkButton(
            self,
            text="Запустить трансляцию",
            height=42,
            fg_color="#1f6aa5",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._toggle_run
        )
        self.toggle_btn.pack(fill="x", padx=15, pady=10)

        # 5. Окно логов
        self.log_box = ctk.CTkTextbox(self, state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Первичная синхронизация отображения фреймов режимов
        self._on_mode_change("Голос (VOX)")

    def _on_mode_change(self, mode):
        if mode == "Голос (VOX)":
            self.ptt_frame.pack_forget()
            self.vox_frame.pack(fill="x", padx=10, pady=(4, 8))
        else:
            self.vox_frame.pack_forget()
            self.ptt_frame.pack(fill="x", padx=10, pady=(4, 8))

    def _start_key_binding(self):
        def _listen_for_key():
            self.ptt_bind_btn.configure(text="Нажмите любую клавишу...", state="disabled")
            event = keyboard.read_event(suppress=False)
            while event.event_type != keyboard.KEY_DOWN:
                event = keyboard.read_event(suppress=False)

            key_pressed = event.name.lower()
            self.ptt_key_var.set(key_pressed)
            self.ptt_bind_btn.configure(
                text=f"Клавиша PTT: [{key_pressed.upper()}] (Нажмите для смены)",
                state="normal"
            )
            self.log(f"Назначена новая клавиша PTT: {key_pressed.upper()}")

        threading.Thread(target=_listen_for_key, daemon=True).start()

    def _on_slider_change(self, value):
        self.pause_label.configure(text=f"Пауза тишины: {value:.2f} сек")

    def _refresh_audio_devices(self):
        devices = sd.query_devices()
        input_names = []
        output_names = []
        default_out = None

        for idx, dev in enumerate(devices):
            d_name = f"[{idx}] {dev['name']}"
            if dev["max_input_channels"] > 0:
                input_names.append(d_name)
            if dev["max_output_channels"] > 0:
                output_names.append(d_name)
                if "cable input" in dev["name"].lower() and default_out is None:
                    default_out = d_name

        self.in_device_menu.configure(values=input_names or ["Нет устройств"])
        if input_names:
            self.in_device_var.set(input_names[0])

        self.out_device_menu.configure(values=output_names or ["Нет устройств"])
        if default_out:
            self.out_device_var.set(default_out)
        elif output_names:
            self.out_device_var.set(output_names[0])

    def log(self, text: str):
        self.log_queue.put(text)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(100, self._process_log_queue)

    def _parse_device_index(self, str_val: str):
        try:
            return int(str_val.split("]")[0].replace("[", ""))
        except Exception:
            return None

    def _load_silero_tts(self):
        if not os.path.exists(SILERO_MODEL_FILE):
            self.log("Скачивание Silero TTS (~100 МБ)...")
            torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v4_ru.pt", SILERO_MODEL_FILE)
        model = torch.package.PackageImporter(SILERO_MODEL_FILE).load_pickle("tts_models", "model")
        model.to(torch.device(DEVICE))
        return model

    def _toggle_run(self):
        if not self.is_running.is_set():
            self.is_running.set()
            self.toggle_btn.configure(text="Остановить трансляцию", fg_color="#c93b2b")
            self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.worker_thread.start()
        else:
            self.is_running.clear()
            self.toggle_btn.configure(text="Запустить трансляцию", fg_color="#1f6aa5")
            self.log("Остановка трансляции...")

    def _synthesize_and_play(self, text: str, out_idx: int):
        self.log(f">> [{self.speaker_var.get()}]: {text}")
        audio_tensor = self.tts_model.apply_tts(
            text=text,
            speaker=self.speaker_var.get(),
            sample_rate=48000,
            put_accent=True,
            put_yo=True
        )
        sd.play(audio_tensor.numpy(), samplerate=48000, device=out_idx)
        sd.wait()

    def _run_loop(self):
        out_idx = self._parse_device_index(self.out_device_var.get())
        in_idx = self._parse_device_index(self.in_device_var.get())
        whisper_size = self.whisper_size_var.get()

        if out_idx is None:
            self.log("Ошибка: не выбрано устройство вывода.")
            self.is_running.clear()
            self.toggle_btn.configure(text="Запустить трансляцию", fg_color="#1f6aa5")
            return

        try:
            if self.stt_model is None or self.current_whisper_size != whisper_size:
                self.log(f"Загрузка Faster-Whisper ({whisper_size}) на {DEVICE.upper()}...")
                self.stt_model = WhisperModel(whisper_size, device=DEVICE, compute_type=COMPUTE_TYPE)
                self.current_whisper_size = whisper_size

            if self.tts_model is None:
                self.log(f"Загрузка Silero TTS на {DEVICE.upper()}...")
                self.tts_model = self._load_silero_tts()

            mode = self.activation_mode_var.get()

            if mode == "Push-to-Talk":
                self._run_ptt_mode(in_idx, out_idx)
            else:
                self._run_vox_mode(in_idx, out_idx)

        except Exception as e:
            self.log(f"Сбой: {e}")
        finally:
            self.is_running.clear()
            self.toggle_btn.configure(text="Запустить трансляцию", fg_color="#1f6aa5")

    def _run_vox_mode(self, in_idx, out_idx):
        pause_val = float(self.pause_slider.get())
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = pause_val

        mic_args = {"device_index": in_idx} if in_idx is not None else {}
        with sr.Microphone(**mic_args) as source:
            self.log("Калибровка фонового шума (1 сек)...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            self.log("Режим VOX активен. Говорите в микрофон...")

            while self.is_running.is_set():
                try:
                    audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    continue

                wav_data = io.BytesIO(audio.get_wav_data())
                segments, _ = self.stt_model.transcribe(wav_data, language="ru", beam_size=1, vad_filter=True)
                text = " ".join([s.text for s in segments]).strip()

                if text:
                    self._synthesize_and_play(text, out_idx)

    def _run_ptt_mode(self, in_idx, out_idx):
        hotkey = self.ptt_key_var.get().lower()
        sample_rate = 16000  # Нативная частота для Whisper
        self.log(f"Режим Push-to-Talk активен. Удерживайте [{hotkey.upper()}] для записи...")

        while self.is_running.is_set():
            if keyboard.is_pressed(hotkey):
                self.log("Запись речи (клавиша удерживается)...")
                frames = []

                def callback(indata, frame_count, time_info, status):
                    frames.append(indata.copy())

                with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=in_idx, callback=callback):
                    while keyboard.is_pressed(hotkey) and self.is_running.is_set():
                        time.sleep(0.02)

                self.log("Клавиша отпущена. Обработка...")

                if frames:
                    audio_array = np.concatenate(frames, axis=0).flatten()
                    duration = len(audio_array) / sample_rate

                    # Игнорируем случайные короткие клики меньше 0.3 сек
                    if duration >= 0.3:
                        segments, _ = self.stt_model.transcribe(audio_array, language="ru", beam_size=1, vad_filter=True)
                        text = " ".join([s.text for s in segments]).strip()
                        if text:
                            self._synthesize_and_play(text, out_idx)
            else:
                time.sleep(0.05)


if __name__ == "__main__":
    app = VoiceRelayApp()
    app.mainloop()