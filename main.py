import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import queue
import numpy as np
import pyaudio
import soundfile as sf
from scipy.fft import fft, ifft, fftfreq
import tkinter.filedialog as filedialog
import os
from pydub import AudioSegment

# --- AudioProcessor Class ---
class AudioProcessor:
    def __init__(self, channels=2, audio_format=pyaudio.paFloat32, chunk_size=1024):
        self.p = pyaudio.PyAudio()
        self.channels = channels
        self.audio_format = audio_format
        self.chunk_size = chunk_size
        self.samplerate = 0

        self.stream = None
        self.audio_data = None
        self.current_frame = 0
        self.is_playing = False
        self.playback_thread = None

        self.total_frames = 0
        self.duration_seconds = 0.0

        self.equalizer_gains = {
            "31Hz": 0.0, "62Hz": 0.0, "125Hz": 0.0, "250Hz": 0.0, "500Hz": 0.0,
            "1kHz": 0.0, "2kHz": 0.0, "4kHz": 00.0, "8kHz": 0.0, "16kHz": 0.0
        }
        self.eq_lock = threading.Lock()

        self.master_volume = 1.0
        self.volume_lock = threading.Lock()

        self.band_frequencies = {
            "31Hz": 31, "62Hz": 62, "125Hz": 125, "250Hz": 250, "500Hz": 500,
            "1kHz": 1000, "2kHz": 2000, "4kHz": 4000, "8kHz": 8000, "16kHz": 16000
        }
        self.band_width_factor = 0.5

        self.ui_update_queue = queue.Queue()
        self.playback_position_lock = threading.Lock()
        self.temp_wav_file = None

        # --- New: Device Selection Attributes ---
        self.output_device_index = None # None means use default output device
        self._actual_stream_output_device_index = None # The device index the current stream was opened with


    def get_output_devices(self):
        """Lists available output audio devices."""
        info = self.p.get_host_api_info_by_index(0) # Default host API
        num_devices = info.get('deviceCount')
        devices = {} # Maps device name to its index
        for i in range(num_devices):
            device_info = self.p.get_device_info_by_host_api_device_index(info.get('index'), i)
            if device_info.get('maxOutputChannels') > 0:
                devices[device_info.get('name')] = device_info.get('index')
        return devices

    def set_output_device(self, index):
        """Sets the output device index."""
        if self.output_device_index != index:
            self.output_device_index = index
            print(f"Output device set to index: {index}")
            # If playing or paused, call play() to ensure stream is restarted with new device
            if self.is_playing or (self.stream and self.stream.is_stopped()):
                self.play()


    def load_audio_file(self, filepath):
        if self.temp_wav_file and os.path.exists(self.temp_wav_file):
            os.remove(self.temp_wav_file)
            self.temp_wav_file = None

        try:
            ext = os.path.splitext(filepath)[1].lower()
            source_filepath = filepath

            if ext in ['.m4a', '.mp3', '.wma', '.aac', '.flv']:
                print(f"Detected {ext} file, converting to WAV using pydub...")
                audio = AudioSegment.from_file(filepath)
                if audio.channels != self.channels:
                    if self.channels == 1 and audio.channels > 1:
                        audio = audio.set_channels(1)
                    elif self.channels == 2 and audio.channels == 1:
                        audio = audio.set_channels(2)

                temp_filename = "temp_audio_file.wav"
                self.temp_wav_file = os.path.join(os.path.dirname(filepath), temp_filename)
                audio.export(self.temp_wav_file, format="wav")
                source_filepath = self.temp_wav_file

            self.audio_data, self.samplerate = sf.read(source_filepath, dtype='float32')

            if self.audio_data.ndim == 1:
                self.audio_data = np.expand_dims(self.audio_data, axis=1)
            if self.audio_data.shape[1] > self.channels:
                self.audio_data = self.audio_data[:, :self.channels]
            elif self.audio_data.shape[1] < self.channels:
                pad_width = ((0, 0), (0, self.channels - self.audio_data.shape[1]))
                self.audio_data = np.pad(self.audio_data, pad_width, 'constant')

            with self.playback_position_lock:
                self.current_frame = 0
            self.total_frames = len(self.audio_data)
            self.duration_seconds = self.total_frames / self.samplerate if self.samplerate > 0 else 0

            print(f"Loaded {filepath} (via {source_filepath if source_filepath != filepath else 'direct'}) with samplerate {self.samplerate}, channels {self.audio_data.shape[1]}, duration {self.duration_seconds:.2f}s")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.audio_data = None
            self.total_frames = 0
            self.duration_seconds = 0.0
            return False

    def _audio_callback(self, _in_data, frame_count, _time_info, _status):
        if not self.is_playing or self.audio_data is None:
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), pyaudio.paContinue)

        with self.playback_position_lock:
            start_frame = self.current_frame
            end_frame = self.current_frame + frame_count
            remaining_frames = self.total_frames - start_frame

        if remaining_frames <= 0:
            self.is_playing = False
            self.ui_update_queue.put((self.total_frames, self.total_frames))
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), pyaudio.paComplete)

        if remaining_frames < frame_count:
            chunk = self.audio_data[start_frame:start_frame + remaining_frames]
            processed_chunk = self._process_chunk(chunk)
            output_chunk = np.zeros((frame_count, self.channels), dtype=np.float32)
            output_chunk[:remaining_frames] = processed_chunk
            with self.playback_position_lock:
                self.current_frame += remaining_frames
        else:
            chunk = self.audio_data[start_frame:end_frame]
            output_chunk = self._process_chunk(chunk)
            with self.playback_position_lock:
                self.current_frame += frame_count

        if self.ui_update_queue.empty() and self.current_frame % (self.chunk_size * 5) < self.chunk_size:
             self.ui_update_queue.put((self.current_frame, self.total_frames))

        return (output_chunk.tobytes(), pyaudio.paContinue)

    def _process_chunk(self, chunk):
        if chunk.shape[0] == 0:
            return chunk

        processed_chunk = np.copy(chunk)
        num_samples = chunk.shape[0]
        xf = fftfreq(num_samples, 1 / self.samplerate)

        with self.eq_lock:
            current_gains = self.equalizer_gains.copy()

        for c in range(self.channels):
            yf = fft(chunk[:, c])

            for band_name, center_freq in self.band_frequencies.items():
                gain_db = current_gains.get(band_name, 0.0)
                gain_linear = 10**(gain_db / 20)

                lower_bound = center_freq * (1 - self.band_width_factor)
                upper_bound = center_freq * (1 + self.band_width_factor)

                indices = np.where(
                    (xf >= lower_bound) & (xf < upper_bound) |
                    (xf <= -lower_bound) & (xf > -upper_bound)
                )

                yf[indices] *= gain_linear

            processed_chunk[:, c] = np.real(ifft(yf))

        with self.volume_lock:
            processed_chunk *= self.master_volume

        processed_chunk = np.clip(processed_chunk, -1.0, 1.0)
        return processed_chunk

    def set_equalizer_gain(self, band_name, value):
        with self.eq_lock:
            self.equalizer_gains[band_name] = value

    def set_master_volume(self, volume_linear):
        with self.volume_lock:
            self.master_volume = volume_linear

    def seek(self, position_seconds):
        if self.audio_data is None or self.samplerate == 0:
            return

        new_frame = int(position_seconds * self.samplerate)
        if new_frame < 0:
            new_frame = 0
        elif new_frame > self.total_frames:
            new_frame = self.total_frames

        with self.playback_position_lock:
            self.current_frame = new_frame
            print(f"Seeked to {position_seconds:.2f}s (frame {new_frame})")
            self.ui_update_queue.put((self.current_frame, self.total_frames))

        # No need to restart stream on seek if callback handles current_frame correctly
        # The PyAudio callback will automatically pick up the new current_frame.
        # if self.is_playing:
        #    pass # No explicit action needed for a running stream.

    def _close_stream(self):
        """Helper to safely close the PyAudio stream."""
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self._actual_stream_output_device_index = None # Reset the recorded device index

    def play(self):
        if self.audio_data is None:
            print("No audio file loaded.")
            return

        # Check if we need to open a new stream (no stream, or device mismatch)
        if self.stream is None or self._actual_stream_output_device_index != self.output_device_index:
            print(f"Attempting to open new stream for device {self.output_device_index}...")
            
            # Capture current playback position if the stream was playing/paused
            # This allows seamless device switching without resetting position to 0
            current_pos_frame_before_close = self.current_frame if self.stream else 0

            self._close_stream() # Close any existing stream cleanly

            try:
                self.stream = self.p.open(format=self.audio_format,
                                          channels=self.channels,
                                          rate=self.samplerate,
                                          output=True,
                                          frames_per_buffer=self.chunk_size,
                                          output_device_index=self.output_device_index, # Use selected device
                                          stream_callback=self._audio_callback)
                self._actual_stream_output_device_index = self.output_device_index # Record the device index used
                
                with self.playback_position_lock:
                    self.current_frame = current_pos_frame_before_close # Restore position
                
                self.is_playing = True
                self.stream.start_stream()
                print(f"Playback started/restarted on device (index: {self.output_device_index}).")
            except Exception as e:
                print(f"Error opening audio stream on device {self.output_device_index}: {e}")
                self.is_playing = False
                self._close_stream() # Ensure stream is closed if error
                # Optionally, try to revert to default device or disable playback if error
                return

        elif self.stream.is_stopped(): # Stream exists and is on the right device, but was paused
            self.is_playing = True
            self.stream.start_stream()
            print("Playback resumed on existing stream.")
        # If it's already playing on the correct device, do nothing.

    def pause(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.is_playing = False
            print("Playback paused.")

    def stop(self):
        if self.stream:
            self._close_stream() # Use the helper to close the stream
            self.is_playing = False
            with self.playback_position_lock:
                self.current_frame = 0
            self.ui_update_queue.put((0, self.total_frames))
            print("Playback stopped and reset.")

    def close(self):
        self._close_stream() # Use the helper to close the stream
        self.p.terminate()
        if hasattr(self, 'temp_wav_file') and self.temp_wav_file and os.path.exists(self.temp_wav_file):
            os.remove(self.temp_wav_file)
        print("Audio processor closed.")

# --- EqualizerApp Class ---
class EqualizerApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("ttkbootstrap Equalizer")
        self.geometry("780x430") # Slightly increased height for device selection

        self.audio_processor = AudioProcessor(channels=2, chunk_size=1024)
        self.current_audio_file = None

        self.load_button = None
        self.file_label = None
        self.frequencies = list(self.audio_processor.band_frequencies.keys())
        self.scales = {}
        self.play_button = None
        self.pause_button = None
        self.stop_button = None
        self.volume_slider = None
        self.current_time_label = None
        self.total_time_label = None
        self.scrobbler_slider = None

        self.eq_canvas = None
        self.eq_curve_id = None
        self.canvas_window_ids = {}

        # New: Device selection attributes
        self.output_devices_map = {} # Maps device name to index
        self.selected_output_device_var = tk.StringVar(self)
        self.output_device_combobox = None

        self.genre_presets = {
            "Flat": {
                "31Hz": 0.0, "62Hz": 0.0, "125Hz": 0.0, "250Hz": 0.0, "500Hz": 0.0,
                "1kHz": 0.0, "2kHz": 0.0, "4kHz": 0.0, "8kHz": 0.0, "16kHz": 0.0
            },
            "Rock": {
                "31Hz": 5.0, "62Hz": 3.0, "125Hz": -2.0, "250Hz": -3.0, "500Hz": 0.0,
                "1kHz": 2.0, "2kHz": 4.0, "4kHz": 5.0, "8kHz": 4.0, "16kHz": 3.0
            },
            "Pop": {
                "31Hz": 3.0, "62Hz": 2.0, "125Hz": 0.0, "250Hz": 1.0, "500Hz": 2.0,
                "1kHz": 2.0, "2kHz": 1.0, "4kHz": 0.0, "8kHz": 3.0, "16kHz": 4.0
            },
            "Jazz": {
                "31Hz": 2.0, "62Hz": 1.0, "125Hz": 0.0, "250Hz": -1.0, "500Hz": 0.0,
                "1kHz": 1.0, "2kHz": 2.0, "4kHz": 3.0, "8kHz": 2.0, "16kHz": 1.0
            },
            "Classical": {
                "31Hz": 1.0, "62Hz": 0.0, "125Hz": -1.0, "250Hz": -2.0, "500Hz": -1.0,
                "1kHz": 0.0, "2kHz": 1.0, "4kHz": 2.0, "8kHz": 2.0, "16kHz": 1.0
            },
            "Dance": {
                "31Hz": 6.0, "62Hz": 4.0, "125Hz": 0.0, "250Hz": -2.0, "500Hz": 0.0,
                "1kHz": 3.0, "2kHz": 5.0, "4kHz": 6.0, "8kHz": 5.0, "16kHz": 4.0
            },
            "Speech": {
                "31Hz": -5.0, "62Hz": -5.0, "125Hz": 0.0, "250Hz": 3.0, "500Hz": 5.0,
                "1kHz": 4.0, "2kHz": 2.0, "4kHz": 0.0, "8kHz": -3.0, "16kHz": -5.0
            },
            "Bass Boost": {
                "31Hz": 8.0, "62Hz": 6.0, "125Hz": 3.0, "250Hz": 0.0, "500Hz": -2.0,
                "1kHz": -3.0, "2kHz": -4.0, "4kHz": -4.0, "8kHz": -3.0, "16kHz": -2.0
            },
            "Vocal Boost": {
                "31Hz": -3.0, "62Hz": -2.0, "125Hz": 0.0, "250Hz": 2.0, "500Hz": 4.0,
                "1kHz": 5.0, "2kHz": 3.0, "4kHz": 1.0, "8kHz": 0.0, "16kHz": -1.0
            }
        }
        self.selected_preset_var = tk.StringVar(self)
        self.preset_combobox = None
        self.preset_label = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.after(100, self.process_audio_updates)

        # Initialize presets and devices
        self.selected_preset_var.set("Flat")
        self.apply_preset()
        self.populate_output_devices()


    def create_widgets(self):
        # Top Control Frame (Load, File Label, Preset)
        control_frame = ttk.Frame(self, padding=(10, 5))
        control_frame.pack(side=TOP, fill=X)

        self.load_button = ttk.Button(control_frame, text="Load Audio", command=self.load_audio_dialog, bootstyle=PRIMARY)
        self.load_button.pack(side=LEFT, padx=5)

        self.file_label = ttk.Label(control_frame, text="No file loaded", bootstyle=INFO, font=("-size", 9))
        self.file_label.pack(side=LEFT, padx=10, expand=True, fill=X)

        self.preset_label = ttk.Label(control_frame, text="Preset:", font=("-size", 9))
        self.preset_label.pack(side=LEFT, padx=(10, 5))

        self.preset_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.selected_preset_var,
            values=list(self.genre_presets.keys()),
            state="readonly",
            bootstyle=PRIMARY,
            width=12
        )
        self.preset_combobox.set("Flat")
        self.preset_combobox.pack(side=LEFT, padx=5)
        self.preset_combobox.bind("<<ComboboxSelected>>", self.apply_preset)

        # --- New: Device Selection Frame ---
        device_selection_frame = ttk.Frame(self, padding=(10, 5))
        device_selection_frame.pack(side=TOP, fill=X)

        ttk.Label(device_selection_frame, text="Output Device:", font=("-size", 9)).pack(side=LEFT, padx=(5, 5))
        self.output_device_combobox = ttk.Combobox(
            device_selection_frame,
            textvariable=self.selected_output_device_var,
            state="readonly",
            bootstyle=INFO,
            width=30 # Adjust width as needed for device names
        )
        self.output_device_combobox.pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        self.output_device_combobox.bind("<<ComboboxSelected>>", self.on_output_device_selected)


        # --- Equalizer Canvas Frame ---
        self.eq_canvas = ttk.Canvas(self, background=self.style.colors.bg, highlightthickness=0)
        self.eq_canvas.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)
        self.eq_canvas.bind("<Configure>", self.on_canvas_resize)

        canvas_bg_color = self.style.colors.bg

        y_center_pos_initial_guess = 100
        
        for i, freq_band in enumerate(self.frequencies):
            band_frame = tk.Frame(self.eq_canvas, bg=canvas_bg_color, relief=FLAT)

            ttk.Label(band_frame, text="+10dB", bootstyle=SECONDARY, font=("-size", 7), background=canvas_bg_color).pack(pady=(1,0))

            scale = ttk.Scale(
                band_frame,
                from_=10,
                to=-10,
                value=0,
                orient=VERTICAL,
                length=120,
                command=lambda val, f=freq_band: (self.update_equalizer(f, val), self.draw_eq_curve())
            )
            scale.pack(fill=Y, expand=True, pady=(1,1))
            self.scales[freq_band] = scale

            ttk.Label(band_frame, text="-10dB", bootstyle=SECONDARY, font=("-size", 7), background=canvas_bg_color).pack(pady=(0,1))
            ttk.Label(band_frame, text=freq_band, bootstyle=INFO, font=("-size", 8), background=canvas_bg_color).pack(pady=(2,0))

            window_id = self.eq_canvas.create_window(
                (i * 50) + 25, y_center_pos_initial_guess,
                window=band_frame,
                anchor=CENTER
            )
            self.canvas_window_ids[freq_band] = window_id

        bottom_controls_frame = ttk.Frame(self, padding=(10, 5))
        bottom_controls_frame.pack(side=BOTTOM, fill=X)

        playback_buttons_frame = ttk.Frame(bottom_controls_frame)
        playback_buttons_frame.pack(side=BOTTOM, fill=X, pady=(5, 0))

        self.play_button = ttk.Button(playback_buttons_frame, text="Play", command=self.play_audio, bootstyle=SUCCESS)
        self.play_button.pack(side=LEFT, padx=5)

        self.pause_button = ttk.Button(playback_buttons_frame, text="Pause", command=self.pause_audio, bootstyle=WARNING)
        self.pause_button.pack(side=LEFT, padx=5)

        self.stop_button = ttk.Button(playback_buttons_frame, text="Stop", command=self.stop_audio, bootstyle=DANGER)
        self.stop_button.pack(side=LEFT, padx=5)

        player_info_frame = ttk.Frame(bottom_controls_frame)
        player_info_frame.pack(side=TOP, fill=X, pady=(0, 5))

        volume_frame = ttk.Frame(player_info_frame)
        volume_frame.pack(side=RIGHT, padx=5)
        ttk.Label(volume_frame, text="Volume", font=("-size", 8)).pack(side=TOP, pady=0)
        self.volume_slider = ttk.Scale(
            volume_frame,
            from_=0.0,
            to=1.0,
            value=self.audio_processor.master_volume,
            orient=HORIZONTAL,
            length=100,
            command=self.set_volume,
            bootstyle=INFO
        )
        self.volume_slider.pack(side=BOTTOM)

        scrobbler_frame = ttk.Frame(player_info_frame)
        scrobbler_frame.pack(side=LEFT, expand=True, fill=X, padx=(5, 5))

        time_labels_frame = ttk.Frame(scrobbler_frame)
        time_labels_frame.pack(side=TOP, fill=X)
        self.current_time_label = ttk.Label(time_labels_frame, text="00:00", bootstyle=SECONDARY, font=("-size", 7))
        self.current_time_label.pack(side=LEFT)
        self.total_time_label = ttk.Label(time_labels_frame, text="00:00", bootstyle=SECONDARY, font=("-size", 7))
        self.total_time_label.pack(side=RIGHT)

        self.scrobbler_slider = ttk.Scale(
            scrobbler_frame,
            from_=0,
            to=0,
            value=0,
            orient=HORIZONTAL,
            command=self.seek_playback,
            bootstyle=PRIMARY
        )
        self.scrobbler_slider.pack(side=TOP, fill=X, expand=True, pady=1)

        self.after(10, self.draw_eq_curve)


    def populate_output_devices(self):
        """Populates the output device combobox."""
        self.output_devices_map = self.audio_processor.get_output_devices()
        device_names = list(self.output_devices_map.keys())

        if not device_names:
            self.output_device_combobox.set("No Output Devices Found")
            self.output_device_combobox.config(state="disabled")
            return

        self.output_device_combobox.config(values=device_names)

        # Try to select the default device, or the first one if no default
        try:
            default_device_info = self.audio_processor.p.get_default_output_device_info()
            default_device_name = default_device_info.get('name')
            if default_device_name in device_names:
                self.selected_output_device_var.set(default_device_name)
                self.audio_processor.set_output_device(self.output_devices_map[default_device_name])
            else:
                # If default isn't found in list of maxOutputChannels > 0, pick first
                self.selected_output_device_var.set(device_names[0])
                self.audio_processor.set_output_device(self.output_devices_map[device_names[0]])
        except Exception as e:
            print(f"Could not get default output device: {e}. Selecting first available.")
            # Fallback to first device if getting default fails (e.g., no default set)
            self.selected_output_device_var.set(device_names[0])
            self.audio_processor.set_output_device(self.output_devices_map[device_names[0]])


    def on_output_device_selected(self, event):
        """Callback for when an output device is selected."""
        selected_name = self.selected_output_device_var.get()
        selected_index = self.output_devices_map.get(selected_name)
        if selected_index is not None:
            self.audio_processor.set_output_device(selected_index)
        else:
            print(f"Error: Could not find index for selected device '{selected_name}'")


    def on_canvas_resize(self, event=None):
        self.reposition_slider_frames()
        self.draw_eq_curve()

    def reposition_slider_frames(self):
        canvas_width = self.eq_canvas.winfo_width()
        canvas_height = self.eq_canvas.winfo_height()
        if canvas_width == 0 or canvas_height == 0:
            return

        num_bands = len(self.frequencies)
        if num_bands == 0:
            return

        padding_x = 5
        slider_frame_height_approx = 142
        
        y_center_pos = canvas_height / 2

        x_margin = 25
        
        if num_bands > 1:
            x_centers = np.linspace(x_margin, canvas_width - x_margin, num_bands)
        else:
            x_centers = [canvas_width / 2]
        
        for i, freq_band in enumerate(self.frequencies):
            if freq_band in self.canvas_window_ids:
                window_id = self.canvas_window_ids[freq_band]
                self.eq_canvas.coords(window_id, x_centers[i], y_center_pos)

    def draw_eq_curve(self):
        self.eq_canvas.delete("eq_curve")
        self.eq_canvas.delete("eq_grid")

        canvas_width = self.eq_canvas.winfo_width()
        canvas_height = self.eq_canvas.winfo_height()

        if canvas_width == 0 or canvas_height == 0:
            return

        points = []
        num_bands = len(self.frequencies)
        if num_bands == 0:
            return

        min_gain_db = -10.0
        max_gain_db = 10.0
        total_gain_range = max_gain_db - min_gain_db

        x_coords = []
        for freq_band in self.frequencies:
            if freq_band in self.canvas_window_ids:
                window_id = self.canvas_window_ids[freq_band]
                x_pos, _ = self.eq_canvas.coords(window_id)
                x_coords.append(x_pos)
            else:
                x_coords.append(0)

        if all(x == 0 for x in x_coords) and num_bands > 0:
             x_margin = 25
             if num_bands > 1:
                 x_coords = np.linspace(x_margin, canvas_width - x_margin, num_bands)
             else:
                 x_coords = [canvas_width / 2]


        margin_y_ratio = 0.1
        effective_canvas_height = canvas_height * (1 - 2 * margin_y_ratio)
        y_start = canvas_height * margin_y_ratio

        for i, freq_band in enumerate(self.frequencies):
            gain_db = self.scales[freq_band].get()
            normalized_gain = (gain_db - min_gain_db) / total_gain_range
            y_coord = y_start + (1 - normalized_gain) * effective_canvas_height
            points.append(x_coords[i])
            points.append(y_coord)
        
        plus_10db_norm = (10.0 - min_gain_db) / total_gain_range
        y_plus10db = y_start + (1 - plus_10db_norm) * effective_canvas_height
        self.eq_canvas.create_line(0, y_plus10db, canvas_width, y_plus10db, fill="gray", dash=(2, 2), tags="eq_grid")

        zero_db_norm = (0.0 - min_gain_db) / total_gain_range
        y_0db = y_start + (1 - zero_db_norm) * effective_canvas_height
        self.eq_canvas.create_line(0, y_0db, canvas_width, y_0db, fill="gray", dash=(2, 2), tags="eq_grid")
        
        minus_10db_norm = (-10.0 - min_gain_db) / total_gain_range
        y_minus10db = y_start + (1 - minus_10db_norm) * effective_canvas_height
        self.eq_canvas.create_line(0, y_minus10db, canvas_width, y_minus10db, fill="gray", dash=(2, 2), tags="eq_grid")

        if len(points) >= 4:
            self.eq_curve_id = self.eq_canvas.create_line(points, smooth=True, fill=self.style.colors.primary, width=3, tags="eq_curve")
        
        self.eq_canvas.tag_lower("eq_grid", "eq_curve")
        for _id in self.canvas_window_ids.values():
            self.eq_canvas.tag_raise(_id)


    def load_audio_dialog(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.wma"),
                ("All Files", "*.*")
            ]
        )
        if filepath:
            self.stop_audio()
            if self.audio_processor.load_audio_file(filepath):
                self.current_audio_file = filepath
                self.file_label.config(text=f"Loaded: {os.path.basename(filepath)}")

                self.scrobbler_slider.config(to=self.audio_processor.duration_seconds)
                self.scrobbler_slider.set(0)
                self.current_time_label.config(text="00:00")
                self.total_time_label.config(text=self._format_time(self.audio_processor.duration_seconds))
            else:
                self.file_label.config(text=f"Failed to load: {os.path.basename(filepath)}")
                self.scrobbler_slider.config(to=0)
                self.scrobbler_slider.set(0)
                self.current_time_label.config(text="00:00")
                self.total_time_label.config(text="00:00")


    def update_equalizer(self, freq_band, value):
        gain_db = float(value)
        self.audio_processor.set_equalizer_gain(freq_band, gain_db)

    def set_volume(self, value):
        volume_linear = float(value)
        self.audio_processor.set_master_volume(volume_linear)

    def seek_playback(self, value):
        position_seconds = float(value)
        self.audio_processor.seek(position_seconds)
        self.current_time_label.config(text=self._format_time(position_seconds))

    def apply_preset(self, event=None):
        selected_genre = self.selected_preset_var.get()
        if selected_genre in self.genre_presets:
            settings = self.genre_presets[selected_genre]
            for freq_band, gain_db in settings.items():
                if freq_band in self.scales:
                    self.scales[freq_band].set(gain_db)
                    self.audio_processor.set_equalizer_gain(freq_band, gain_db)
            print(f"Applied '{selected_genre}' preset.")
            self.draw_eq_curve()
        else:
            print(f"Preset '{selected_genre}' not found.")

    def play_audio(self):
        if self.current_audio_file:
            self.audio_processor.play()
        else:
            print("Please load an audio file first.")

    def pause_audio(self):
        self.audio_processor.pause()

    def stop_audio(self):
        self.audio_processor.stop()

    def process_audio_updates(self):
        """Polls the queue for audio progress updates and updates the UI."""
        try:
            current_frame, total_frames = self.audio_processor.ui_update_queue.get_nowait()
            if total_frames > 0:
                current_seconds = current_frame / self.audio_processor.samplerate
                if not self.scrobbler_slider.instate(['!pressed']):
                     self.scrobbler_slider.set(current_seconds)
                self.current_time_label.config(text=self._format_time(current_seconds))
            else:
                self.current_time_label.config(text="00:00")
                self.scrobbler_slider.set(0)

        except queue.Empty:
            pass

        self.after(100, self.process_audio_updates)

    @staticmethod
    def _format_time(seconds):
        """Helper to format seconds into MM:SS string."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def on_closing(self):
        self.audio_processor.close()
        self.destroy()

if __name__ == "__main__":
    app = EqualizerApp()
    app.mainloop()