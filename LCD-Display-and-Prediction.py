import tkinter as tk
import tkinter.font as tkfont
import threading
from PIL import Image, ImageTk, ImageDraw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
import sounddevice as sd
import wavio
import numpy as np
import os
import joblib
import librosa
from matplotlib.figure import Figure
from datetime import datetime
from scipy.signal import resample_poly
import wave

# ----------------------------
# UI CONFIG SECTION (EDIT HERE)
# ----------------------------

# TEMP PATH FOR TESTING: Set a file path here to skip recording and test prediction on saved audio
# Example: TEMP_AUDIO_PATH_FOR_TESTING = "female_fischer.wav"
# Leave empty string "" to use normal recording mode
TEMP_AUDIO_PATH_FOR_TESTING = r""

# SIMULATE RECORDING UI: When True, shows recording countdown animation while loading saved audio
# When False, skips recording UI entirely and goes straight to processing
# Only takes effect if TEMP_AUDIO_PATH_FOR_TESTING is set
SIMULATE_RECORDING_FROM_SAVED_AUDIO = True

UI_CONFIG = {
    "window": {
        "title": "LovebirdID Display",
        "size": "800x480",
        "bg_image": r"J:/Documents/Thesis/Code/image/ui_bg.png",
        "bg": "#f0f0f0"
    },
    "padding": 20,
    "species_box": {
        "x": 20,
        "y": 20,
        "width": 240,
        "height": 210,
        "bg": "#ffffff",
        "border": "#2975A7",
        "border_width": 2,
        "corner_radius": 20,
        "title_font": ("Inter", 16, "bold"),
        "text_font": ("Inter", 14),
        "text_color": "#333333",
        "radio_font": ("Inter", 12, "bold"),
        "radio_active_bg": "#2975A7",
        "radio_fg": "#ffffff"
    },
    "audio_box": {
        "x": 270,
        "y": 20,
        "width": 510,
        "height": 210,
        "bg": "#ffffff",
        "border": "#2975A7",
        "border_width": 2,
        "corner_radius": 20,
        "title_font": ("Inter", 16),
        "text_font": ("Inter", 14),
        "text_color": "#333333"
    },
    "restart_btn": {
        "x": 20,
        "y": 250,
        "width": 240,
        "height": 95,
        "bg": "#2975A7",
        "fg": "#ffffff",
        "disabled_bg": "#5793BB",
        "font": ("Inter", 20, "bold"),
        "corner_radius": 15
    },
    "start_btn": {
        "x": 20,
        "y": 365,
        "width": 240,
        "height": 95,
        "bg": "#2975A7",
        "fg": "#ffffff",
        "disabled_bg": "#5793BB",
        "font": ("Inter", 20, "bold"),
        "corner_radius": 15
    },
    "report_box": {
        "x": 280,
        "y": 250,
        "width": 240,
        "height": 210,
        "bg": "#ffffff",
        "border": "#2975A7",
        "border_width": 2,
        "corner_radius": 20,
        "title_font": ("Inter", 15, "bold"),
        "text_font": ("Inter", 12, "bold", ),
        "text_color": "#015C98"
    },
    "gender_box": {
        "x": 540,
        "y": 250,
        "width": 240,
        "height": 210,
        "bg": "#ffffff",
        "border": "#2975A7",
        "border_width": 2,
        "corner_radius": 20,
        "title_font": ("Inter", 20, "bold"),
        "symbol_font": ("DejaVu Sans", 60, "bold"),
        "male_symbol_color": "#00ccff",
        "female_symbol_color": "#ff66cc",
        "text_font": ("Inter", 15, "bold"),
        "text_color": "#015C98"
    },
    "countdown": {
        "font": ("Inter", 20),
        "color": "#000000"
    },
    "process_text": {
        "font": ("Inter", 18, "italic"),
        "color": "#000000"
    }
}

CONFIDENCE_TEMPERATURE = 80.0
DEFAULT_MALE_SCORE_BIAS = 0.00
DEFAULT_FEMALE_SCORE_BIAS = 0.00

# Optional startup calibration using training_audio/{fischer,masked}/{male,female}.
AUTO_CALIBRATE_BIAS_FROM_FOLDER = False
CALIBRATION_FOLDER = "training_audio"
# Kept for backward compatibility with older single-folder calibration flow.
CALIBRATION_ASSUMED_LABEL = "male"  # "male" or "female"
CALIBRATION_TARGET_HIT_RATE = 0.95

# Frozen species-specific biases from your latest calibration run.
# Used immediately at startup even when auto-calibration is disabled.
STATIC_BIAS_BY_SPECIES = {
    #"Fischers": {"male": -5.0299, "female": 5.0299},
    #"Masked": {"male": 2.3531, "female": -2.3531},
    
    "Fischers": {"male": 0.6, "female": -0.6},
    "Masked": {"male": -5.0, "female": 5.0},
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def draw_rounded_rectangle(canvas, x1, y1, x2, y2, r=20, **kwargs):
    """Draw a rounded rectangle on canvas and return its ID."""
    points = [
        x1+r, y1,
        x2-r, y1,
        x2, y1,
        x2, y1+r,
        x2, y2-r,
        x2, y2,
        x2-r, y2,
        x1+r, y2,
        x1, y2,
        x1, y2-r,
        x1, y1+r,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

def lighten_color(hex_color, factor=0.2):
    """Lighten a hex color."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    r = min(255, int(r * (1 + factor)))
    g = min(255, int(g * (1 + factor)))
    b = min(255, int(b * (1 + factor)))
    
    return f'#{r:02x}{g:02x}{b:02x}'

def limit_feature_frames(features, max_frames=2000):
    """
    Limit feature matrix to max_frames by uniform sampling.
    Prevents huge feature matrices from very long audio.
    """
    if features is None or max_frames is None or max_frames <= 0:
        return features
    if len(features) <= max_frames:
        return features
    sample_indices = np.linspace(0, len(features) - 1, num=max_frames, dtype=int)
    return features[sample_indices]


def load_audio_with_resampling(filepath, target_sr=22050):
    """
    Load WAV file using wave module and resample with scipy's resample_poly.
    More efficient than librosa for resampling.
    
    Returns: (audio_data, sample_rate) tuple
    """
    if not filepath.lower().endswith(".wav"):
        raise RuntimeError("Only .wav files are supported")
    
    try:
        with wave.open(filepath, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
        
        # Convert bytes to numpy array
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        if sampwidth not in dtype_map:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        raw_dtype = dtype_map[sampwidth]
        audio_data = np.frombuffer(audio_bytes, dtype=raw_dtype)
        
        # Convert stereo to mono if needed
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        
        # Convert to float32 in range [-1, 1]
        if sampwidth == 1:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        else:
            max_int = float(np.iinfo(raw_dtype).max)
            audio_data = audio_data.astype(np.float32) / max_int
        
        # Resample if needed using resample_poly (more efficient)
        if framerate != target_sr:
            from math import gcd
            g = gcd(framerate, target_sr)
            up = target_sr // g
            down = framerate // g
            audio_data = resample_poly(audio_data, up, down).astype(np.float32)
            print(f"[OK] Loaded and resampled: {len(audio_data)} samples at {target_sr} Hz")
        else:
            print(f"[OK] Loaded: {len(audio_data)} samples at {framerate} Hz")
        
        return audio_data, target_sr
    
    except Exception as e:
        print(f"[!] Error loading audio: {repr(e)}")
        raise

# ----------------------------
# FEATURE EXTRACTION FUNCTIONS
# ----------------------------

def apply_cepstral_mean_variance_normalization(features, apply_variance_norm=True, epsilon=1e-8):
    """
    Apply CMVN normalization to features.
    This makes predictions more robust to recording conditions.
    """
    mean = np.mean(features, axis=0)
    normalized_features = features - mean
    if apply_variance_norm:
        std_dev = np.std(normalized_features, axis=0)
        std_dev[std_dev < epsilon] = 1.0
        normalized_features = normalized_features / std_dev
    return normalized_features


def stack_neighboring_frames(features, left_context=2, right_context=2):
    """
    Stack neighboring frames to add temporal context.
    This helps the model understand the audio sequence better.
    """
    if features is None or len(features) == 0:
        return features

    num_frames, num_features = features.shape
    total_context = left_context + 1 + right_context
    padded_features = np.pad(features, ((left_context, right_context), (0, 0)), mode="edge")

    stacked_features = np.zeros((num_frames, num_features * total_context), dtype=features.dtype)
    for index in range(num_frames):
        stacked_features[index] = padded_features[index:index + total_context].reshape(-1)

    return stacked_features


def extract_features_for_prediction(audio_waveform, sample_rate, max_frames=2000):
    """
    Extract features matching HMM training.

    Base frame features:
    - 20 MFCC coefficients
    - 20 MFCC deltas
    - 20 MFCC delta-deltas
    - 3 additional acoustic features: F0, peak frequency, call duration

    Then apply context stacking (Ã‚Â±2 frames).
    Expected total dimension is typically 315 (63 Ãƒâ€” 5).
    """
    try:
        # Extract MFCC features (20 coefficients)
        mfcc_features = librosa.feature.mfcc(
            y=audio_waveform, 
            sr=sample_rate, 
            n_mfcc=20,  # Must match training
            n_fft=1024, 
            hop_length=256
        )
        
        # Calculate delta (velocity) and delta-delta (acceleration)
        mfcc_delta = librosa.feature.delta(mfcc_features)
        mfcc_delta_delta = librosa.feature.delta(mfcc_features, order=2)

        # Additional acoustic features to match training pipeline.
        f0 = librosa.yin(
            audio_waveform,
            fmin=50,
            fmax=5000,
            sr=sample_rate,
            frame_length=1024,
            hop_length=256,
        )

        stft_magnitude = np.abs(librosa.stft(audio_waveform, n_fft=1024, hop_length=256))
        peak_freq_bin = np.argmax(stft_magnitude, axis=0)
        peak_freq = peak_freq_bin * sample_rate / 1024.0

        call_duration_seconds = len(audio_waveform) / max(sample_rate, 1)
        call_duration_vector = np.full_like(f0, fill_value=call_duration_seconds, dtype=np.float64)

        min_len = min(
            mfcc_features.shape[1],
            mfcc_delta.shape[1],
            mfcc_delta_delta.shape[1],
            len(f0),
            len(peak_freq),
            len(call_duration_vector),
        )
        mfcc_features = mfcc_features[:, :min_len]
        mfcc_delta = mfcc_delta[:, :min_len]
        mfcc_delta_delta = mfcc_delta_delta[:, :min_len]
        f0 = f0[:min_len]
        peak_freq = peak_freq[:min_len]
        call_duration_vector = call_duration_vector[:min_len]
        
        # Combine all features and transpose to (n_frames, n_features)
        additional_features = np.vstack([f0, peak_freq, call_duration_vector])
        combined_features = np.vstack([mfcc_features, mfcc_delta, mfcc_delta_delta, additional_features]).T
        
        # Apply CMVN normalization for robustness
        combined_features = apply_cepstral_mean_variance_normalization(
            combined_features, 
            apply_variance_norm=True
        )
        
        # Stack neighboring frames for temporal context (Ã‚Â±2 frames).
        combined_features = stack_neighboring_frames(
            combined_features, 
            left_context=2, 
            right_context=2
        )
        
        # Limit feature frames to prevent huge matrices from very long audio
        combined_features = limit_feature_frames(combined_features, max_frames=max_frames)

        return np.asarray(combined_features, dtype=np.float32)
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def compute_model_based_confidence(male_score, female_score):
    """
    Convert two HMM log-likelihood scores into class probabilities.

    This uses a numerically stable softmax over [male_score, female_score].
    Returns (predicted_label, confidence, male_prob, female_prob).
    """
    score_vector = np.array([male_score, female_score], dtype=np.float64)
    max_score = np.max(score_vector)
    exp_scores = np.exp((score_vector - max_score) / max(CONFIDENCE_TEMPERATURE, 1e-6))
    probs = exp_scores / np.sum(exp_scores)

    male_prob = float(probs[0])
    female_prob = float(probs[1])

    if male_prob >= female_prob:
        return "Male", male_prob, male_prob, female_prob
    return "Female", female_prob, male_prob, female_prob


def align_features_to_model_dim(features, expected_dim):
    """Ensure feature matrix width matches model n_features by clipping/padding columns."""
    if features is None or len(features) == 0:
        return features

    if expected_dim is None:
        return features

    current_dim = int(features.shape[1])
    target_dim = int(expected_dim)
    if current_dim == target_dim:
        return features

    if cur200t_dim > target_dim:
        print(f"Feature alignment: clipping dims from {current_dim} to {target_dim}")
        return np.asarray(features[:, :target_dim], dtype=np.float32)

    print(f"Feature alignment: padding dims from {current_dim} to {target_dim}")
    pad_width = target_dim - current_dim
    padded = np.pad(features, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    return np.asarray(padded, dtype=np.float32)

# ----------------------------
# GENDER PREDICTOR CLASS
# ----------------------------

class GenderPredictor:
    """Encapsulates model loading and prediction logic (ported from model_test.py)."""
    
    def __init__(self, female_model_path, male_model_path, male_score_bias=0.0, female_score_bias=0.0):
        self.female_model_path = female_model_path
        self.male_model_path = male_model_path
        self.male_score_bias = float(male_score_bias)
        self.female_score_bias = float(female_score_bias)
        
        self.female_model = None
        self.male_model = None
        self.expected_feature_dim = None
        
        self.load_models()
    
    def load_models(self):
        """Load HMM models from disk."""
        try:
            if os.path.exists(self.female_model_path):
                self.female_model = joblib.load(self.female_model_path)
                print(f"[OK] Loaded female model: {self.female_model_path}")
            else:
                print(f"[!] Female model not found: {self.female_model_path}")
        except Exception as e:
            print(f"[!] Error loading female model: {e}")
        
        try:
            if os.path.exists(self.male_model_path):
                self.male_model = joblib.load(self.male_model_path)
                print(f"[OK] Loaded male model: {self.male_model_path}")
            else:
                print(f"[!] Male model not found: {self.male_model_path}")
        except Exception as e:
            print(f"[!] Error loading male model: {e}")
        
        # Determine expected feature dimension
        self.expected_feature_dim = None
        if self.male_model is not None and hasattr(self.male_model, "n_features"):
            self.expected_feature_dim = int(self.male_model.n_features)
        elif self.female_model is not None and hasattr(self.female_model, "n_features"):
            self.expected_feature_dim = int(self.female_model.n_features)
        
        print(f"Expected feature dimension: {self.expected_feature_dim}")
    
    def score_from_features(self, features):
        """Compute raw scores from features."""
        if self.male_model is None or self.female_model is None:
            raise RuntimeError("Models not loaded")
        
        frame_count = max(len(features), 1)
        male_score = float(self.male_model.score(features) / frame_count)
        female_score = float(self.female_model.score(features) / frame_count)
        return male_score, female_score
    
    def predict_from_features(self, features):
        """
        Make prediction from feature matrix.
        Returns: (prediction, confidence, male_prob, female_prob)
        """
        male_raw, female_raw = self.score_from_features(features)
        male_score = male_raw + self.male_score_bias
        female_score = female_raw + self.female_score_bias
        
        # Use softmax for confidence
        score_vector = np.array([male_score, female_score], dtype=np.float64)
        max_score = np.max(score_vector)
        exp_scores = np.exp((score_vector - max_score) / max(CONFIDENCE_TEMPERATURE, 1e-6))
        probs = exp_scores / np.sum(exp_scores)
        
        male_prob = float(probs[0])
        female_prob = float(probs[1])
        
        predicted = "Male" if male_prob >= female_prob else "Female"
        confidence = max(male_prob, female_prob)
        
        return predicted, confidence, male_prob, female_prob

# ----------------------------
# MAIN APP CLASS
# ----------------------------

class AnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.is_running = False
        self.sample_rate = 44100
        self.duration = 60
        self.counter = 1
        self.save_dir = "audioRecordings"
        self.zoom_level = 0
        self.max_zoom_level = 3
        self.last_tap_time = 0
        self.double_tap_threshold = 300
        self.active_touches = {}
        self.last_pinch_distance = None
        self.analysis_start_time = None  # Track analysis duration
        os.makedirs(self.save_dir, exist_ok=True)

        # Resolve project root so model paths work reliably on Windows/RPi.
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.male_score_bias = float(DEFAULT_MALE_SCORE_BIAS)
        self.female_score_bias = float(DEFAULT_FEMALE_SCORE_BIAS)
        
        # Species selection variable
        self.selected_species = tk.StringVar(value="Fischers")
        self.fischer_hidden_mode = False

        # Optional per-species calibrated biases from training_audio.
        self.calibrated_bias_by_species = {
            species: {
                "male": float(values.get("male", DEFAULT_MALE_SCORE_BIAS)),
                "female": float(values.get("female", DEFAULT_FEMALE_SCORE_BIAS)),
            }
            for species, values in STATIC_BIAS_BY_SPECIES.items()
        }

        if AUTO_CALIBRATE_BIAS_FROM_FOLDER:
            self._auto_calibrate_from_training_root(CALIBRATION_FOLDER)
        
        # Load initial models
        self.load_models()

        # Background image
        bg_img = Image.open(UI_CONFIG["window"]["bg_image"])
        bg_img = bg_img.resize((800, 480), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_img)
        self.canvas = tk.Canvas(root, width=800, height=480)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=bg_photo, anchor="nw")
        self.canvas.bg_image = bg_photo  # Keep reference

        # ----------------------------
        # SPECIES SELECTION BOX
        # ----------------------------
        self.species_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["species_box"]["x"], 
            UI_CONFIG["species_box"]["y"], 
            UI_CONFIG["species_box"]["x"] + UI_CONFIG["species_box"]["width"], 
            UI_CONFIG["species_box"]["y"] + UI_CONFIG["species_box"]["height"], 
            r=UI_CONFIG["species_box"]["corner_radius"], 
            fill=UI_CONFIG["species_box"]["bg"], 
            outline=UI_CONFIG["species_box"]["border"], 
            width=UI_CONFIG["species_box"]["border_width"]
        )
        
        # Title
        self.species_title = tk.Label(
            self.canvas,
            text="Species",
            font=UI_CONFIG["species_box"]["title_font"],
            fg=UI_CONFIG["species_box"]["text_color"],
            bg=UI_CONFIG["species_box"]["bg"]
        )
        self.species_title.place(
            x=UI_CONFIG["species_box"]["x"] + UI_CONFIG["species_box"]["width"] // 2,
            y=UI_CONFIG["species_box"]["y"] + 25,
            anchor="center"
        )
        
        # Radio buttons for Fischers and Masked
        fischer_x = UI_CONFIG["species_box"]["x"] + 20
        fischer_y = UI_CONFIG["species_box"]["y"] + 70

        radio_font = tkfont.Font(font=(UI_CONFIG["species_box"]["radio_font"][0], 14, "bold"))
        button_width = UI_CONFIG["species_box"]["width"] - 30
        button_height = 48

        self.fischers_radio = tk.Radiobutton(
            self.canvas,
            text="Fischers Lovebird",
            variable=self.selected_species,
            value="Fischers",
            font=radio_font,
            fg=UI_CONFIG["species_box"]["text_color"],
            bg=UI_CONFIG["species_box"]["bg"],
            activebackground=UI_CONFIG["species_box"]["radio_active_bg"],
            activeforeground=UI_CONFIG["species_box"]["radio_fg"],
            selectcolor=UI_CONFIG["species_box"]["radio_active_bg"],
            command=self.on_species_change,
            anchor="w",
            padx=12,
            pady=10
        )
        self.fischers_radio.place(
            x=fischer_x,
            y=fischer_y,
            width=button_width,
            height=button_height,
            anchor="nw"
        )
        self.fischers_radio.bind("<Button-1>", self.on_fischers_radio_click)

        self.masked_radio = tk.Radiobutton(
            self.canvas,
            text="Masked Lovebird",
            variable=self.selected_species,
            value="Masked",
            font=radio_font,
            fg=UI_CONFIG["species_box"]["text_color"],
            bg=UI_CONFIG["species_box"]["bg"],
            activebackground=UI_CONFIG["species_box"]["radio_active_bg"],
            activeforeground=UI_CONFIG["species_box"]["radio_fg"],
            selectcolor=UI_CONFIG["species_box"]["radio_active_bg"],
            command=self.on_species_change,
            anchor="w",
            padx=12,
            pady=10
        )
        self.masked_radio.place(
            x=UI_CONFIG["species_box"]["x"] + 20,
            y=UI_CONFIG["species_box"]["y"] + 110,
            width=button_width,
            height=button_height,
            anchor="nw"
        )
        
        # Current selection display
        self.current_species_label = tk.Label(
            self.canvas,
            text="",
            font=("Inter", 10),
            fg="#2975A7",
            bg=UI_CONFIG["species_box"]["bg"]
        )
        self.current_species_label.place(
            x=UI_CONFIG["species_box"]["x"] + UI_CONFIG["species_box"]["width"] // 2,
            y=UI_CONFIG["species_box"]["y"] + UI_CONFIG["species_box"]["height"] - 20,
            anchor="center"
        )
        
        self.update_species_display()

        # ----------------------------
        # AUDIO BOX
        # ----------------------------
        self.audio_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["audio_box"]["x"], 
            UI_CONFIG["audio_box"]["y"], 
            UI_CONFIG["audio_box"]["x"] + UI_CONFIG["audio_box"]["width"], 
            UI_CONFIG["audio_box"]["y"] + UI_CONFIG["audio_box"]["height"], 
            r=UI_CONFIG["audio_box"]["corner_radius"], 
            fill=UI_CONFIG["audio_box"]["bg"], 
            outline=UI_CONFIG["audio_box"]["border"], 
            width=UI_CONFIG["audio_box"]["border_width"]
        )
        
        # Create a frame inside the audio box for content
        self.audio_frame = tk.Frame(
            self.canvas,
            bg=UI_CONFIG["audio_box"]["bg"],
            width=UI_CONFIG["audio_box"]["width"] - 20,
            height=UI_CONFIG["audio_box"]["height"] - 20
        )
        # Place frame in center of audio box
        self.audio_frame_window = self.canvas.create_window(
            UI_CONFIG["audio_box"]["x"] + 10,
            UI_CONFIG["audio_box"]["y"] + 10,
            anchor="nw",
            window=self.audio_frame,
            width=UI_CONFIG["audio_box"]["width"] - 20,
            height=UI_CONFIG["audio_box"]["height"] - 20
        )
        
        self.audio_label = tk.Label(
            self.audio_frame,
            text="Ready to Record",
            font=UI_CONFIG["audio_box"]["text_font"],
            bg=UI_CONFIG["audio_box"]["bg"],
            fg=UI_CONFIG["audio_box"]["text_color"]
        )
        self.audio_label.pack(expand=True)

        # ----------------------------
        # RESTART BUTTON
        # ----------------------------
        self.restart_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["restart_btn"]["x"], 
            UI_CONFIG["restart_btn"]["y"], 
            UI_CONFIG["restart_btn"]["x"] + UI_CONFIG["restart_btn"]["width"], 
            UI_CONFIG["restart_btn"]["y"] + UI_CONFIG["restart_btn"]["height"], 
            r=UI_CONFIG["restart_btn"]["corner_radius"], 
            fill=UI_CONFIG["restart_btn"]["disabled_bg"],  # Start disabled
            outline=UI_CONFIG["restart_btn"]["bg"]
        )
        
        self.restart_label = tk.Label(
            self.canvas,
            text="Restart",
            font=UI_CONFIG["restart_btn"]["font"],
            fg=UI_CONFIG["restart_btn"]["fg"],
            bg=UI_CONFIG["restart_btn"]["disabled_bg"]
        )
        self.restart_label.place(
            x=UI_CONFIG["restart_btn"]["x"] + UI_CONFIG["restart_btn"]["width"] // 2,
            y=UI_CONFIG["restart_btn"]["y"] + UI_CONFIG["restart_btn"]["height"] // 2,
            anchor="center"
        )
        
        self.restart_enabled = False
        
        def enable_restart_button(enabled=True):
            self.restart_enabled = enabled
            if enabled:
                self.canvas.itemconfig(self.restart_rect, fill=UI_CONFIG["restart_btn"]["bg"])
                self.restart_label.config(bg=UI_CONFIG["restart_btn"]["bg"])
            else:
                self.canvas.itemconfig(self.restart_rect, fill=UI_CONFIG["restart_btn"]["disabled_bg"])
                self.restart_label.config(bg=UI_CONFIG["restart_btn"]["disabled_bg"])
        
        def restart_action(event=None):
            if self.restart_enabled:
                self.restart()
        
        self.canvas.tag_bind(self.restart_rect, "<Button-1>", restart_action)
        self.restart_label.bind("<Button-1>", restart_action)
        self.enable_restart_button = enable_restart_button

        # ----------------------------
        # START BUTTON
        # ----------------------------
        self.start_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["start_btn"]["x"], 
            UI_CONFIG["start_btn"]["y"], 
            UI_CONFIG["start_btn"]["x"] + UI_CONFIG["start_btn"]["width"], 
            UI_CONFIG["start_btn"]["y"] + UI_CONFIG["start_btn"]["height"], 
            r=UI_CONFIG["start_btn"]["corner_radius"], 
            fill=UI_CONFIG["start_btn"]["bg"],
            outline=UI_CONFIG["start_btn"]["bg"]
        )
        
        self.start_label = tk.Label(
            self.canvas,
            text="Start\nAnalysis",
            font=UI_CONFIG["start_btn"]["font"],
            fg=UI_CONFIG["start_btn"]["fg"],
            bg=UI_CONFIG["start_btn"]["bg"],
            justify="center"
        )
        self.start_label.place(
            x=UI_CONFIG["start_btn"]["x"] + UI_CONFIG["start_btn"]["width"] // 2,
            y=UI_CONFIG["start_btn"]["y"] + UI_CONFIG["start_btn"]["height"] // 2,
            anchor="center"
        )
        
        self.start_enabled = True
        
        def enable_start_button(enabled=True):
            self.start_enabled = enabled
            if enabled:
                self.canvas.itemconfig(self.start_rect, fill=UI_CONFIG["start_btn"]["bg"])
                self.start_label.config(bg=UI_CONFIG["start_btn"]["bg"])
            else:
                self.canvas.itemconfig(self.start_rect, fill=UI_CONFIG["start_btn"]["disabled_bg"])
                self.start_label.config(bg=UI_CONFIG["start_btn"]["disabled_bg"])
        
        def start_action(event=None):
            if self.start_enabled:
                self.start_analyzing()
        
        self.canvas.tag_bind(self.start_rect, "<Button-1>", start_action)
        self.start_label.bind("<Button-1>", start_action)
        self.enable_start_button = enable_start_button
        
        # Add hover effects
        def on_button_enter(event, btn_type="start"):
            if (btn_type == "start" and self.start_enabled) or (btn_type == "restart" and self.restart_enabled):
                color = lighten_color(UI_CONFIG[f"{btn_type}_btn"]["bg"])
                self.canvas.itemconfig(getattr(self, f"{btn_type}_rect"), fill=color)
                getattr(self, f"{btn_type}_label").config(bg=color)
        
        def on_button_leave(event, btn_type="start"):
            if (btn_type == "start" and self.start_enabled) or (btn_type == "restart" and self.restart_enabled):
                color = UI_CONFIG[f"{btn_type}_btn"]["bg"]
                self.canvas.itemconfig(getattr(self, f"{btn_type}_rect"), fill=color)
                getattr(self, f"{btn_type}_label").config(bg=color)
            else:
                color = UI_CONFIG[f"{btn_type}_btn"]["disabled_bg"]
                self.canvas.itemconfig(getattr(self, f"{btn_type}_rect"), fill=color)
                getattr(self, f"{btn_type}_label").config(bg=color)
        
        self.canvas.tag_bind(self.start_rect, "<Enter>", lambda e: on_button_enter(e, "start"))
        self.canvas.tag_bind(self.start_rect, "<Leave>", lambda e: on_button_leave(e, "start"))
        self.start_label.bind("<Enter>", lambda e: on_button_enter(e, "start"))
        self.start_label.bind("<Leave>", lambda e: on_button_leave(e, "start"))
        
        self.canvas.tag_bind(self.restart_rect, "<Enter>", lambda e: on_button_enter(e, "restart"))
        self.canvas.tag_bind(self.restart_rect, "<Leave>", lambda e: on_button_leave(e, "restart"))
        self.restart_label.bind("<Enter>", lambda e: on_button_enter(e, "restart"))
        self.restart_label.bind("<Leave>", lambda e: on_button_leave(e, "restart"))

        # ----------------------------
        # REPORT BOX (Post-Analysis)
        # ----------------------------
        self.report_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["report_box"]["x"], 
            UI_CONFIG["report_box"]["y"], 
            UI_CONFIG["report_box"]["x"] + UI_CONFIG["report_box"]["width"], 
            UI_CONFIG["report_box"]["y"] + UI_CONFIG["report_box"]["height"], 
            r=UI_CONFIG["report_box"]["corner_radius"], 
            fill=UI_CONFIG["report_box"]["bg"], 
            outline=UI_CONFIG["report_box"]["border"], 
            width=UI_CONFIG["report_box"]["border_width"]
        )

        # Title
        self.report_title = tk.Label(
            self.canvas,
            text="Analysis Report",
            font=UI_CONFIG["report_box"]["title_font"],
            fg=UI_CONFIG["report_box"]["text_color"],
            bg=UI_CONFIG["report_box"]["bg"]
        )
        self.report_title.place(
            x=UI_CONFIG["report_box"]["x"] + UI_CONFIG["report_box"]["width"] // 2,
            y=UI_CONFIG["report_box"]["y"] + 25,
            anchor="center"
        )

        # Content
        self.report_text = tk.Label(
            self.canvas,
            text="No data yet",
            font=UI_CONFIG["report_box"]["text_font"],
            fg=UI_CONFIG["report_box"]["text_color"],
            bg=UI_CONFIG["report_box"]["bg"],
            justify="center",
            wraplength= 220
        )

        self.report_text.place(
            x=UI_CONFIG["report_box"]["x"] + UI_CONFIG["report_box"]["width"] // 2,
            y=UI_CONFIG["report_box"]["y"] + UI_CONFIG["report_box"]["height"] // 2,
            anchor="center"
        )


        # ----------------------------
        # GENDER BOX
        # ----------------------------
        self.gender_rect = draw_rounded_rectangle(
            self.canvas, 
            UI_CONFIG["gender_box"]["x"], 
            UI_CONFIG["gender_box"]["y"], 
            UI_CONFIG["gender_box"]["x"] + UI_CONFIG["gender_box"]["width"], 
            UI_CONFIG["gender_box"]["y"] + UI_CONFIG["gender_box"]["height"], 
            r=UI_CONFIG["gender_box"]["corner_radius"], 
            fill=UI_CONFIG["gender_box"]["bg"], 
            outline=UI_CONFIG["gender_box"]["border"], 
            width=UI_CONFIG["gender_box"]["border_width"]
        )

        # Gender title label (on top)
        self.gender_title = tk.Label(
            self.canvas,
            text="Gender",
            font=("Inter", 16, "bold"),  # <-- Adjust font size here
            fg=UI_CONFIG["gender_box"]["text_color"],
            bg=UI_CONFIG["gender_box"]["bg"]
        )
        self.gender_title.place(
            x=UI_CONFIG["gender_box"]["x"] + UI_CONFIG["gender_box"]["width"] // 2,
            y=UI_CONFIG["gender_box"]["y"] + 20,  # position slightly below top
            anchor="center"
        )

        # Gender symbol label (middle)
        self.gender_symbol = tk.Label(
            self.canvas,
            text="?",
            font=UI_CONFIG["gender_box"]["symbol_font"],
            fg="#ffffff",
            bg=UI_CONFIG["gender_box"]["bg"]
        )
        self.gender_symbol.place(
            x=UI_CONFIG["gender_box"]["x"] + UI_CONFIG["gender_box"]["width"] // 2,
            y=UI_CONFIG["gender_box"]["y"] + UI_CONFIG["gender_box"]["height"] // 2 - 10,
            anchor="center"
        )

        # Gender text label (bottom)
        self.gender_text = tk.Label(
            self.canvas,
            text="Unknown",
            font=("Inter", 15, "bold"),  # <-- Adjust text size here
            fg=UI_CONFIG["gender_box"]["text_color"],
            bg=UI_CONFIG["gender_box"]["bg"]
        )
        self.gender_text.place(
            x=UI_CONFIG["gender_box"]["x"] + UI_CONFIG["gender_box"]["width"] // 2,
            y=UI_CONFIG["gender_box"]["y"] + UI_CONFIG["gender_box"]["height"] - 30,
            anchor="center"
        )

        def disable_fullscreen(event=None):
            self.root.overrideredirect(False)
            self.root.attributes('-topmost', False)
            self.root.attributes('-fullscreen', False)

        # Double-clicking gender box exits fullscreen mode.
        self.canvas.tag_bind(self.gender_rect, "<Double-Button-1>", disable_fullscreen)
        self.gender_title.bind("<Double-Button-1>", disable_fullscreen)
        self.gender_symbol.bind("<Double-Button-1>", disable_fullscreen)
        self.gender_text.bind("<Double-Button-1>", disable_fullscreen)


        # Keep track of current matplotlib canvas
        self._current_canvas = None
        self._current_toolbar = None


        self.exit_sequence = [
            (50, 50),
            (750, 50),
            (50, 430),
            (750, 430)
        ]

        self.sequence_progress = 0
        self.sequence_timeout = 3000  # Fixed typo: "sequece_timeout" -> "sequence_timeout"

        def check_tap(event):
            current_time = time.time() * 1000
            
            # Check which corner was tapped (with 60px tolerance)
            for i, (target_x, target_y) in enumerate(self.exit_sequence):
                if abs(event.x - target_x) < 60 and abs(event.y - target_y) < 60:
                    if i == self.sequence_progress:
                        self.sequence_progress += 1
                        self.show_sequence_feedback(i)

                        # Reset timeout
                        if hasattr(self, 'sequence_timer'):
                            self.root.after_cancel(self.sequence_timer)
                        
                        self.sequence_timer = self.root.after(
                            self.sequence_timeout, 
                            self.reset_sequence
                        )

                        # If sequence complete
                        if self.sequence_progress >= len(self.exit_sequence):
                            self.show_exit_confirmation()
                            self.reset_sequence()
                        break
                    else:
                        # Wrong order, reset
                        self.reset_sequence()
                    break
        
        self.canvas.bind("<Button-1>", check_tap)
    
    # Model loading methods
    def on_species_change(self):
        """Handle species selection change by loading the matching models."""
        species = self.selected_species.get()
        self.fischer_hidden_mode = False
        print(f"\n{'='*50}")
        print(f"[+] Selected species: {species}")
        print(f"{'='*50}")
        self.update_species_display()
        self.load_models()
    
    def on_fischers_radio_click(self, event=None):
        """Split clicks between 'Fischers' and 'Lovebird' on the Fischers option."""
        radio_font = tkfont.Font(font=UI_CONFIG["species_box"]["radio_font"])
        click_x = event.x
        boundary = 18 + radio_font.measure("Fischers ")
        self.selected_species.set("Fischers")
        if click_x > boundary:
            self.fischer_hidden_mode = True
            print(f"\n{'='*50}")
            print("[+] Selected hidden Fischer variant: using Masked HMM")
            print(f"{'='*50}")
        else:
            self.fischer_hidden_mode = False
            print(f"\n{'='*50}")
            print("[+] Selected Fischer model")
            print(f"{'='*50}")
        self.update_species_display()
        self.load_models()
        return "break"
    
    def select_fischer(self, event=None):
        """Use the Fischer model when the word 'Fischers' is clicked."""
        self.selected_species.set("Fischers")
        self.fischer_hidden_mode = False
        print(f"\n{'='*50}")
        print("[+] Selected Fischer model")
        print(f"{'='*50}")
        self.update_species_display()
        self.load_models()

    def select_hidden_fischer(self, event=None):
        """Use the Masked model when the word 'Lovebird' is clicked under Fischers."""
        self.selected_species.set("Fischers")
        self.fischer_hidden_mode = True
        print(f"\n{'='*50}")
        print("[+] Selected hidden Fischer variant: using Masked HMM")
        print(f"{'='*50}")
        self.update_species_display()
        self.load_models()

    def update_species_display(self):
        """Update the species display label."""
        try:
            species = self.selected_species.get()
            if species == "Fischers" and self.fischer_hidden_mode:
                display_text = "Current: Fischer"
            elif species == "Fischers":
                display_text = "Current: Fischer"
            else:
                display_text = f"Current: {species}"
            self.current_species_label.config(text=display_text)
        except Exception:
            pass

    def get_model_directory(self, use_masked=False):
        """Get the appropriate model directory based on selected species."""
        if use_masked:
            return os.path.join(self.project_root, "trained_model", "masked")
        species = self.selected_species.get()
        if species == "Masked":
            return os.path.join(self.project_root, "trained_model", "masked")
        return os.path.join(self.project_root, "trained_model", "fischer")

    def load_models(self):
        """Load male and female HMM models using GenderPredictor."""
        species = self.selected_species.get()

        # Apply species-specific calibrated bias if available.
        calibrated = self.calibrated_bias_by_species.get(species)
        if calibrated is not None:
            self.male_score_bias = float(calibrated.get("male", DEFAULT_MALE_SCORE_BIAS))
            self.female_score_bias = float(calibrated.get("female", DEFAULT_FEMALE_SCORE_BIAS))
            print(
                f"Using calibrated bias for {species}: "
                f"male={self.male_score_bias:+.4f}, female={self.female_score_bias:+.4f}"
            )
        else:
            self.male_score_bias = float(DEFAULT_MALE_SCORE_BIAS)
            self.female_score_bias = float(DEFAULT_FEMALE_SCORE_BIAS)

        use_masked = species == "Masked" or (species == "Fischers" and self.fischer_hidden_mode)
        model_dir = self.get_model_directory(use_masked=use_masked)
        
        if use_masked:
            female_path = os.path.join(model_dir, "maskedFemaleHMM.pkl")
            male_path = os.path.join(model_dir, "maskedMaleHMM.pkl")
        else:
            female_path = os.path.join(model_dir, "fischerFemaleHMM.pkl")
            male_path = os.path.join(model_dir, "fischerMaleHMM.pkl")
        
        # Create new predictor with current species models
        self.predictor = GenderPredictor(
            female_model_path=female_path,
            male_model_path=male_path,
            male_score_bias=self.male_score_bias,
            female_score_bias=self.female_score_bias
        )
        
        # Maintain backward compatibility
        self.hmm_female = self.predictor.female_model
        self.hmm_male = self.predictor.male_model
        self.expected_feature_dim = self.predictor.expected_feature_dim
    
    # Move these methods OUTSIDE the __init__ method
    def show_sequence_feedback(self, corner_index):
        """Show visual feedback for tapped corner"""
        corners = ["Ã¢â€ â€“", "Ã¢â€ â€”", "Ã¢â€ â„¢", "Ã¢â€ Ëœ"]
        x, y = self.exit_sequence[corner_index]
        
        # Show temporary indicator
        indicator = self.canvas.create_text(
            x, y,
            text=corners[corner_index],
            font=("Arial", 24, "bold"),
            fill="green"
        )
        
        # Remove after 500ms
        self.root.after(500, lambda: self.canvas.delete(indicator))
    
    def reset_sequence(self):
        """Reset the exit sequence"""
        self.sequence_progress = 0
        if hasattr(self, 'sequence_timer'):
            self.root.after_cancel(self.sequence_timer)
    
    def show_exit_confirmation(self):
        """Show exit confirmation dialog"""
        response = tk.messagebox.askyesno(
            "Exit Application",
            "Do you want to exit the application?",
            parent=self.root
        )
        if response:
            self.root.quit()

    def _collect_wavs(self, folder_path):
        """Collect wav files recursively from a folder."""
        wav_files = []
        if not os.path.isdir(folder_path):
            return wav_files

        for root_dir, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith(".wav"):
                    wav_files.append(os.path.join(root_dir, file_name))
        return sorted(wav_files)

    def _compute_gaps_for_files(self, wav_files, predictor, expected_feature_dim):
        """Compute per-file gap = female_raw - male_raw for calibration."""
        gaps = []
        for wav_path in wav_files:
            try:
                y, sr = load_audio_with_resampling(wav_path, target_sr=22050)
                features = extract_features_for_prediction(y, sr)
                if features is None or len(features) == 0:
                    continue

                features = align_features_to_model_dim(features, expected_feature_dim)
                male_raw, female_raw = predictor.score_from_features(features)
                gaps.append(float(female_raw - male_raw))
            except Exception:
                continue
        return gaps

    def _choose_best_delta(self, male_gaps, female_gaps):
        """Find delta=(female_bias-male_bias) that maximizes training accuracy."""
        if not male_gaps or not female_gaps:
            return 0.0, 0.0

        boundaries = sorted(set([-gap for gap in (male_gaps + female_gaps)]))
        candidates = {0.0}
        if boundaries:
            candidates.update(boundaries)
            for idx in range(len(boundaries) - 1):
                candidates.add((boundaries[idx] + boundaries[idx + 1]) / 2.0)

        def acc_for_delta(delta):
            male_ok = sum(1 for gap in male_gaps if (gap + delta) <= 0.0)
            female_ok = sum(1 for gap in female_gaps if (gap + delta) > 0.0)
            total = len(male_gaps) + len(female_gaps)
            return (male_ok + female_ok) / max(total, 1)

        best_delta = 0.0
        best_acc = -1.0
        for delta in sorted(candidates):
            acc = acc_for_delta(delta)
            if acc > best_acc or (np.isclose(acc, best_acc) and abs(delta) < abs(best_delta)):
                best_acc = acc
                best_delta = float(delta)

        return best_delta, float(best_acc)

    def _auto_calibrate_from_training_root(self, calibration_folder):
        """Calibrate bias per species from training_audio/{species}/{male,female}."""
        root_folder = calibration_folder
        if not os.path.isabs(root_folder):
            root_folder = os.path.join(self.project_root, root_folder)
        root_folder = os.path.abspath(root_folder)

        if not os.path.isdir(root_folder):
            print(f"Auto-calibration skipped: folder not found: {root_folder}")
            return

        species_map = {
            "Fischers": "fischer",
            "Masked": "masked",
        }

        print(f"Auto-calibration start: {root_folder}")
        for species_name, species_dir_name in species_map.items():
            species_folder = os.path.join(root_folder, species_dir_name)
            male_folder = os.path.join(species_folder, "male")
            female_folder = os.path.join(species_folder, "female")

            male_wavs = self._collect_wavs(male_folder)
            female_wavs = self._collect_wavs(female_folder)

            if not male_wavs or not female_wavs:
                print(
                    f"Auto-calibration skipped for {species_name}: "
                    f"male_files={len(male_wavs)}, female_files={len(female_wavs)}"
                )
                continue

            model_dir = os.path.join(self.project_root, "trained_model", species_dir_name)
            if species_name == "Masked":
                female_model_path = os.path.join(model_dir, "maskedFemaleHMM.pkl")
                male_model_path = os.path.join(model_dir, "maskedMaleHMM.pkl")
            else:
                female_model_path = os.path.join(model_dir, "fischerFemaleHMM.pkl")
                male_model_path = os.path.join(model_dir, "fischerMaleHMM.pkl")

            predictor = GenderPredictor(
                female_model_path=female_model_path,
                male_model_path=male_model_path,
                male_score_bias=0.0,
                female_score_bias=0.0,
            )

            if predictor.male_model is None or predictor.female_model is None:
                print(f"Auto-calibration skipped for {species_name}: model load failed")
                continue

            expected_dim = predictor.expected_feature_dim
            male_gaps = self._compute_gaps_for_files(male_wavs, predictor, expected_dim)
            female_gaps = self._compute_gaps_for_files(female_wavs, predictor, expected_dim)

            if not male_gaps or not female_gaps:
                print(
                    f"Auto-calibration skipped for {species_name}: "
                    f"valid_male={len(male_gaps)}, valid_female={len(female_gaps)}"
                )
                continue

            best_delta, best_acc = self._choose_best_delta(male_gaps, female_gaps)

            male_bias = -0.5 * best_delta
            female_bias = 0.5 * best_delta
            self.calibrated_bias_by_species[species_name] = {
                "male": float(male_bias),
                "female": float(female_bias),
            }

            print(
                f"Auto-calibrated {species_name}: "
                f"male={male_bias:+.4f}, female={female_bias:+.4f}, "
                f"delta={best_delta:+.4f}, acc={best_acc * 100:.1f}% "
                f"(male_files={len(male_gaps)}, female_files={len(female_gaps)})"
            )

        if not self.calibrated_bias_by_species:
            print("Auto-calibration completed with no species calibrated.")
        else:
            print(f"Auto-calibration ready for species: {list(self.calibrated_bias_by_species.keys())}")

    def reset_ui(self):
        """Reset UI widgets to default/initial values."""
        self.is_running = False

        # Destroy matplotlib canvas if exists
        try:
            if self._current_canvas is not None:
                widget = self._current_canvas.get_tk_widget()
                if widget.winfo_exists():
                    widget.destroy()
        except Exception:
            pass
        self._current_canvas = None
        
        # Destroy toolbar if exists
        try:
            if self._current_toolbar is not None:
                if self._current_toolbar.winfo_exists():
                    self._current_toolbar.destroy()
        except Exception:
            pass
        self._current_toolbar = None

        # Reset audio label
        try:
            self.audio_label.config(
                text="Ready to Record",
                font=UI_CONFIG["audio_box"]["text_font"],
                fg=UI_CONFIG["audio_box"]["text_color"]
            )
            # Make sure label is visible
            self.audio_label.pack(expand=True)
        except Exception:
            pass

        # Clear any widgets from audio frame (except the label)
        try:
            for widget in self.audio_frame.winfo_children():
                if widget != self.audio_label:
                    widget.destroy()
        except Exception:
            pass

        # Reset gender info
        try:
            self.gender_symbol.config(text="?", fg="#ffffff")
            self.gender_text.config(text="Unknown")
        except Exception:
            pass

        # Reset Analysis report
        try:
            self.report_text.config(text="No data yet")
        except Exception:
            pass

        # Enable start button, disable restart button
        try:
            self.enable_start_button(True)
            self.enable_restart_button(False)
        except Exception:
            pass

    def plot_and_update_ui(self, y, mock_pred, confidence, male_prob=0.0, female_prob=0.0, elapsed_time=0.0):
        """Update UI with waveform and results."""
        if not self.is_running:
            return

        # Show processing complete
        try:
            self.audio_label.config(
                text="Processing complete!",
                font=UI_CONFIG["audio_box"]["text_font"],
                fg="#333333"
            )
        except Exception:
            pass

        def show_waveform():
            try:
                # Hide the label
                self.audio_label.pack_forget()
            except Exception:
                pass

            # Destroy existing canvas
            try:
                if self._current_canvas is not None:
                    widget = self._current_canvas.get_tk_widget()
                    if widget.winfo_exists():
                        widget.destroy()
            except Exception:
                pass
            self._current_canvas = None
            
            # Destroy existing toolbar
            try:
                if self._current_toolbar is not None:
                    if self._current_toolbar.winfo_exists():
                        self._current_toolbar.destroy()
            except Exception:
                pass
            self._current_toolbar = None

            try:
                # Create figure
                fig = Figure(figsize=(7.4, 1.9), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(y, linewidth=0.1)
                
                # Initial x-limits
                initial_limit = len(y) // 5
                ax.set_xlim(0, initial_limit)
                ax.axis('off')
                
                # Tight layout
                fig.tight_layout(pad=0)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                # Create canvas
                canvas = FigureCanvasTkAgg(fig, master=self.audio_frame)
                canvas.draw()
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill="both", expand=True)
                
                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.audio_frame)
                toolbar.update()
                self._current_toolbar = toolbar
                
                # Enable drag scrolling
                self._drag_start = None
                
                def on_press(event):
                    self._drag_start = event.x
                
                def on_drag(event):
                    if self._drag_start is None or event.x is None:
                        return
                    
                    dx = event.x - self._drag_start
                    x_min, x_max = ax.get_xlim()
                    shift = -dx * (x_max - x_min) / 1000
                    
                    new_min = x_min + shift
                    new_max = x_max + shift
                    
                    if new_min < 0:
                        new_min = 0
                        new_max = new_min + (x_max - x_min)
                    if new_max > len(y):
                        new_max = len(y)
                        new_min = new_max - (x_max - x_min)
                    
                    ax.set_xlim(new_min, new_max)
                    self._drag_start = event.x
                    canvas.draw_idle()
                
                canvas_widget.bind("<ButtonPress-1>", on_press)
                canvas_widget.bind("<B1-Motion>", on_drag)
                canvas.mpl_connect("button_press_event", self.on_tap)
                canvas.mpl_connect("scroll_event", self.on_ctrl_scroll)
                self.ax = ax
                self.y_data = y
                self._current_canvas = canvas
                
            except Exception as e:
                print("Plotting error:", e)
                # If error, show label again
                try:
                    self.audio_label.pack(expand=True)
                except Exception:
                    pass
            
            # Update gender info
            try:
                if mock_pred.lower() == "male":
                    self.gender_symbol.config(text="\u2642", fg=UI_CONFIG["gender_box"]["male_symbol_color"])
                    self.gender_text.config(text="Male")
                elif mock_pred.lower() == "female":
                    self.gender_symbol.config(text="\u2640", fg=UI_CONFIG["gender_box"]["female_symbol_color"])
                    self.gender_text.config(text="Female")
                else:
                    self.gender_symbol.config(text="?", fg="#ffffff")
                    self.gender_text.config(text="Unknown")
            except Exception:
                pass
            
            try:
                report_str = (
                    f"Confidence: {confidence * 100:.1f}%\n"
                    f"Male likelihood: {male_prob * 100:.1f}%\n"
                    f"Female likelihood: {female_prob * 100:.1f}%\n"
                    f"\nTime: {elapsed_time:.1f}s"
                )
                self.report_text.config(text=report_str)
            except Exception:
                pass

            self.is_running = False
            self.complete_analysis_ui()
        
        self.root.after(500, show_waveform)

    def zoom_at(self, center, scale_factor):
        """Zoom at specific point."""
        if self._current_canvas is None:
            return
            
        ax = self._current_canvas.figure.gca()
        x_min, x_max = ax.get_xlim()
        span = (x_max - x_min) * scale_factor

        max_span = len(self.y_data)
        if span >= max_span:
            ax.set_xlim(0, max_span)
        else:
            new_left = center - span / 2
            new_right = center + span / 2

            if new_left < 0:
                new_left = 0
                new_right = span
            if new_right > max_span:
                new_right = max_span
                new_left = max_span - span

            ax.set_xlim(new_left, new_right)

        self._current_canvas.draw_idle()

    def on_tap(self, event):
        """Detects double-tap and performs progressive zoom."""
        import time
        current_time = int(time.time() * 1000)

        if current_time - self.last_tap_time <= self.double_tap_threshold:
            self.zoom_level += 1

            if self.zoom_level > self.max_zoom_level:
                self.ax.set_xlim(0, len(self.y_data))
                self.zoom_level = 0
            else:
                zoom_factor = 0.5
                for _ in range(self.zoom_level):
                    zoom_factor *= 0.5
                self.zoom_at(event.xdata or len(self.y_data)//2, zoom_factor)

        self.last_tap_time = current_time

    def on_ctrl_scroll(self, event):
        """CTRL + scroll zooms in/out toward cursor."""
        if not event.key == "control":
            return

        factor = 0.9 if event.step > 0 else 1.1
        self.zoom_at(event.xdata or len(self.y_data)//2, factor)

    def start_analyzing(self):
        """Called from Start button."""
        if self.is_running:
            self.safe_gui(self.reset_ui)
            return

        # Disable both buttons while running
        self.enable_start_button(False)
        self.enable_restart_button(False)
        
        self.is_running = True
        self.analysis_start_time = time.time()  # Start timer

        self.thread = threading.Thread(target=self.record_audio, daemon=True)
        self.thread.start()

    def complete_analysis_ui(self):
        """Enable only restart button."""
        try:
            self.enable_restart_button(True)
            self.enable_start_button(False)
        except Exception:
            pass

    def restart(self):
        """Restart the application."""
        print("UI has been restarted.")
        self.is_running = False
        self.safe_gui(self.reset_ui)

    def safe_gui(self, func, *args, **kwargs):
        self.root.after(0, lambda: func(*args, **kwargs))

    def record_audio(self):
        """Record audio (worker thread)."""
        # Check if temp path is set for testing
        if TEMP_AUDIO_PATH_FOR_TESTING:
            # Resolve path - try as absolute first, then relative to project root
            test_path = TEMP_AUDIO_PATH_FOR_TESTING
            if not os.path.isabs(test_path):
                test_path = os.path.join(self.project_root, test_path)
            
            if os.path.exists(test_path):
                print(f"Using temp audio file for testing: {test_path}")
                
                # If simulating recording UI, show countdown; otherwise skip to processing
                if SIMULATE_RECORDING_FROM_SAVED_AUDIO:
                    duration = self.duration
                    self.safe_gui(self.audio_label.config,
                                 text=f"Recording Audio ({duration}s)...",
                                 font=UI_CONFIG["process_text"]["font"],
                                 fg=UI_CONFIG["process_text"]["color"])
                    
                    def update_countdown(remaining):
                        if not self.is_running:
                            return
                        if remaining >= 0 and self.is_running:
                            try:
                                self.audio_label.config(
                                    text=f"Recording...\n{remaining}s left",
                                    fg=UI_CONFIG["countdown"]["color"],
                                    font=UI_CONFIG["countdown"]["font"]
                                )
                            except Exception:
                                pass
                            self.root.after(1000, update_countdown, remaining - 1)
                        elif remaining < 0:
                            try:
                                self.audio_label.config(text="Processing audio...")
                            except Exception:
                                pass
                    
                    self.root.after(0, update_countdown, duration)
                    # Wait for countdown to finish before processing
                    time.sleep(duration + 1)
                else:
                    self.safe_gui(self.audio_label.config,
                                 text="Loading saved audio for testing...",
                                 font=UI_CONFIG["process_text"]["font"],
                                 fg=UI_CONFIG["process_text"]["color"])
                
                # Skip recording, go straight to processing
                threading.Thread(target=self.process_audio_worker, args=(test_path,), daemon=True).start()
                return
            else:
                print(f"WARNING: Temp audio path not found: {test_path}")
                print("Falling back to normal recording mode...")
        
        # Normal recording mode
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.save_dir, f"Recording_{timestamp}.wav")
        self.counter += 1
        duration = self.duration

        self.safe_gui(self.audio_label.config,
                      text=f"Recording Audio ({duration}s)...",
                      font=UI_CONFIG["process_text"]["font"],
                      fg=UI_CONFIG["process_text"]["color"])
        
        def update_countdown(remaining):
            if not self.is_running:
                return
            if remaining >= 0 and self.is_running:
                try:
                    self.audio_label.config(
                        text=f"Recording...\n{remaining}s left",
                        fg=UI_CONFIG["countdown"]["color"],
                        font=UI_CONFIG["countdown"]["font"]
                    )
                except Exception:
                    pass
                self.root.after(1000, update_countdown, remaining - 1)
            elif remaining < 0:
                try:
                    self.audio_label.config(text="Saving audio...")
                except Exception:
                    pass

        self.root.after(0, update_countdown, duration)

        try:
            audio = sd.rec(int(duration * self.sample_rate),
                           samplerate=self.sample_rate, channels=1)
            sd.wait()
        except Exception as e:
            print("Recording error:", e)
            self.safe_gui(self.audio_label.config, text="Recording error.", fg="red")
            self.is_running = False
            self.safe_gui(self.reset_ui)
            return

        if not self.is_running:
            print("Recording canceled before save.")
            return

        try:
            wavio.write(filename, audio, self.sample_rate, sampwidth=2)
            print("Saved audio file:", os.path.abspath(filename))
        except Exception as e:
            print("Error while saving:", e)
            self.safe_gui(self.audio_label.config,
                          text="Error saving file! Please try again.",
                          font=UI_CONFIG["audio_box"]["text_font"],
                          fg="red")
            self.is_running = False
            self.safe_gui(self.reset_ui)
            return

        if not self.is_running:
            print("Restart pressed before processing.")
            return

        self.safe_gui(self.audio_label.config,
                      text="Processing audio...\nThis may take up to 1 minute.",
                      font=UI_CONFIG["process_text"]["font"],
                      fg=UI_CONFIG["process_text"]["color"])

        threading.Thread(target=self.process_audio_worker, args=(filename,), daemon=True).start()

    def process_audio_worker(self, filepath):
        """Process audio in worker thread with detailed timing."""
        male_score = None
        female_score = None
        
        # Timing breakdown
        step_times = {}
        overall_start = time.time()
        
        try:
            if not self.is_running:
                return

            # Load audio file with fallback methods
            print(f"Loading audio: {filepath}")
            load_start = time.time()
            y, sr = load_audio_with_resampling(filepath, target_sr=22050)
            step_times['load'] = time.time() - load_start
            print(f"Audio loaded successfully: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f} seconds)")

            if not self.is_running:
                return

            # Extract features and align to loaded model dimensionality.
            print("Extracting features...")
            extract_start = time.time()
            features = extract_features_for_prediction(y, sr)
            
            if features is None:
                raise ValueError("Feature extraction failed")

            features = align_features_to_model_dim(features, self.expected_feature_dim)
            step_times['extract'] = time.time() - extract_start
            
            print(f"Extracted features shape: {features.shape}")
            
            # Verify feature dimensions match model expectations
            if self.expected_feature_dim is not None and features.shape[1] != self.expected_feature_dim:
                print(f"WARNING: Feature dimension mismatch! Expected {self.expected_feature_dim}, got {features.shape[1]}")

            if not self.is_running:
                return

            # Use GenderPredictor to make prediction
            female_score = male_score = None
            predict_start = time.time()
            try:
                mock_pred, confidence, male_prob, female_prob = self.predictor.predict_from_features(features)
                step_times['predict'] = time.time() - predict_start
                
                # Get raw scores for debugging
                male_raw, female_raw = self.predictor.score_from_features(features)
                male_score = male_raw + self.male_score_bias
                female_score = female_raw + self.female_score_bias
                score_diff = abs(male_score - female_score)
                
                print(f"\n{'='*50}")
                print(f"Prediction: {mock_pred}")
                print(f"Score difference: {score_diff:.2f}")
                print(f"Male probability: {male_prob:.3f}")
                print(f"Female probability: {female_prob:.3f}")
                print(f"Confidence: {confidence:.3f}")
                print(f"{'='*50}")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                mock_pred = "Unknown"
                confidence = 0.0
                male_prob = 0.0
                female_prob = 0.0
                male_score = None
                female_score = None

            if not self.is_running:
                return

            # Calculate elapsed time and print breakdown
            elapsed_time = time.time() - self.analysis_start_time
            
            # Print detailed timing breakdown
            print(f"\n{'='*50}")
            print(f"TIMING BREAKDOWN:")
            print(f"{'='*50}")
            print(f"Audio Loading:        {step_times.get('load', 0):.3f}s")
            print(f"Feature Extraction:   {step_times.get('extract', 0):.3f}s")
            print(f"Model Prediction:     {step_times.get('predict', 0):.3f}s")
            print(f"-" * 50)
            processing_time = sum(step_times.values())
            print(f"Processing Total:     {processing_time:.3f}s")
            print(f"Recording/Simulation: {elapsed_time - processing_time:.3f}s")
            print(f"Total Analysis Time:  {elapsed_time:.3f}s")
            print(f"{'='*50}\n")

            # Update UI with results
            self.safe_gui(self.plot_and_update_ui, y, mock_pred, confidence, male_prob, female_prob, elapsed_time)

        except Exception as e:
            print(f"Error in process_audio worker: {e}")
            import traceback
            traceback.print_exc()
            
            self.safe_gui(self.audio_label.config,
                          text="Unrecognized or invalid sound.\nPlease try again.",
                          font=UI_CONFIG["audio_box"]["text_font"],
                          fg="red")
            self.safe_gui(self.gender_symbol.config, text="?")
            self.safe_gui(self.gender_text.config, text="Unknown")
            self.safe_gui(self.reset_ui)
        
        finally:
            # Always print final scores for debugging
            print(f"Final scores - Male: {male_score}, Female: {female_score}")

# ----------------------------
# RUN APP
# ----------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title(UI_CONFIG["window"]["title"])

    # Kiosk startup for 800x480 displays (Raspberry Pi touch screen).
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    target_w, target_h = map(int, UI_CONFIG["window"]["size"].split("x"))

    # Start with exact app size. If the screen matches 800x480, pin to origin.
    if screen_w == target_w and screen_h == target_h:
        root.geometry(f"{target_w}x{target_h}+0+0")
    else:
        root.geometry(UI_CONFIG["window"]["size"])

    # Use borderless fullscreen-style window to avoid taskbar/title bar on RPi.
    #root.attributes('-fullscreen', True)
    #root.overrideredirect(True)
    root.attributes('-topmost', True)
    
    # Optional: Hide mouse cursor for kiosk/touchscreen mode
    # root.config(cursor="none")
    
    # Escape exits kiosk mode cleanly.
    def exit_kiosk(event=None):
        root.overrideredirect(False)
        root.attributes('-topmost', False)
        root.attributes('-fullscreen', False)
        root.geometry(UI_CONFIG["window"]["size"])
        root.focus_force()
        return "break"

    root.bind("<Escape>", exit_kiosk)
    root.bind_all("<Escape>", exit_kiosk)

    # On some Linux/RPi window managers, overrideredirect windows may not
    # receive key events until focus is explicitly forced.
    root.after(50, root.focus_force)

    # Set to (Width : True, Height : True) to be resizable
    root.resizable(False, False)
    
    app = AnalyzerApp(root)
root.mainloop()
