import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from e4_model import E4MultiScaleCNN


# ============================================================
# CONFIG
# ============================================================

TARGET_SR = 16000
CHUNK_SECONDS = 2
TARGET_LEN = TARGET_SR * CHUNK_SECONDS

# Locked using validation set
THRESHOLD = 0.44

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = MODEL_DIR / "best_e4.pt"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("[1/3] Loading E4 model...")

    model = E4MultiScaleCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"      Device : {DEVICE}")
    print(f"      Model  : {MODEL_PATH.name}")

    return model


# ============================================================
# LOAD + CONVERT AUDIO
# ============================================================

def load_audio(path):

    print("[2/3] Loading audio...")

    audio = AudioSegment.from_file(path)

    print(f"      Original format : {Path(path).suffix}")
    print(f"      Duration        : {len(audio) / 1000:.2f} sec")
    print(f"      Channels        : {audio.channels}")
    print(f"      Sample rate     : {audio.frame_rate} Hz")

    # Standardize audio
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(TARGET_SR)

    # Temporary WAV for torchaudio
    temp_wav = PROJECT_ROOT / "temp_inference.wav"

    try:
        audio.export(temp_wav, format="wav")
        waveform, sr = torchaudio.load(temp_wav)
    finally:
        if temp_wav.exists():
            temp_wav.unlink()

    return waveform, sr


# ============================================================
# PREPROCESS ONE CHUNK
# ============================================================

def preprocess_audio(waveform):

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # Exactly 2 seconds
    if waveform.shape[1] > TARGET_LEN:

        waveform = waveform[:, :TARGET_LEN]

    elif waveform.shape[1] < TARGET_LEN:

        waveform = F.pad(
            waveform,
            (0, TARGET_LEN - waveform.shape[1])
        )

    # Same normalization used during E4 training
    max_val = waveform.abs().max()

    if max_val > 0:
        waveform = waveform / max_val

    return waveform


# ============================================================
# PREDICT ONE CHUNK
# ============================================================

def predict_chunk(model, waveform):

    waveform = preprocess_audio(waveform)

    # [1, 32000] -> [1, 1, 32000]
    waveform = waveform.unsqueeze(0)

    waveform = waveform.to(DEVICE)

    with torch.no_grad():

        logits = model(waveform)

        probability = torch.sigmoid(logits).item()

    return probability


# ============================================================
# PREDICT COMPLETE FILE
# ============================================================

def predict_file(path, model):

    waveform, sr = load_audio(path)

    total_samples = waveform.shape[1]
    chunk_size = TARGET_LEN

    probabilities = []

    print("[3/3] Running E4 predictions...")

    for start in range(0, total_samples, chunk_size):

        chunk = waveform[
            :,
            start:start + chunk_size
        ]

        probability = predict_chunk(
            model,
            chunk
        )

        probabilities.append(probability)

    if not probabilities:
        raise RuntimeError(
            "No audio chunks were generated."
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    fake_count = sum(
        p >= THRESHOLD
        for p in probabilities
    )

    real_count = len(probabilities) - fake_count

    average_probability = (
        sum(probabilities) / len(probabilities)
    )

    fake_ratio = (
        fake_count / len(probabilities)
    )

    final_prediction = (
        "FAKE"
        if average_probability >= THRESHOLD
        else "REAL"
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("       VYOM-AI AUDIO DEEPFAKE DETECTION")
    print("=" * 60)

    print(f"File               : {Path(path).name}")
    print(f"Duration           : {total_samples / sr:.2f} sec")
    print(f"Total 2-sec chunks : {len(probabilities)}")

    print()
    print(f"REAL chunks        : {real_count}")
    print(f"FAKE chunks        : {fake_count}")

    print()
    print(
        f"Average Fake Prob. : "
        f"{average_probability:.4f}"
    )

    print(
        f"Fake Chunk Ratio   : "
        f"{fake_ratio:.2%}"
    )

    print()
    print(f"Threshold          : {THRESHOLD:.2f}")
    print(
        f"FINAL PREDICTION   : "
        f"{final_prediction}"
    )

    print("=" * 60)

    return {
        "prediction": final_prediction,
        "fake_probability": average_probability,
        "threshold": THRESHOLD,
        "chunks_processed": len(probabilities),
        "fake_chunks": fake_count,
        "real_chunks": real_count,
    }


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            'python inference/predict_e4_universal.py '
            '"audio_file"'
        )
        print()

        sys.exit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.exists():

        print(
            f"\nERROR: File not found:\n"
            f"{audio_path}"
        )

        sys.exit(1)

    model = load_model()

    predict_file(
        audio_path,
        model
    )
