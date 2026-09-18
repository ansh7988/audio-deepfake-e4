E4 Audio Deepfake Detection Model
API Integration Guide — ML Model Handoff
Model: E4 Multi-Scale Raw-Waveform CNN
Checkpoint: best_e4.pt
Decision threshold: 0.44
1. Purpose
This repository contains the trained E4 model and its current inference pipeline. It is intended for the
software/backend team to build an API around the ML model. The model classifies audio as REAL or FAKE.
2. Repository Structure
audio-deepfake-e4/
n
nnn inference/
n nnn predict_e4_universal.py
n
nnn model/
nnn best_e4.pt
nnn e4_model.py
3. Files
File Purpose
model/e4_model.py Exact E4 neural-network architecture required to construct the model.
model/best_e4.pt Trained E4 model checkpoint/weights.
inference/predict_e4_universal.py Current audio preprocessing, 2-second chunking, E4 inference and result aggregation pipeline.
4. E4 Architecture
E4 is a Multi-Scale 1D Convolutional Neural Network operating directly on the raw audio waveform.
Raw waveform
↓
Initial Conv1D stem
↓
Multi-Scale Block 1
↓
Multi-Scale Block 2
↓
Multi-Scale Block 3
↓
Global Average Pooling
↓
Fully Connected Classifier
↓
REAL / FAKE
Each Multi-Scale Block contains three parallel temporal convolution branches with kernel sizes 5, 11 and 21. Their
outputs are concatenated and passed through a 1×1 projection convolution.
5. Input Processing
The model was trained on mono audio at 16,000 Hz with 2-second inputs (32,000 samples). The current universal
inference pipeline decodes the input, converts it to mono, resamples to 16 kHz, and processes it in 2-second
chunks.
Input audio
↓
Decode
↓
Mono
↓
16 kHz
↓
2-second chunks
↓
Waveform normalization
↓
E4 inference
6. Long Audio Handling
Audio longer than 2 seconds is divided into consecutive 2-second chunks. Each chunk receives a fake probability.
The current inference script averages the chunk probabilities to obtain the final fake probability.
7. Classification
The model produces a logit. A sigmoid converts it to a fake probability. The current locked threshold is 0.44.
Fake Probability >= 0.44 → FAKE
Fake Probability < 0.44 → REAL
The threshold was selected using the validation set and locked before the final test evaluation.
8. Recommended API Contract
The API team can expose a POST endpoint such as /predict. The backend should accept an uploaded audio file,
pass it through the ML inference pipeline, and return JSON. The exact API framework and deployment method are
up to the software team.
POST /predict
Request:
multipart/form-data
audio = <audio file>
Recommended response:
{
"prediction": "FAKE",
"fake_probability": 0.8732,
"threshold": 0.44,
"chunks_processed": 12,
"fake_chunks": 10,
"real_chunks": 2
}
9. Response Fields
Field Meaning
prediction Final REAL or FAKE classification.
fake_probability Average fake probability across processed chunks.
threshold Current decision threshold: 0.44.
chunks_processed Number of 2-second chunks processed.
fake_chunks Chunks with probability ≥ 0.44.
real_chunks Chunks with probability < 0.44.
10. API / ML Responsibilities
API/backend: receive audio, authenticate requests, validate files, call ML inference, handle errors, return JSON.
ML: decode/preprocess audio, load E4, run inference, aggregate chunk probabilities, produce REAL/FAKE.
11. Model Loading
from model.e4_model import E4MultiScaleCNN
model = E4MultiScaleCNN()
checkpoint = torch.load(
"model/best_e4.pt",
map_location=device
)
model.load_state_dict(
checkpoint["model_state_dict"]
)
model.eval()
12. Device Support
The model supports CUDA GPU when available and CPU inference otherwise. The API deployment environment
should install a compatible PyTorch/Torchaudio setup.
device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)
13. Current Baseline Evaluation
The current E4 baseline was evaluated on a balanced WaveFake-based test set of 600 samples (300 REAL and
300 FAKE).
Metric Result
ROC-AUC 0.8859
Accuracy 77.67%
Precision 70.85%
Recall 94.00%
F1 Score 80.80%
FAR 38.67%
FRR 6.00%
These are baseline results on the current development/test setup and should not be interpreted as universal performance across
all real-world audio sources.
14. Future Model Updates
The API contract should remain stable while the underlying checkpoint can be updated. Future versions may be
fine-tuned for WhatsApp compression, telephone audio, TTS, voice cloning, microphone recordings, re-recorded
audio, and background-dominant conditions.
Current:
best_e4.pt
Possible future versions:
best_e4_whatsapp.pt
best_e4_robust.pt
15. Security Note
If API-key authentication is implemented, the key must be stored as a server-side environment variable or secret.
Do not commit API keys, tokens, passwords, or other credentials to GitHub.
16. Handoff Note
This repository contains the ML model and inference components; it does not contain the API implementation. The
software/backend team can wrap the inference pipeline in FastAPI, Flask, or another suitable API framework.
