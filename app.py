import os
from io import BytesIO
from pathlib import Path
from flask import Flask, request, send_file, jsonify, send_from_directory
import numpy as np
import torch
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models

app = Flask(__name__)

# Load models on startup
MODEL_DIR = Path("saved_models/default")
ensure_default_models(MODEL_DIR.parent)

encoder.load_model(MODEL_DIR / "encoder.pt")
synthesizer = Synthesizer(MODEL_DIR / "synthesizer.pt")
vocoder.load_model(MODEL_DIR / "vocoder.pt")

@app.route("/")
def index():
    return send_from_directory('frontend', 'index.html')

@app.route("/synthesize", methods=["POST"])
def synthesize():
    if "reference_audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing reference_audio file or text parameter"}), 400

    ref_audio_file = request.files["reference_audio"]
    text = request.form["text"]

    # Save uploaded audio to a temporary buffer
    audio_bytes = ref_audio_file.read()
    audio_np, sampling_rate = sf.read(BytesIO(audio_bytes))
    if sampling_rate != encoder.sampling_rate:
        return jsonify({"error": f"Reference audio must have sampling rate {encoder.sampling_rate}"}), 400

    # Preprocess and embed reference audio
    preprocessed_wav = encoder.preprocess_wav(audio_np)
    embed = encoder.embed_utterance(preprocessed_wav)

    # Synthesize spectrogram from text and embedding
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    spec = specs[0]

    # Generate waveform from spectrogram
    generated_wav = vocoder.infer_waveform(spec)

    # Post-process waveform
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Save to buffer as WAV
    out_buffer = BytesIO()
    sf.write(out_buffer, generated_wav.astype(np.float32), synthesizer.sample_rate, format="WAV")
    out_buffer.seek(0)

    return send_file(out_buffer, mimetype="audio/wav", as_attachment=True, download_name="synthesized.wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
