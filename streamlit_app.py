import streamlit as st
import numpy as np
import soundfile as sf
from io import BytesIO
from pathlib import Path
from pydub import AudioSegment
import resampy

# Real-Time Voice Cloning imports
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models

# Load models on startup
MODEL_DIR = Path("saved_models/default")
#ensure_default_models(MODEL_DIR.parent)

encoder.load_model(MODEL_DIR / "encoder.pt")
synthesizer = Synthesizer(MODEL_DIR / "synthesizer.pt")
vocoder.load_model(MODEL_DIR / "vocoder.pt")

# Streamlit UI
st.title("🎤 Real-Time Voice Cloning with Streamlit")

uploaded_audio = st.file_uploader("📁 Upload Reference Audio (wav, mp3, m4a, flac)", type=["wav", "mp3", "m4a", "flac"])
text_to_synthesize = st.text_area("💬 Enter Text to Synthesize:")

if st.button("🔊 Synthesize"):
    if uploaded_audio is None:
        st.error("❗ Please upload a reference audio file.")
    elif not text_to_synthesize.strip():
        st.error("❗ Please enter text to synthesize.")
    else:
        try:
            # Convert uploaded file to WAV in memory
            audio_bytes = uploaded_audio.read()
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            # Read audio with soundfile
            audio_np, sampling_rate = sf.read(wav_io)

            # Convert stereo to mono if necessary
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)

            # Check for very short audio
            if len(audio_np) < 1000:
                st.error("❗ Uploaded audio is too short. Please upload at least 1 second of audio.")
                st.stop()

            # Resample if sampling rate does not match encoder's requirement
            if sampling_rate != encoder.sampling_rate:
                audio_np = resampy.resample(audio_np, sampling_rate, encoder.sampling_rate)
                sampling_rate = encoder.sampling_rate

            # Preprocess and embed the voice
            preprocessed_wav = encoder.preprocess_wav(audio_np)
            embed = encoder.embed_utterance(preprocessed_wav)

            # Synthesize spectrogram from text and embedding
            specs = synthesizer.synthesize_spectrograms([text_to_synthesize], [embed])
            spec = specs[0]

            # Generate waveform using vocoder
            generated_wav = vocoder.infer_waveform(spec)

            # Post-process audio
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Save generated audio to BytesIO
            out_buffer = BytesIO()
            sf.write(out_buffer, generated_wav.astype(np.float32), synthesizer.sample_rate, format="WAV")
            out_buffer.seek(0)

            st.success("✅ Voice cloned successfully! Listen below:")
            st.audio(out_buffer.read(), format="audio/wav")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
