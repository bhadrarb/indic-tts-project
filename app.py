import re, time, io
import streamlit as st
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import os
from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

MODEL_ID = "ai4bharat/indic-parler-tts"
MAX_CHARS = 500
VOICES = {"Malayalam": ["Anjali", "Anju", "Harish"],
          "English": ["Thoma", "Mary", "Swapna"]}

@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, tok, desc_tok, device

def validate_text(text, language):
    text = text.strip()
    if not text: return False, "Please enter some text."
    if len(text) > MAX_CHARS: return False, f"Max {MAX_CHARS} characters."
    has_ml = bool(re.search(r"[\u0D00-\u0D7F]", text))
    if language == "Malayalam" and not has_ml: return False, "Enter Malayalam text."
    if language == "English" and has_ml: return False, "Text is Malayalam; change language."
    return True, ""

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[<>{}\[\]\\|^~`]", "", text)
    if text and text[-1] not in ".?!।": text += "."
    return text

st.title("Multilingual Text-to-Speech (Indic Parler-TTS)")
language = st.selectbox("Language", list(VOICES.keys()))
speaker = st.selectbox("Voice", VOICES[language])
pace = st.selectbox("Pace", ["slow", "moderate", "fast"], index=1)
text = st.text_area("Enter text", height=150)

if st.button("Generate Speech"):
    ok, msg = validate_text(text, language)
    if not ok:
        st.error(msg)
    else:
        with st.spinner("Loading model / generating..."):
            model, tok, desc_tok, device = load_model()
            clean = preprocess_text(text)
            desc = (f"{speaker} speaks with a {pace} pace and a clear tone "
                    "in a very close-sounding environment with no background noise.")
            d = desc_tok(desc, return_tensors="pt").to(device)
            p = tok(clean, return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                     prompt_input_ids=p.input_ids,
                                     prompt_attention_mask=p.attention_mask)
            secs = time.time() - t0
            audio = gen.cpu().numpy().squeeze()
            buf = io.BytesIO()
            sf.write(buf, audio, model.config.sampling_rate, format="WAV")
            wav_bytes = buf.getvalue()
        st.success(f"Generated in {secs:.1f} seconds")
        st.audio(wav_bytes, format="audio/wav")
        st.download_button("Download audio", wav_bytes, "speech.wav", "audio/wav")
