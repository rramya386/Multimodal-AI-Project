import streamlit as st
from PIL import Image
import base64
import os
from openai import OpenAI
import whisper

# Load OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

st.title("🧠 Multimodal Q&A Assistant")
st.write("Upload an image or audio, and ask natural-language questions.")

# Image section
st.subheader("📷 Image Upload + Question")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
image_question = st.text_input("Ask a question about the image:")

if uploaded_image and image_question:
    base64_image = encode_image_to_base64(uploaded_image)
    st.image(Image.open(uploaded_image), caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Send to OpenAI Vision model
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=300
    )

    st.success(response.choices[0].message.content)

# Audio section
st.subheader("🎧 Audio Upload")
uploaded_audio = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_audio:
    st.write("Transcribing...")
    model = whisper.load_model("base")
    audio_path = f"temp_audio.{uploaded_audio.type.split('/')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    result = model.transcribe(audio_path)
    st.write("Transcription:")
    st.success(result["text"])
