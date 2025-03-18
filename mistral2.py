import streamlit as st
import base64
import os
from mistralai import Mistral
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import io

# Load environment variables from a .env file
load_dotenv()

# Initialize the Mistral client with your API key
api_key = st.secrets['MISTRAL_API_KEY']
client = Mistral(api_key=api_key)

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to process OCR for images
def process_image_ocr(image, model):
    base64_image = encode_image(image)
    ocr_response = client.ocr.process(
        model=model,
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        }
    )
    return extract_ocr_text(ocr_response)

# Function to process OCR for PDFs
def process_pdf_ocr(pdf_file, model):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# Function to extract text from OCR response
def extract_ocr_text(ocr_response):
    ocr_text = ""
    for page in ocr_response.pages:
        ocr_text += page.markdown
    return ocr_text

# Function to process chat for images
def process_image_chat(image, model):
    base64_image = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

# Custom CSS for better UI
st.markdown("""
    <style>
    .stRadio > div {
        flex-direction: row;
        justify-content: space-around;
    }
    .stRadio > div[data-baseweb="radio"] > div {
        margin-right: 1rem;
    }
    .reportview-container .main .block-container{{
        max-width: 800px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: #1e1e1e;
        font-family: "Helvetica Neue", sans-serif;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Mistral OCR and Chat with PDF and Image Support")

# Select Mistral model using radio buttons
model_options = {
    # "Pixtral 12B (OCR)": "mistral-large-2411",
    "Mistral OCR": "mistral-ocr-latest",
    "Pixtral 12B ": "pixtral-12b-2409"
}
selected_model = st.radio("Select Mistral Model", options=list(model_options.keys()))

# File uploader for PDF or images
uploaded_files = st.file_uploader("Upload PDF or Images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Submit"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in [".jpg", ".jpeg", ".png"]:
                # Process image
                image = Image.open(uploaded_file)
                if "Chat" in selected_model:
                    extracted_text = process_image_chat(image, model_options[selected_model])
                else:
                    extracted_text = process_image_ocr(image, model_options[selected_model])
            elif file_extension == ".pdf":
                # Process PDF
                extracted_text = process_pdf_ocr(uploaded_file, model_options[selected_model])
            else:
                st.error("Unsupported file type")
                continue

            st.subheader(f"Extracted Text from {uploaded_file.name}")
            st.write(extracted_text)
    else:
        st.warning("Please upload a file.")
