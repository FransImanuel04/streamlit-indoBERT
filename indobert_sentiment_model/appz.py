import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Label mapping
id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load model dan tokenizer
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "./sentiment_model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "./sentiment_model"
    )

    return tokenizer, model

tokenizer, model = load_model()

# Title
st.title("Sentiment Analysis IndoBERT")
st.write("Analisis sentimen review Tokopedia")

# Input user
user_input = st.text_area(
    "Masukkan Review"
)

# Tombol prediksi
if st.button("Predict"):

    if user_input.strip() != "":

        # Tokenisasi
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Prediksi
        with torch.no_grad():

            outputs = model(**inputs)

            prediction = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

        # Hasil
        sentiment = id2label[prediction]

        st.success(
            f"Hasil Sentiment: {sentiment}"
        )

    else:
        st.warning(
            "Masukkan teks terlebih dahulu"
        )