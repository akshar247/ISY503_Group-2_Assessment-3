
import streamlit as stream
import numpy as np
import joblib 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(text):
    token = Tokenizer()
    joblib_model = joblib.load('model.pkl')
    encoded_sentence = token.texts_to_sequences([text])
    padded_sentence = pad_sequences(encoded_sentence, maxlen=20, padding='post', truncating='post')
    predicted_sentiment = joblib_model.predict(padded_sentence)
    return predicted_sentiment

def main():
    stream.title("Sentiment Analysis")
    input_text = stream.text_area("Enter Text")
    
    if stream.button("Submit"):
        prediction = predict_sentiment(input_text)
        stream.write("Prediction for Text:", "Positive" if prediction >0.49 else "Negative")

if __name__ == "__main__":
    main()
