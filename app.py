import streamlit as st
from data_clean import data_cleaning
from keras.utils import pad_sequences
import pickle
import keras

# Load the pickel file
with open('tokenizer.pickle','rb') as f:
    load_tokenizer = pickle.load(f)

# Load the model file
load_model=keras.models.load_model("model.h5")

st.title("Harmony Classifier: Hate Speech Classifier")

input_tweet = st.text_area('Enter the Tweet')

if st.button('predict'):
    
    #   1. Preprocess
    clean_text = [data_cleaning(input_tweet)]

    #   2. Vectorize
    vector_input = load_tokenizer.texts_to_sequences(clean_text)
    padded = pad_sequences(vector_input, maxlen=300)

    #   3. Predict
    pred = load_model.predict(padded)
    
    #   4. Display
    if (pred<0.5).any():
        st.write("No Hate")
    else:
        st.write("Hate and Abusive")
    
    st.write(pred)
