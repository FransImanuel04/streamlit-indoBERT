import streamlit as st
import os

st.write("Current directory:")
st.write(os.getcwd())

st.write("Files in current directory:")
st.write(os.listdir())

st.write("Files inside sentiment_model:")

if os.path.exists("sentiment_model"):
    st.write(os.listdir("sentiment_model"))
else:
    st.write("sentiment_model folder NOT FOUND")