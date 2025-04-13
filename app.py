import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Setup
ps = PorterStemmer()
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer1.pkl','rb'))
mnb = pickle.load(open('model1.pkl','rb'))

# App Title and Styling
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        color:#4B8BBE;
        text-align:center;
        margin-bottom:30px;
    }
    .result-box {
        font-size:30px;
        font-weight:bold;
        text-align:center;
        padding:20px;
        border-radius:10px;
        margin-top:20px;
    }
    .spam {
        background-color:#ffcccc;
        color:#cc0000;
    }
    .ham {
        background-color:#ccffcc;
        color:#006600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📩 Email Spam Classifier</div>', unsafe_allow_html=True)

# Input Area
st.subheader("💬 Enter the message below:")
input_sms = st.text_area("", placeholder="Type or paste your message here...")

# Predict Button
if st.button('🔍 Predict') and input_sms.strip() != "":
    # Preprocess and predict
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = mnb.predict(vector_input)[0]

    if result == 1:
        st.markdown('<div class="result-box spam">🚫 This message is <strong>Spam</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box ham">✅ This message is <strong>Not Spam</strong></div>', unsafe_allow_html=True)
