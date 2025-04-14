import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Styling the page
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Download NLTK resources
ps = PorterStemmer()
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
mnb = pickle.load(open('model1.pkl', 'rb'))

# App UI design
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📨 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("#### Paste any message below to check if it's **SPAM** or **NOT SPAM** 👇")

input_sms = st.text_area("🔤 Enter the message here:", height=150)

# Button and result display
if st.button('🚀 Predict', use_container_width=True):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = mnb.predict(vector_input)[0]

    if result == 1:
        st.error("🚫 This message is classified as **SPAM**.")
    else:
        st.success("✅ This message is classified as **NOT SPAM**.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with 💻 by <b>Prabal</b></p>", unsafe_allow_html=True)
