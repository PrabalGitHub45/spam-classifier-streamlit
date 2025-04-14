import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Setup for nltk
ps = PorterStemmer()
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download resources only if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
mnb = pickle.load(open('model1.pkl', 'rb'))

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.markdown(
    """
    <h2 style='text-align: center; color: #6c63ff;'>📩 SMS Spam Classifier</h2>
    """,
    unsafe_allow_html=True
)

with st.form("spam_form"):
    input_sms = st.text_area("Enter the message", height=150)
    submitted = st.form_submit_button("Predict")

if submitted:
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = mnb.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div style='background-color:#ffcccc; padding:20px; border-radius:10px;'>"
            "<h3 style='color:red;'>🚨 This is Spam!</h3>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#ccffcc; padding:20px; border-radius:10px;'>"
            "<h3 style='color:green;'>✅ This is Not Spam!</h3>"
            "</div>",
            unsafe_allow_html=True
        )

st.markdown("<hr style='margin-top: 50px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>💻 Made with Prabal</p>", unsafe_allow_html=True)
