import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)


# text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    #include stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    # used stemming using porter stemmer
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
# importing both the pickle files that we used to train model and vectorizer as tfidf and naive bayes

tfidf = pickle.load(open('vectorizer1.pkl','rb'))
mnb = pickle.load(open('model1.pkl','rb'))

#title of the app
st.title("Email/SMS Spam Classifier")

#input area
input_sms = st.text_area("Enter the message")

#predict button
if st.button('Predict'):

    # preprocess the sms
    transformed_sms = transform_text(input_sms)
    # vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = mnb.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
