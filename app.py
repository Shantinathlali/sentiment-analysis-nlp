import os
import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# 1. BASE DIRECTORY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 2. NLTK RESOURCES

# Make sure required NLTK resources are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test", "v")
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()


# 3. TEXT PREPROCESSING
#    Same preprocessing used during model training

def remove_stopword(text):
    tokens = nltk.word_tokenize(text)
    new_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(new_tokens)


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def lemmatize(text):
    words = text.split()
    lemmatized_words = []

    for word in words:
        lemmatized_words.append(
            lemmatizer.lemmatize(word, 'v')
        )

    return ' '.join(lemmatized_words)


def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove stopwords
    text = remove_stopword(text)

    # 3. Remove special characters
    text = clean_text(text)

    # 4. Lemmatize
    text = lemmatize(text)

    return text


# 4. LOAD MODEL AND TF-IDF

@st.cache_resource
def load_assets():

    model_path = os.path.join(
        BASE_DIR,
        'logistic_sentiment_model.pkl'
    )

    vectorizer_path = os.path.join(
        BASE_DIR,
        'tfidf_vectorizer.pkl'
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


model, vectorizer = load_assets()


# 5. LABEL MAPPING
#    LabelEncoder:
#    0 = Negative
#    1 = Neutral
#    2 = Positive

label_map = {
    0: 'Negative 😞',
    1: 'Neutral 😐',
    2: 'Positive 😊'
}


# 6. STREAMLIT UI

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("Sentiment Analysis App")
st.subheader("Logistic Regression Model")

st.write(
    "Enter a customer review below to predict its sentiment."
)


# 7. USER INPUT

user_input = st.text_area(
    "Enter Review:",
    placeholder="Example: This product is excellent and I really love it!",
    height=150
)


# 8. PREDICTION

if st.button("Analyze Sentiment"):

    if user_input.strip():

        # Preprocess input

        clean_input = preprocess_text(user_input)

        # Convert to TF-IDF

        vectorized_text = vectorizer.transform(
            [clean_input]
        )

        # Predict sentiment

        prediction = model.predict(
            vectorized_text
        )[0]

        sentiment_label = label_map.get(
            prediction,
            "Unknown"
        )

        # Display result

        st.subheader("Result")

        if prediction == 2:
            st.success(
                f"Predicted Sentiment: **{sentiment_label}**"
            )

        elif prediction == 0:
            st.error(
                f"Predicted Sentiment: **{sentiment_label}**"
            )

        else:
            st.info(
                f"Predicted Sentiment: **{sentiment_label}**"
            )

        # Confidence scores

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(
                vectorized_text
            )[0]

            st.write("---")
            st.write("### Confidence Scores")

            st.write(
                f"Negative: **{probs[0]:.2%}**"
            )

            st.write(
                f"Neutral: **{probs[1]:.2%}**"
            )

            st.write(
                f"Positive: **{probs[2]:.2%}**"
            )

    else:

        st.warning(
            "Please enter a review before analyzing."
        )