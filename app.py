import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle

# --- SAFE NLTK INITIALIZATION ---
try:
    import nltk
    from nltk.corpus import stopwords
except ModuleNotFoundError:
    st.error("🚨 **Missing Dependency:** Please run `pip install nltk` in your terminal and restart the app.")
    st.stop()

# Auto-download stopwords if they aren't locally cached
@st.cache_resource
def initialize_nlp():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))

STOPWORDS = initialize_nlp()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cinematic Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM ELEGANT STYLING ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .genre-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 1.75rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-top: 1rem;
        text-align: center;
    }
    .genre-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #166534;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .genre-title {
        font-size: 1.1rem;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 0.5rem;
        width: 100%;
        font-weight: 600;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- TEXT CLEANING UTILITY ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

# --- ENGINE TRAINING AND BUNDLE LOADING ---
# Define paths to your project files
DATA_PATH = "Genre Classification Dataset/train_data.txt" 
MODEL_BUNDLE_PATH = "movie_pipeline_bundle.pkl"

@st.cache_resource
def load_or_train_factory():
    """
    Checks for a valid trained pipeline bundle. 
    If missing, fallback-trains a Logistic Regression classifier directly from the raw dataset.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    if os.path.exists(MODEL_BUNDLE_PATH):
        with open(MODEL_BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
            if isinstance(bundle, dict) and 'model' in bundle:
                return bundle

    # Fallback compilation loop if pickle is missing or broken
    if os.path.exists(DATA_PATH):
        with st.spinner("🔧 First-time setup: Training classification model from dataset structure..."):
            # Load raw training data assuming standard ::: delimiter format used in the competition dataset
            train_df = pd.read_csv(DATA_PATH, sep=":::", engine="python", names=["ID", "Title", "Genre", "Description"])
            
            train_df['Cleaned_Description'] = train_df['Description'].apply(clean_text)
            
            # Feature extraction engineering
            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(train_df['Cleaned_Description'])
            
            le = LabelEncoder()
            y = le.fit_transform(train_df['Genre'].str.strip())
            
            # Fit optimized classifier model
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            
            bundle = {
                'model': clf,
                'tfidf': tfidf,
                'encoder': le
            }
            
            # Cache pipeline bundle
            with open(MODEL_BUNDLE_PATH, "wb") as f:
                pickle.dump(bundle, f)
            return bundle
    return None

# Execute robust initialization pipeline
pipeline_bundle = load_or_train_factory()

# --- HEADER SECTION ---
st.title("🎬 Cinematic Genre Classifier")
st.markdown(
    "Welcome to the intelligent script analyzer. Input any narrative description, outline, "
    "or movie plot below, and our model will instantly categorize its cinematic archetype."
)
st.divider()

# --- APP INTERFACE ---
if pipeline_bundle is None:
    st.error(
        f"**Configuration Error:** Unable to find a saved pipeline bundle (`{MODEL_BUNDLE_PATH}`) "
        f"or raw dataset text file (`{DATA_PATH}`). Please ensure your dataset folder or bundle file "
        f"is present in the current working directory."
    )
else:
    # Safely unpack component parameters
    classifier = pipeline_bundle['model']
    tfidf_vectorizer = pipeline_bundle['tfidf']
    label_encoder = pipeline_bundle['encoder']

    # Layout structure split
    left_input_col, right_output_col = st.columns([3, 2])

    with left_input_col:
        st.subheader("📝 Plot / Outline Entry")
        movie_title = st.text_input("Movie Title (Optional)", placeholder="e.g., Interstellar")
        plot_description = st.text_area(
            "Plot Synopsis", 
            placeholder="Type or paste the story summary here...",
            height=220
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("✨ Analyze & Classify Genre")

    with right_output_col:
        st.subheader("🔮 Categorization Result")
        
        if analyze_clicked:
            if not plot_description.strip():
                st.warning("⚠️ Please provide a valid plot description text block before classifying.")
            else:
                try:
                    with st.spinner("Extracting textual features..."):
                        # Clean the text entry
                        processed_text = clean_text(plot_description)
                        
                        # Pipeline execution transformation
                        features = tfidf_vectorizer.transform([processed_text])
                        numeric_pred = classifier.predict(features)
                        genre_result = label_encoder.inverse_transform(numeric_pred)[0]
                    
                    display_title = f'"{movie_title}"' if movie_title.strip() else "Submitted Script"
                    st.markdown(f"""
                        <div class="genre-card">
                            <div class="genre-title">Predicted Genre for {display_title}</div>
                            <div class="genre-value">{genre_result}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("Analysis finalized successfully.")
                    
                except Exception as e:
                    st.error(f"Classification runtime exception. Detail log: {e}")
        else:
            st.info("👈 Supply a description and tap **Analyze & Classify Genre** to launch calculations.")