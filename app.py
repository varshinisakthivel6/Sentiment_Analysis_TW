import streamlit as st
import pickle
import re
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# 1. Configuration ========================================================
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.fly.dev",
    "https://nitter.it"
]

# 2. Core Functions =======================================================
@st.cache_resource
def load_models():
    nltk.download('stopwords', quiet=True)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer, stopwords.words('english')

def verify_account(username):
    """Check if account exists and is public"""
    for instance in NITTER_INSTANCES:
        try:
            url = f"{instance}/{username}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 404:
                return "not_found"
            if "This account's tweets are protected" in response.text:
                return "private"
            if "This account doesn't exist" in response.text:
                return "not_found"
            return "public"
        except:
            continue
    return "error"

def fetch_tweets(username):
    """Fetch tweets with improved empty account handling"""
    scraper = Nitter()
    scraper.working_instances = NITTER_INSTANCES
    
    try:
        data = scraper.get_tweets(username, mode='user', number=3)
        
        if not data:
            return {"status": "error", "message": "No response from Nitter"}
            
        if 'tweets' not in data:
            return {"status": "error", "message": "Unexpected response format"}
            
        if not data['tweets']:
            # Verify account status when no tweets found
            account_status = verify_account(username)
            if account_status == "private":
                return {"status": "error", "message": "Account is private"}
            elif account_status == "not_found":
                return {"status": "error", "message": "Account not found"}
            else:
                return {"status": "empty", "message": "Account exists but has no visible tweets"}
                
        return {"status": "success", "tweets": [t['text'] for t in data['tweets']]}
        
    except Exception as e:
        return {"status": "error", "message": f"Nitter error: {str(e)}"}

def analyze_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = [w for w in text.split() if w not in stop_words]
    clean_text = ' '.join(words)
    prediction = model.predict(vectorizer.transform([clean_text]))[0]
    return "😊 Positive" if prediction == 1 else "😠 Negative"

# 3. UI Components ========================================================
def show_tweet(tweet, sentiment):
    color = "#4CAF50" if "😊" in sentiment else "#F44336"
    st.markdown(
        f"""<div style='background:{color};padding:10px;border-radius:5px;color:white;margin:10px 0'>
            <b>{sentiment}</b>: {tweet}
        </div>""",
        unsafe_allow_html=True
    )

def show_account_status(username, status):
    if status["status"] == "success":
        st.success(f"Found {len(status['tweets'])} tweets from @{username}")
        for tweet in status['tweets']:
            sentiment = analyze_sentiment(tweet, *load_models())
            show_tweet(tweet, sentiment)
            
    elif status["status"] == "empty":
        st.warning(f"Account @{username} exists but has no visible tweets")
        st.info(f"""
        Possible reasons:
        - Account may be new
        - Tweets may be protected
        - User may have deleted all tweets
        
        Check directly on: [Twitter/X](https://twitter.com/{username})
        """)
        
    else:
        st.error(f"Error: {status['message']}")
        if "private" in status["message"]:
            st.info(f"""
            This account is private. You can:
            - Request to follow on [Twitter/X](https://twitter.com/{username})
            - Try a different public account
            """)

# 4. Main App =============================================================
def main():
    st.title("Twitter/X Sentiment Analysis")
    model, vectorizer, stop_words = load_models()
    
    tab1, tab2 = st.tabs(["Analyze Text", "Check Account"])
    
    with tab1:
        text = st.text_area("Enter text to analyze:")
        if st.button("Analyze"):
            if text.strip():
                sentiment = analyze_sentiment(text, model, vectorizer, stop_words)
                show_tweet(text, sentiment)
    
    with tab2:
        username = st.text_input("Enter Twitter/X username (without @):")
        if st.button("Check Account"):
            if not username.strip():
                st.warning("Please enter a username")
            else:
                with st.spinner(f"Checking @{username}..."):
                    result = fetch_tweets(username)
                    show_account_status(username, result)

if __name__ == "__main__":
    main()