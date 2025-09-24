# app.py

import streamlit as st
import spacy
import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Initial Setup ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))

# --- Functions for Processing ---

def preprocess_text(text):
    if not text:
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    filtered_tokens = [word for word in tokens if word not in nltk_stopwords]
    return " ".join(filtered_tokens)

def read_pdf(file):
    if file is None:
        return None
    try:
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            text = " ".join(pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- Streamlit UI ---

st.title("AI Resume Ranker 🤖")
st.write("Check your resume against a job description. Use either text or PDF files!")

# Choose input method
input_method = st.radio("Choose your input method:", ("Paste Text", "Upload PDF"))

resume_content = None
job_desc_content = None

# Conditional UI based on input method
if input_method == "Paste Text":
    resume_content = st.text_area("Paste Your Resume Here")
    job_desc_content = st.text_area("Paste Job Description Here")
else:
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    
    if resume_file:
        resume_content = read_pdf(resume_file)
    if job_desc_file:
        job_desc_content = read_pdf(job_desc_file)

# --- Main Logic ---

if st.button("Calculate Match"):
    if resume_content and job_desc_content:
        processed_resume = preprocess_text(resume_content)
        processed_job_desc = preprocess_text(job_desc_content)
        
        if processed_resume and processed_job_desc:
            vectorizer = TfidfVectorizer()
            documents = [processed_resume, processed_job_desc]
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_percentage = round(cosine_sim[0][0] * 100, 2)
            
            st.success(f"Your resume matches the job description by *{match_percentage}%*! 🎉")
            
            st.subheader("Tips to Improve Your Resume")
            if match_percentage < 50:
                st.warning("Your match is low. Try to incorporate more keywords and skills directly from the job description.")
            elif 50 <= match_percentage < 75:
                st.info("Good match! Highlight your experience and skills that directly align with the job requirements.")
            else:
                st.info("Excellent match! You are a great fit.")
        else:
            st.error("Could not process text. Please ensure the content is valid.")
    else:
        st.warning("Please provide both a resume and a job description to proceed.")