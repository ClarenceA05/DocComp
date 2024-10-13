import streamlit as st
import re
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import docx  # For reading .docx files
import pdfplumber  # For reading .pdf files
from PIL import Image
import io
from fuzzywuzzy import fuzz
import difflib  # For Differ comparison
import time  # To simulate timing and estimate processing duration

# Simple password-based authentication
def check_password():
    password = st.text_input("Enter the password", type="password")
    if password == "Cloud@Compare11#a":  # Replace with your secure password
        return True
    else:
        st.warning("Please enter a valid password")
        return False

if check_password():
    st.title("Clarence & Partners Document Preprocessing App")

# Load the provided logo
logo_path = "Logo_For white or light backgrounds.png"
logo_image = Image.open(logo_path)

# Set custom styling using colors from the logo
primary_color = "#1B75BC"
secondary_color = "#8AC640"
st.markdown(f"""
    <style>
    .reportview-container {{
        background: linear-gradient(90deg, {secondary_color} 30%, white 30%, white 70%, {secondary_color} 70%);
    }}
    .sidebar .sidebar-content {{
        background: {primary_color};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {primary_color};
    }}
    </style>
    """, unsafe_allow_html=True)

# Display the logo and title
st.image(logo_image, width=300)
st.title("Clarence & Partners Document Preprocessing App")

# Add introductory text and disclaimer
st.markdown("""
### This app is used to preprocess documents (using Python functions) in advance of seeking outputs from the ChatGPT Custom GPT from Clarence & Partners.

**Disclaimer**: No output provided by this tool should be treated as legal advice and users are encouraged to seek advice of specialist legal counsel.
""")

# Use caching to download NLTK data once
@st.cache_resource
def download_nltk_data():
    """Download necessary NLTK data."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Call the function to download NLTK data
download_nltk_data()

# Function to read DOCX files
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to read PDF files
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Enhanced text extraction function with encoding fallback handling
def read_file(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        # Read .txt file with multiple encoding fallback
        encodings = ['utf-8', 'ISO-8859-1', 'Windows-1252']
        for enc in encodings:
            try:
                content = uploaded_file.read().decode(enc, errors='replace')
                return content
            except UnicodeDecodeError:
                continue
        st.error(f"Error: Unable to decode the file using standard encodings. Please check the file format.")
        return None
    elif uploaded_file.name.endswith('.docx'):
        # Read .docx file
        content = read_docx(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        # Read .pdf file
        content = read_pdf(uploaded_file)
    else:
        st.error('Unsupported file type. Please upload a .txt, .docx, or .pdf file.')
        return None
    return content

# Function to normalize and preprocess text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to preprocess and tokenize text into sentences and words
@st.cache_data
def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word.lower() not in stop_words]
        processed_sentence = ' '.join(tokens)
        processed.append(processed_sentence)
    return sentences, processed

# Function to align sentences between documents using combined cosine similarity and fuzzy matching
def align_sentences_combined(sentences1, sentences2, threshold=0.75, cosine_weight=0.5, fuzzy_weight=0.5):
    # Vectorizer for cosine similarity
    vectorizer = lambda s: np.array([1 if w in s else 0 for w in set(" ".join(sentences1 + sentences2).split())])
    sentence_vectors1 = np.array([vectorizer(sentence) for sentence in sentences1])
    sentence_vectors2 = np.array([vectorizer(sentence) for sentence in sentences2])
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors1, sentence_vectors2)

    aligned_pairs = []
    for idx1, sentence1 in enumerate(sentences1):
        best_combined_score = 0
        best_match_sentence = None
        best_cosine_score = 0
        best_fuzzy_score = 0
        
        for idx2, sentence2 in enumerate(sentences2):
            # Calculate cosine similarity
            cosine_score = similarity_matrix[idx1][idx2]
            
            # Calculate fuzzy similarity (convert to a percentage scale)
            fuzzy_score = fuzz.ratio(sentence1, sentence2) / 100.0
            
            # Calculate the combined similarity score
            combined_score = (cosine_weight * cosine_score) + (fuzzy_weight * fuzzy_score)
            
            # Check if the current score is the best
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_match_sentence = sentence2
                best_cosine_score = cosine_score
                best_fuzzy_score = fuzzy_score

        # Check if the best combined score meets the threshold
        if best_combined_score >= threshold:
            aligned_pairs.append({
                'doc1_sentence': sentence1,
                'doc2_sentence': best_match_sentence,
                'combined_similarity_score': best_combined_score,
                'cosine_similarity_score': best_cosine_score,
                'fuzzy_similarity_score': best_fuzzy_score
            })
        else:
            aligned_pairs.append({
                'doc1_sentence': sentence1,
                'doc2_sentence': None,
                'combined_similarity_score': best_combined_score,
                'cosine_similarity_score': best_cosine_score,
                'fuzzy_similarity_score': best_fuzzy_score
            })
    return aligned_pairs

# Function to compare sentences using difflib's Differ
def compare_sentences_difflib(sentences1, sentences2):
    aligned_pairs = []
    differ = difflib.Differ()

    for idx1, sentence1 in enumerate(sentences1):
        if idx1 < len(sentences2):
            sentence2 = sentences2[idx1]
            diff = list(differ.compare(sentence1.split(), sentence2.split()))
            aligned_pairs.append({
                'doc1_sentence': sentence1,
                'doc2_sentence': sentence2,
                'diff': ' '.join(diff)
            })
        else:
            aligned_pairs.append({
                'doc1_sentence': sentence1,
                'doc2_sentence': None,
                'diff': "No matching sentence found."
            })

    return aligned_pairs

# Main Streamlit app function
def main():
    st.title("Advanced Document Comparison and Analysis Tool")

    st.write("This application allows you to compare two documents (TXT, DOCX, or PDF) and provides detailed sentence alignment and similarity analysis.")

    doc1_file = st.file_uploader("Upload Document 1", type=['txt', 'docx', 'pdf'])
    doc2_file = st.file_uploader("Upload Document 2", type=['txt', 'docx', 'pdf'])

    # Similarity threshold slider (between 0 and 1)
    threshold = st.slider('Similarity Threshold', min_value=0.0, max_value=1.0, value=0.75)
    
    # Add sliders to adjust the weights for cosine and fuzzy similarity
    cosine_weight = st.slider('Cosine Similarity Weight', min_value=0.0, max_value=1.0, value=0.5)
    fuzzy_weight = st.slider('Fuzzy Matching Weight', min_value=0.0, max_value=1.0, value=0.5)

    # Choose comparison method: Combined Cosine + Fuzzy or Differ
    comparison_method = st.selectbox("Choose Comparison Method", ["Cosine + Fuzzy Matching", "Difflib Differ"])

    if st.button("Compare Documents"):
        if doc1_file and doc2_file:
            try:
                doc1_content = read_file(doc1_file)
                doc2_content = read_file(doc2_file)

                if doc1_content and doc2_content:
                    doc1 = normalize_text(doc1_content)
                    doc2 = normalize_text(doc2_content)

                    # Initialize a progress bar
                    progress = st.progress(0)
                    progress.update(10)

                    with st.spinner('Preprocessing and analyzing documents...'):
                        # Simulate a processing time (to be removed in real cases)
                        time.sleep(1)

                        sentences1, processed_sentences1 = preprocess_sentences(doc1)
                        sentences2, processed_sentences2 = preprocess_sentences(doc2)

                        progress.update(40)  # Progress update after preprocessing

                    if comparison_method == "Cosine + Fuzzy Matching":
                        # Use the combined similarity function
                        aligned_sentences = align_sentences_combined(
                            sentences1, sentences2, threshold, cosine_weight, fuzzy_weight
                        )
                        st.success('Documents have been compared and aligned.')

                        progress.update(80)  # Progress update before displaying results

                        # Display the aligned sentences
                        st.subheader("Aligned Sentences")
                        output_text = []
                        for idx, pair in enumerate(aligned_sentences, start=1):
                            st.markdown(f"**Comparison {idx}:**")
                            st.write(f"**Sentence from Document 1:** {pair['doc1_sentence']}")
                            if pair.get('doc2_sentence'):
                                st.write(f"**Sentence from Document 2:** {pair['doc2_sentence']}")
                                output_text.append(f"Comparison {idx}:\nSentence from Document 1: {pair['doc1_sentence']}\nSentence from Document 2: {pair['doc2_sentence']}\nCombined Similarity Score: {pair['combined_similarity_score']:.2f}\nCosine Similarity Score: {pair['cosine_similarity_score']:.2f}\nFuzzy Similarity Score: {pair['fuzzy_similarity_score']:.2f}\n")
                            else:
                                st.write("**Sentence from Document 2:** *No corresponding sentence found*")
                                output_text.append(f"Comparison {idx}:\nSentence from Document 1: {pair['doc1_sentence']}\nSentence from Document 2: None\nCombined Similarity Score: {pair['combined_similarity_score']:.2f}\nCosine Similarity Score: {pair['cosine_similarity_score']:.2f}\nFuzzy Similarity Score: {pair['fuzzy_similarity_score']:.2f}\n")
                            st.write(f"**Combined Similarity Score:** {pair['combined_similarity_score']:.2f}")
                            st.write(f"**Cosine Similarity Score:** {pair['cosine_similarity_score']:.2f}")
                            st.write(f"**Fuzzy Similarity Score:** {pair['fuzzy_similarity_score']:.2f}")
                            st.write("---")

                    elif comparison_method == "Difflib Differ":
                        # Use the difflib Differ comparison
                        aligned_sentences = compare_sentences_difflib(sentences1, sentences2)
                        st.success('Documents have been compared using difflib Differ.')

                        progress.update(80)  # Progress update before displaying results

                        # Display the diff output
                        st.subheader("Differences in Sentences")
                        output_text = []
                        for idx, pair in enumerate(aligned_sentences, start=1):
                            st.markdown(f"**Comparison {idx}:**")
                            st.write(f"**Sentence from Document 1:** {pair['doc1_sentence']}")
                            if pair.get('doc2_sentence'):
                                st.write(f"**Sentence from Document 2:** {pair['doc2_sentence']}")
                                st.write(f"**Diff Output:** {pair['diff']}")
                                output_text.append(f"Comparison {idx}:\nSentence from Document 1: {pair['doc1_sentence']}\nSentence from Document 2: {pair['doc2_sentence']}\nDiff Output: {pair['diff']}\n")
                            else:
                                st.write("**Sentence from Document 2:** *No corresponding sentence found*")
                                output_text.append(f"Comparison {idx}:\nSentence from Document 1: {pair['doc1_sentence']}\nSentence from Document 2: None\nDiff Output: No corresponding sentence found\n")
                            st.write("---")

                    # Combine results into a downloadable text file
                    final_output = "\n".join(output_text)
                    st.download_button(
                        label="Download Comparison Results",
                        data=final_output,
                        file_name='document_comparison_results.txt',
                        mime='text/plain'
                    )

                    progress.update(100)  # Progress completed

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please upload both documents.")

if __name__ == "__main__":
    main()
