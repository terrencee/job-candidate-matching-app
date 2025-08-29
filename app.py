import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Cache model + tokenizer so they don't reload each time
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("recruitment_best_model")
    tokenizer = AutoTokenizer.from_pretrained("recruitment_best_model")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get a prediction score
def get_score(job_text, cand_text):
    combined = job_text + " [SEP] " + cand_text
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.squeeze().cpu().item()
    return score

# Normalize scores to 0–1
def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# Function to rank matches
def get_top_matches(query_text, corpus_texts, query_is_candidate=True, top_n=5):
    scores = []
    for doc in corpus_texts:
        if query_is_candidate:
            score = get_score(doc, query_text)  # job, candidate
        else:
            score = get_score(query_text, doc)  # job, candidate
        scores.append((doc, score))

    raw_scores = [s for _, s in scores]

    # --- NEW: threshold check ---
    if all(s < 30 for s in raw_scores):
        return None  # signal no proper match

    norm_scores = normalize_scores(raw_scores)
    normalized_results = [(doc, norm) for (doc, _), norm in zip(scores, norm_scores)]
    ranked = sorted(normalized_results, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# --- Input validation ---
def is_valid_input(text):
    # Require at least 7 words to consider valid
    return len(text.split()) >= 7

# --- Streamlit UI ---
st.title("🔍 Job ↔ Candidate Matching App")
st.write("Match candidates and jobs using a fine-tuned DistilBERT regression model.")

mode = st.radio("Choose a mode:", ["Candidate → Jobs", "Job → Candidates"])

if mode == "Candidate → Jobs":
    candidate_input = st.text_area("Enter candidate profile (skills, experience, etc.):")
    jobs_input = st.text_area("Enter job profiles (one per line):")
    top_n = st.slider("Number of matches", 1, 10, 5)

    if st.button("Find Best Jobs"):
        job_list = [j.strip() for j in jobs_input.split("\n") if j.strip()]
        
        if not is_valid_input(candidate_input):
            st.error("❌ Candidate profile is too short or invalid. Please provide a detailed description.")
        elif any(not is_valid_input(j) for j in job_list):
            st.error("❌ One or more job profiles are too short or invalid. Please provide detailed descriptions.")
        else:
            results = get_top_matches(candidate_input, job_list, query_is_candidate=True, top_n=top_n)
            if results is None:
                st.error("❌ No proper match found. Please provide more detailed inputs.")
            else:
                st.subheader("Best Job Matches")
                for i, (job, score) in enumerate(results, 1):
                    st.write(f"**{i}. {job}** (score: {score:.2f})")

else:
    job_input = st.text_area("Enter job description:")
    candidates_input = st.text_area("Enter candidate profiles (one per line):")
    top_n = st.slider("Number of matches", 1, 10, 5)

    if st.button("Find Best Candidates"):
        cand_list = [c.strip() for c in candidates_input.split("\n") if c.strip()]
        
        if not is_valid_input(job_input):
            st.error("❌ Job description is too short or invalid. Please provide a detailed description.")
        elif any(not is_valid_input(c) for c in cand_list):
            st.error("❌ One or more candidate profiles are too short or invalid. Please provide detailed descriptions.")
        else:
            results = get_top_matches(job_input, cand_list, query_is_candidate=False, top_n=top_n)
            if results is None:
                st.error("❌ No proper match found. Please provide more detailed inputs.")
            else:
                st.subheader("Best Candidate Matches")
                for i, (cand, score) in enumerate(results, 1):
                    st.write(f"**{i}. {cand}** (score: {score:.2f})")
