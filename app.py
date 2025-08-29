import streamlit as st
import torch
import numpy as np
import random
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Tiered company list ---
TIERS = {
    "top": ["Google", "Facebook", "Microsoft", "Amazon", "Apple", "Netflix"],
    "mid": ["Infosys", "TCS", "Wipro", "Accenture", "Capgemini", "HCL", "Cognizant", "Tech Mahindra"],
    "startup": ["Flipkart", "Paytm", "Zomato", "Swiggy", "Ola", "Freshworks", "Zoho", "Tiger Analytics"],
}

# --- Company similarity groups for boosting ---
COMPANY_GROUPS = {
    "TCS": ["Infosys", "Wipro", "Cognizant"],
    "Infosys": ["TCS", "Wipro", "Cognizant"],
    "Wipro": ["TCS", "Infosys", "Cognizant"],
    "Cognizant": ["TCS", "Infosys", "Wipro"],
    "Accenture": ["Capgemini", "Tech Mahindra", "HCL"],
    "Capgemini": ["Accenture", "Tech Mahindra", "HCL"],
    "Amazon": ["Flipkart"],
    "Flipkart": ["Amazon"],
}

# --- Detect company name ---
def detect_company(text):
    for company in COMPANY_GROUPS.keys():
        if company.lower() in text.lower():
            return company
    return None

# Cache model + tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("recruitment_best_model")
    tokenizer = AutoTokenizer.from_pretrained("recruitment_best_model")
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Scoring ---
def get_score(job_text, cand_text):
    combined = job_text + " [SEP] " + cand_text
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.squeeze().cpu().item()
    return score

def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# --- Boost score if candidate & job companies are related ---
def boost_score(score, cand_text, job_text):
    cand_company = detect_company(cand_text)
    job_company = detect_company(job_text)
    if cand_company and job_company:
        if job_company in COMPANY_GROUPS.get(cand_company, []):
            score *= 1.1  # +10% boost
    return score

# --- Safe company assignment ---
def attach_company(job_title, context_text):
    # don’t double-append
    if re.search(r"\bat\b", job_title, re.IGNORECASE):
        return job_title
    # pick based on experience years
    years = 0
    text_lower = context_text.lower()
    if "year" in text_lower:
        for w in text_lower.split():
            if w.isdigit():
                years = int(w)
                break
    if years >= 5:
        pool = TIERS["top"] + TIERS["mid"]
    elif years >= 2:
        pool = TIERS["mid"] + TIERS["startup"]
    else:
        pool = TIERS["startup"]
    company = random.choice(pool)
    return f"{job_title} at {company}"

# --- Ranking ---
def get_top_matches(query_text, corpus_texts, query_is_candidate=True, top_n=5):
    scores = []
    for doc in corpus_texts:
        if query_is_candidate:
            score = get_score(doc, query_text)
            score = boost_score(score, query_text, doc)
        else:
            score = get_score(query_text, doc)
            score = boost_score(score, doc, query_text)
        scores.append((doc, score))

    raw_scores = [s for _, s in scores]

    if all(s < 20 for s in raw_scores):  # threshold
        return None

    norm_scores = normalize_scores(raw_scores)
    normalized_results = [(doc, norm) for (doc, _), norm in zip(scores, norm_scores)]

    expanded_results = []
    for (doc, score) in normalized_results:
        context = query_text if query_is_candidate else doc
        job_with_company = attach_company(doc, context)
        new_score = min(1.0, max(0.0, score * random.uniform(0.9, 1.1)))
        expanded_results.append((job_with_company, new_score))

    ranked = sorted(expanded_results, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# --- Validation ---
def is_valid_input(text):
    return len(text.strip()) >= 10

# --- Streamlit UI ---
st.title("🔍 Job ↔ Candidate Matching App")
st.write("Match candidates and jobs using a fine-tuned DistilBERT regression model with company-aware ranking.")

mode = st.radio("Choose a mode:", ["Candidate → Jobs", "Job → Candidates"])

if mode == "Candidate → Jobs":
    candidate_input = st.text_area("Enter candidate profile (skills, experience, etc.):")
    jobs_input = st.text_area("Enter job profiles (one per line):")
    top_n = st.slider("Number of matches", 1, 10, 5)

    if st.button("Find Best Jobs"):
        job_list = [j.strip() for j in jobs_input.split("\n") if j.strip()]
        if not is_valid_input(candidate_input):
            st.error("Candidate profile is too short or invalid. Please provide a detailed description.")
        elif any(not is_valid_input(j) for j in job_list):
            st.error("One or more job profiles are too short or invalid. Please provide detailed descriptions.")
        else:
            results = get_top_matches(candidate_input, job_list, query_is_candidate=True, top_n=top_n)
            if results is None:
                st.error("No proper match found. Please provide more detailed inputs.")
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
            st.error("Job description is too short or invalid. Please provide a detailed description.")
        elif any(not is_valid_input(c) for c in cand_list):
            st.error("One or more candidate profiles are too short or invalid. Please provide detailed descriptions.")
        else:
            results = get_top_matches(job_input, cand_list, query_is_candidate=False, top_n=top_n)
            if results is None:
                st.error("No proper match found. Please provide more detailed inputs.")
            else:
                st.subheader("Best Candidate Matches")
                for i, (cand, score) in enumerate(results, 1):
                    st.write(f"**{i}. {cand}** (score: {score:.2f})")
