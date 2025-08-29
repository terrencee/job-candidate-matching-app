import streamlit as st
import torch
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
    # Same format you trained on: concatenate job + candidate
    combined = job_text + " [SEP] " + cand_text
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.squeeze().cpu().item()
    return score

# Function to rank matches
def get_top_matches(query_text, corpus_texts, query_is_candidate=True, top_n=5):
    scores = []
    for doc in corpus_texts:
        if query_is_candidate:
            score = get_score(doc, query_text)  # job, candidate
        else:
            score = get_score(query_text, doc)  # job, candidate
        scores.append((doc, score))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

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
        results = get_top_matches(candidate_input, job_list, query_is_candidate=True, top_n=top_n)

        st.subheader("Best Job Matches")
        for i, (job, score) in enumerate(results, 1):
            st.write(f"**{i}. {job}** (score: {score:.4f})")

else:
    job_input = st.text_area("Enter job description:")
    candidates_input = st.text_area("Enter candidate profiles (one per line):")
    top_n = st.slider("Number of matches", 1, 10, 5)

    if st.button("Find Best Candidates"):
        cand_list = [c.strip() for c in candidates_input.split("\n") if c.strip()]
        results = get_top_matches(job_input, cand_list, query_is_candidate=False, top_n=top_n)

        st.subheader("Best Candidate Matches")
        for i, (cand, score) in enumerate(results, 1):
            st.write(f"**{i}. {cand}** (score: {score:.4f})")
