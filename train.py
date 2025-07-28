import os
import json
import fitz
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
model_e5 = SentenceTransformer("intfloat/e5-small-v2")

DEBUG = True

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def is_heading_like(text):
    text = text.strip()
  
    if (
        bool(re.match(r"^\d+(\.\d+)*\s", text)) or
        bool(re.match(r"^[A-Z][A-Z\s]{3,}$", text)) or
        bool(re.match(r"^[A-Z][^a-z]{3,}$", text)) or
        len(text.split()) <= 12
    ) and len(text) <= 120:
        return True
    return False

def extract_sections(pdf):
    sections = []
    for page_num in range(len(pdf)):
        blocks = pdf[page_num].get_text("blocks")
       
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

        for i, block in enumerate(blocks_sorted):
            text = clean_text(block[4])
            if is_heading_like(text):
               
                context = ""
                for j in range(i+1, min(i+4, len(blocks_sorted))):
                    context += " " + clean_text(blocks_sorted[j][4])
                sections.append({
                    "text": text,
                    "context": context.strip(),
                    "page": page_num
                })
    return sections

def extract_doc_embedding(pdf):
    
    full_text = " ".join([clean_text(pdf[i].get_text()) for i in range(len(pdf))])
    emb = model_sbert.encode(full_text[:2000], convert_to_tensor=True)
    return emb, full_text

def extract_tf_idf_keywords(texts, top_k=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    vectorizer.fit(texts)
    return set(vectorizer.get_feature_names_out())

def compute_keyword_overlap(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

def dual_similarity_score(text, context, persona_sbert_emb, persona_e5_emb):
    combined = f"{text} {context}"
    sbert_emb = model_sbert.encode(combined, convert_to_tensor=True)
    e5_emb = model_e5.encode("query: " + combined, convert_to_tensor=True)
    sbert_score = util.cos_sim(persona_sbert_emb, sbert_emb).item()
    e5_score = util.cos_sim(persona_e5_emb, e5_emb).item()
    return 0.5 * sbert_score + 0.5 * e5_score

def rank_documents(docs, persona_sbert_emb, persona_e5_emb, input_dir):
    doc_scores = []
    full_texts = []
    for doc in docs:
        try:
            path = os.path.join(input_dir, doc["filename"])
            pdf = fitz.open(path)
            emb, full_text = extract_doc_embedding(pdf)
            sbert_score = util.cos_sim(persona_sbert_emb, emb).item()
            e5_emb = model_e5.encode("query: " + full_text[:2000], convert_to_tensor=True)
            e5_score = util.cos_sim(persona_e5_emb, e5_emb).item()
            combined_score = 0.5 * sbert_score + 0.5 * e5_score
            doc_scores.append((combined_score, doc["filename"]))
            full_texts.append(full_text)
            if DEBUG:
                print(f"[DEBUG] Document '{doc['filename']}': relevance={combined_score:.3f}")
        except Exception as e:
            print(f"[WARNING] Failed to process document {doc['filename']}: {str(e)}")
    return sorted(doc_scores, key=lambda x: x[0], reverse=True), full_texts

def rank_sections_by_relevance(sections, persona_sbert_emb, persona_e5_emb, tfidf_keywords):
    ranked = []
    seen = set()
    for item in sections:
        text = item["text"]
        norm_text = text.lower()
        if norm_text in seen:
            continue
        context = item.get("context", "")
        score = dual_similarity_score(text, context, persona_sbert_emb, persona_e5_emb)
        keyword_boost = 0.02 * compute_keyword_overlap(text + " " + context, tfidf_keywords)
        page_boost = 0.01 * (1 / (1 + item["page"])) 
        final_score = score + keyword_boost + page_boost
        ranked.append({
            "document": item["document"],
            "section_title": text,
            "page_number": item["page"] ,  
            "relevance_score": round(final_score, 4)
        })
        seen.add(norm_text)
    ranked = sorted(ranked, key=lambda x: x["relevance_score"], reverse=True)
    return ranked

def extract_contextual_paragraphs(pdf, page, persona_sbert_emb, persona_e5_emb):
    paragraphs = pdf[page].get_text("blocks")
    results = []
    for block in paragraphs:
        text = clean_text(block[4])
        if len(text.split()) < 6:
            continue
        para_sbert = model_sbert.encode(text, convert_to_tensor=True)
        para_e5 = model_e5.encode("passage: " + text, convert_to_tensor=True)
        sim_sbert = util.cos_sim(persona_sbert_emb, para_sbert).item()
        sim_e5 = util.cos_sim(persona_e5_emb, para_e5).item()
        sim = 0.5 * sim_sbert + 0.5 * sim_e5
        if sim > 0.4:
            results.append((sim, text))
    results.sort(reverse=True)
    return [r[1] for r in results[:3]]

def main():
    input_dir = "./round1b/PDFs"
    output_dir = "./round1b/output"
    os.makedirs(output_dir, exist_ok=True)

    with open("round1b/challenge1b_input2.json") as f:
        challenge_input = json.load(f)

    persona = challenge_input.get("persona", {}).get("role", "")
    job = challenge_input.get("job_to_be_done", {}).get("task", "")
    doc_info = challenge_input.get("documents", [])

    persona_job_text = f"You are a {persona}. Your job is to: {job}"
    persona_sbert_emb = model_sbert.encode(persona_job_text, convert_to_tensor=True)
    persona_e5_emb = model_e5.encode("query: " + persona_job_text, convert_to_tensor=True)

    ranked_docs, full_texts = rank_documents(doc_info, persona_sbert_emb, persona_e5_emb, input_dir)


    threshold = 0.15
    top_docs = [d for score, d in ranked_docs if score > threshold]
    if len(top_docs) < 5:
        top_docs = [d for _, d in ranked_docs[:12]]

    if DEBUG:
        print(f"[DEBUG] Selected top {len(top_docs)} documents for section extraction.")

    all_sections = []
    extracted_docs = []

    for filename in top_docs:
        try:
            filepath = os.path.join(input_dir, filename)
            pdf = fitz.open(filepath)
            sections = extract_sections(pdf)
            for s in sections:
                s["document"] = filename
                all_sections.append(s)
            if filename not in extracted_docs:
                extracted_docs.append(filename)
        except Exception as e:
            print(f"[WARNING] Skipping {filename}: {e}")

    if not all_sections:
        print("[WARNING] No valid sections extracted from documents.")

    tfidf_keywords = extract_tf_idf_keywords(full_texts)

    top_sections = rank_sections_by_relevance(all_sections, persona_sbert_emb, persona_e5_emb, tfidf_keywords)[:6]

    subanalysis = []
    for i, sec in enumerate(top_sections):
        doc_path = os.path.join(input_dir, sec["document"])
        pdf = fitz.open(doc_path)
        extracted = extract_contextual_paragraphs(pdf, sec["page_number"]-1, persona_sbert_emb, persona_e5_emb)
        for para in extracted:
            subanalysis.append({
                "document": sec["document"],
                "refined_text": para,
                "page_number": sec["page_number"]
            })
        sec["importance_rank"] = i + 1

    output_data = {
        "metadata": {
            "input_documents": extracted_docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": top_sections,
        "subsection_analysis": subanalysis
    }

    output_path = os.path.join(output_dir, "challenge1b_output_contextual.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Round 1B {output_path}")


if __name__ == "__main__":
    main()
