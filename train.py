import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
import fitz
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer


model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
model_e5 = SentenceTransformer("intfloat/e5-small")

DEBUG = True  


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def is_heading_like(text):
    text = text.strip()
    if (re.match(r"^\d+(\.\d+)*\s", text) or
            re.match(r"^[A-Z][A-Z\s]{3,}$", text) or
            re.match(r"^[A-Z][^a-z]{3,}$", text) or
            len(text.split()) <= 12) and len(text) <= 120:
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
                for j in range(i + 1, min(i + 4, len(blocks_sorted))):
                    context += " " + clean_text(blocks_sorted[j][4])
                sections.append({
                    "text": text,
                    "context": context.strip(),
                    "page": page_num,
                })
    return sections


def extract_doc_embedding(pdf):
    
    full_text = " ".join(clean_text(pdf[i].get_text()) for i in range(len(pdf)))
    emb = model_sbert.encode(full_text[:2000], convert_to_tensor=True)
    return emb, full_text


def extract_tf_idf_keywords(texts, top_k=20):
    if not texts:
        return set()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    vectorizer.fit(texts)
    return set(vectorizer.get_feature_names_out())


def compute_keyword_overlap(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)


def dual_similarity_score(text, context, persona_sbert_emb, persona_e5_emb):
    combined = f"{text} {context}"
    emb_sbert = model_sbert.encode(combined, convert_to_tensor=True)
    emb_e5 = model_e5.encode("query: " + combined, convert_to_tensor=True)
    score_sbert = util.cos_sim(persona_sbert_emb, emb_sbert).item()
    score_e5 = util.cos_sim(persona_e5_emb, emb_e5).item()
    return 0.5 * score_sbert + 0.5 * score_e5


def rank_documents(docs, persona_sbert_emb, persona_e5_emb, input_dir):
    doc_scores = []
    full_texts = []
    for doc in docs:
       
        filename = doc if isinstance(doc, str) else (doc.get('filename') or doc.get('file'))

        pdf_path = Path(input_dir) / filename
        if not pdf_path.exists():
            print(f"[WARNING] Document {filename} not found in input directory")
            continue

        try:
            pdf = fitz.open(pdf_path)
            emb, full_text = extract_doc_embedding(pdf)
            score_sbert = util.cos_sim(persona_sbert_emb, emb).item()
            emb_e5 = model_e5.encode(full_text[:2000], convert_to_tensor=True)
            score_e5 = util.cos_sim(persona_e5_emb, emb_e5).item()
            combined_score = 0.5 * score_sbert + 0.5 * score_e5
            doc_scores.append((combined_score, filename))
            full_texts.append(full_text)
            if DEBUG:
                print(f"[DEBUG] '{filename}' relevance score: {combined_score:.4f}")
        except Exception as e:
            print(f"[WARNING] Error processing {filename}: {e}")

    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[0], reverse=True)
    return doc_scores_sorted, full_texts


def rank_sections(sections, persona_sbert_emb, persona_e5_emb, keyword_set):
    ranked = []
    seen = set()
    for s in sections:
        norm_text = s['text'].lower()
        if norm_text in seen:
            continue
        score = dual_similarity_score(s['text'], s.get('context', ''), persona_sbert_emb, persona_e5_emb)
        keyword_boost = 0.02 * compute_keyword_overlap(s['text'] + ' ' + s.get('context', ''), keyword_set)
        page_boost = 0.01 * (1 / (1 + s['page']))
        combined_score = score + keyword_boost + page_boost
        ranked.append({
            'document': s.get('document', ''),
            'section_title': s['text'],
            'page_number': s['page'] + 1,
            'relevance_score': round(combined_score, 4),
        })
        seen.add(norm_text)
    ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
    return ranked


def extract_relevant_paragraphs(pdf, page_idx, persona_sbert_emb, persona_e5_emb):
    blocks = pdf[page_idx].get_text('blocks')
    candidates = []
    for block in blocks:
        text = clean_text(block[4])
        if len(text.split()) < 6:
            continue
        emb_sbert = model_sbert.encode(text, convert_to_tensor=True)
        emb_e5 = model_e5.encode('passage: ' + text, convert_to_tensor=True)
        score_sbert = util.cos_sim(persona_sbert_emb, emb_sbert).item()
        score_e5 = util.cos_sim(persona_e5_emb, emb_e5).item()
        combined_score = 0.5 * score_sbert + 0.5 * score_e5
        if combined_score > 0.4:
            candidates.append((combined_score, text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in candidates[:3]]


def main():
    parser = argparse.ArgumentParser(description="Run Round 2 Adobe Hackathon solution")
    parser.add_argument('--input_json', required=True, help='Path to the input JSON file')
    parser.add_argument('--pdf_dir', required=True, help='Path to the directory containing PDF files')
    parser.add_argument('--output_json', required=True, help='Path to write the output JSON')
    args = parser.parse_args()

    input_json_path = args.input_json
    pdf_dir = args.pdf_dir
    output_json_path = args.output_json

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(input_json_path, 'r') as f:
        data = json.load(f)

    persona = data.get('persona', {}).get('role', '')
    job = data.get('job', '') or data.get('job_to_be_done', '') or (data.get('job_to_be_done', {}).get('task'))
    if isinstance(job, dict):
        job = job.get('task', '')
    documents = data.get('documents', [])

    persona_job_text = f"You are a {persona}. You need to: {job}"
    persona_emb_sbert = model_sbert.encode(persona_job_text, convert_to_tensor=True)
    persona_emb_e5 = model_e5.encode('query: ' + persona_job_text, convert_to_tensor=True)

    ranked_docs, full_texts = rank_documents(documents, persona_emb_sbert, persona_emb_e5, pdf_dir)

    threshold = 0.15
    selected_docs = [d for d in ranked_docs if d[0] > threshold]
    if len(selected_docs) < 1:
     
        selected_docs = ranked_docs[:3]

    if DEBUG:
        print(f"[DEBUG] Selected {len(selected_docs)} documents for detailed analysis.")

    all_sections = []
    processed_docs = []
    for _, filename in selected_docs:
        try:
            pdf_path = os.path.join(pdf_dir, filename)
            pdf = fitz.open(pdf_path)
            sections = extract_sections(pdf)
            for s in sections:
                s['document'] = filename
            all_sections.extend(sections)
            processed_docs.append(filename)
        except Exception as e:
            print(f"[WARNING] Failed to process {filename}: {e}")

    if not all_sections:
        print("[WARNING] No sections extracted from selected documents.")

    tfidf_keywords = extract_tf_idf_keywords(full_texts)

    top_sections = rank_sections(all_sections, persona_emb_sbert, persona_emb_e5, tfidf_keywords)[:6]

    subsection_analysis = []
    for idx, sec in enumerate(top_sections):
        pdf_path = os.path.join(pdf_dir, sec['document'])
        pdf = fitz.open(pdf_path)
        paras = extract_relevant_paragraphs(pdf, sec['page_number'] - 1, persona_emb_sbert, persona_emb_e5)
        for p in paras:
            subsection_analysis.append({
                "document": sec['document'],
                "refined_text": p,
                "page_number": sec['page_number']
            })
        sec['importance_rank'] = idx + 1

    output = {
        "metadata": {
            "input_documents": processed_docs,
            "persona": persona,
            "job": job,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": s['document'],
                "page_number": s['page_number'],
                "section_title": s['section_title'],
                "importance_rank": s['importance_rank']
            } for s in top_sections
        ],
        "subsection_analysis": subsection_analysis
    }

    with open(output_json_path, 'w', encoding='utf-8') as outf:
        json.dump(output, outf, indent=2, ensure_ascii=False)

    print("Processing complete. Output saved to:", output_json_path)


if __name__ == '__main__':
    import argparse
    main()
