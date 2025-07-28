
# 🧑‍💼💡 Persona-Driven Multi-PDF Section Relevance Extractor
# Team Name: Fruits of Binary Tree

###### Challenge: Adobe Hackathon Round 1B

## 🔎 Overview \& Approach

This solution is designed to analyze a collection of PDFs, extract the most relevant section headings (not paragraphs), and surface the most useful content according to a **persona** and **job-to-be-done**. It is robust, domain-agnostic, and runs entirely offline using compact modern LLM embedding models.

**Key Steps:**

1. **Heading Extraction:**
Each PDF is processed to find genuine section headings—using structural layout (font, style, position) and simple heuristics, not just body text.
2. **Relevance Scoring/ranking:**
Each candidate heading (and its context) is semantically compared to the persona and job-to-be-done using _dual semantic embedding models_ for maximum contextual understanding.
3. **Section Ranking:**
The most relevant section headings are ranked and deduplicated.
4. **Subsection Analysis \& Refinement:**
For each top heading, relevant paragraphs are extracted. Very similar paragraphs are clustered/merged to form coherent, deduplicated, fact-only summaries (no hallucinations).
5. **Single Output JSON:**
Results are provided as a single well-formed JSON file adhering to the challenge schema.

## Models \& Libraries Used

| Library | Purpose |
| :-- | :-- |
| **PyMuPDF (fitz)** | Robust PDF parsing \& block text extraction |
| **sentence-transformers** | State-of-the-art semantic embeddings (MiniLM, E5) |
| **scikit-learn** | TF-IDF keyword extraction |
| **re, collections** | Heuristics, data wrangling |
| **datetime, os, json** | Utilities for IO \& formatting |
| **torch** | Backend for neural model inference |

### **Embedding Models**

- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
Fast and compact semantic embedding model (384 dim), ideal for CPU-only setups.
- [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
QA/search-optimized model (384 dim), adds complementary context understanding.



## ⚡ Solution Pipeline: Flowchart

```plaintext
┌────────────────────────────┐
│       Input JSON           │
│ documents, persona, job    │
└────────────┬───────────────┘
             │
       (for each PDF)
             │
┌────────────▼──────────────┐
│ PDF Layout Analysis       │
│ - Extract block text      │
│ - Find heading-like lines │
└────────────┬──────────────┘
             │
   ┌─────────▼─────────┐
   │ Candidate Headings│
   └─────────┬─────────┘
             │
     For each Heading+Context:
             │
┌────────────▼────────────┐
│  Dual Embedding Encode  │
│  (MiniLM + E5-small)    │
└────────────┬────────────┘
             │
┌────────────▼─────────────┐
│  Semantic Similarity     │
│  Rank w/ Persona+Job     │
└────────────┬─────────────┘
             │
      Top K Sections     TF-IDF Boost
             │                 │
        ┌────▼─────┐      ┌────▼─────┐
        │ Rank &   │      │ Keyword  │
        │  Filter  │      │  Score   │
        └────┬─────┘      └────┬─────┘
             │                 │
         ┌───▼─────────────────┴────────┐
         │   Select Top Most Relevant   │
         └─────────────┬────────────────┘
                       │
        For each section/page:
                       │
         ┌─────────────▼─────────────┐
         │ Extract Contextual        │
         │ Paragraphs; Cluster &     │
         │ Deduplicate (semantic)    │
         └─────────────┬─────────────┘
                       │
                ┌──────▼──────┐
                │  Output     │
                │  JSON File  │
                └─────────────┘
```


## Build \& Run Instructions

### 1. **Environment Setup**

**1.1.** Create \& activate a Python virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate     

```

**1.2.** Install required libraries:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` contains:

```plaintext
pymupdf>=1.22.0
torch>=1.12
sentence-transformers>=2.2.2
scikit-learn>=0.24.0
transformers>=4.32.0
```


### 2. **Prepare Inputs**

- Place all **PDFs** to be analyzed in a folder, e.g. `./input/`.
- Place your challenge input JSON (e.g. `challenge1b_input.json`) in the script directory.


### 3. **Run the Solution**

```bash
python your_script_name.py
```

- By default, looks for PDFs in `./input` and saves the output in `./output/challenge1b_output.json`.


### 4. **Results**

- The output JSON structure matches the [challenge schema](#):
    - `metadata` (input documents, persona, job, timestamp)
    - `extracted_sections` (ranked, relevant section headings, doc/page, rank)
    - `subsection_analysis` (fact-preserving, non-duplicate refined content per section)


## How it Works (Summary)

1. Each PDF is scanned to extract actual section headings (using layout heuristics, not just any sentence!).
2. Candidate headings and contexts are encoded and scored for relevance to the persona and job using two lightweight but very strong embedding models.
3. Top sections across all PDFs are deduplicated and ranked.
4. For each top section, relevant paragraphs are pulled and any duplicates/near-duplicates are merged, ensuring high factual accuracy (no hallucination).
5. All results are written to a single, easy-to-use JSON output.

## Customization \& Troubleshooting

- **Increase or decrease the number of output sections**: Change the slice in `top_sections = ...[:6]`.
- **Change similarity thresholds** in the top of the script to be more/less strict.
- **If you want to use a different embedding model** (e.g., MPNet, LaBSE), just change the model name in the model loading lines.
- **For OCR/scanned PDFs:** You can extend the extractor to use Tesseract OCR as fallback.
- **Issues with TF-IDF being empty?** Lower document filtering thresholds.
