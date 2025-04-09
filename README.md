# HER2 Q&A RAG Chatbot

A question-answering chatbot built on a scientific paper about HER2-positive breast cancer. Powered by:

- FAISS vector search with sentence-transformer embeddings
- Reranking using `BAAI/bge-reranker-base`
- Local LLM (TinyLlama) for RAG-based Q&A
- Streamlit for an interactive UI

---

## 📖 Project Overview

This chatbot enables interactive exploration of the paper:  
[Slamon et al., *Human breast cancer: correlation of relapse and survival with amplification of the HER-2/neu oncogene*](https://www.researchgate.net/...).

- 📄 PDF: [`/data/her2_paper.pdf`](/data/her2_paper.pdf)  
- 🔍 Use case: Biomedical research Q&A  
- ⚙️ Local + lightweight setup (TinyLlama, CPU inference)  
- 📈 Evaluated with curated QA dataset

---

## 🧠 How It Works

1. Parse the HER2 scientific paper PDF.  
2. Split into chunks and embed with a sentence transformer.  
3. Store embeddings in FAISS.  
4. Retrieve top chunks for a user query.  
5. Optionally rerank using `bge-reranker-base`.  
6. Generate response using TinyLlama with RAG.

---

## 🛠 Setup Instructions

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate her2-rag-env
```

### 2. Build the Vectorstore
```bash
jupyter notebook notebook/build_vectorstore.ipynb
```

### 3. Launch the Chatbot
```bash
streamlit run app/rag_app.py --server.fileWatcherType none
```

---

## 💬 User Interface

Streamlit-based interactive UI with:

- Text input for natural language questions
- Markdown-formatted model response
- Loading spinner while processing

**Screenshot:**  
![Chatbot UI](data/HER2%20Q&A%20Chatbot%20-%20screenshot.png)

---

## 📊 Evaluation Methodology

Evaluation performed using a curated set of 10 QA pairs from the HER2 paper.

### 🧪 Method
- Questions focus on key findings
- Answers generated via `get_answer()`
- Compared against gold answers using:
  - **Exact Match (EM)**
  - **F1 Score**

### 🗞 Results Summary
- **Avg. EM:** _0.0_  
- **Avg. F1 Score:** _Low to moderate_  
_(Expected due to TinyLlama not being fine-tuned for biomedical QA)_

### ⚠️ Limitations
- TinyLlama is small and untuned
- CPU inference = slow eval
- Domain mismatch for biomedical text

---

## 🚀 Future Improvements

- Swap in larger models (e.g., Mistral, MedAlpaca, Phi-2)
- Fine-tune LLMs for biomedical QA
- Add context-aware reranking
- Enable faster latency with GPU hosting or cloud services
- Add user feedback to the UI

---

## 📁 Output

Evaluation predictions saved to:
```
data/qa/her2_eval_predictions.csv
```
Includes: question, gold answer, prediction, EM/F1 scores.

---

## 🧪 Evaluation Framework

### 📌 Business KPIs

| KPI               | Metric        | Target   | Method                    |
|------------------|---------------|----------|---------------------------|
| **Accuracy**      | F1 Score      | > 70%    | Curated QA dataset        |
| **Satisfaction**  | Feedback score| Optional | Streamlit feedback        |
| **Latency**       | Response time | ≤ 2 sec  | Logging in Streamlit app  |

### 🧪 Testing Strategy

**1. Offline QA Dataset Evaluation**  
- Based on HER2 paper  
- Manual or LLM-curated QA  
- EM + F1 scoring  
- Color-coded weak answer inspection

**2. Live User Testing (Optional)**  
- Collect real queries  
- Identify hallucinations or low-F1 responses  
- Expand QA set with validated pairs

### ♻️ Continuous Improvement

- Rolling logs for queries and responses  
- Human-in-the-loop review  
- Embedding model upgrades  
- Fine-tune or switch LLMs  
- Optional dashboard for tracking

---

## 🧠 Integrating Clinical Knowledge Embeddings (CKE)

From [MIMS Harvard: Clinical Knowledge Embeddings](https://github.com/mims-harvard/Clinical-knowledge-embeddings)

### Why Use CKE?

- Boosts retrieval relevance with concept-level semantics
- Expands queries with medically similar terms
- Reranks passages based on clinical concept overlap

### Ideas for Integration

1. **Query Expansion**  
   Add synonyms/concepts using CKE embeddings  
2. **Semantic Reranking**  
   Rerank chunks by clinical relevance  
3. **Context Boosting/Filtering**  
   Emphasize or filter context chunks by concept overlap  
4. **Answer Enrichment**  
   Highlight or reinforce clinical terms in the answer

### Use Cases

- Biomedical QA  
- Translational research  
- Document summarization  
- PubMed-style search with RAG

### 🔗 Key Takeaways

1. **Standardized Clinical Knowledge Representation**  
   - Integrates seven major medical vocabularies (diagnoses, labs, medications, etc.) into a single embedding space.  
   - Removes the need for manual curation by learning from over 1.3 million relationships in a clinical knowledge graph.  
   - Enables seamless use of EHR-coded data in machine learning applications.

2. **Validated Across Large-Scale Real-World Data**  
   - Evaluated using a phenotype risk score (PheRS) across 4.57 million patients (Clalit Healthcare).  
   - Embeddings effectively stratify patient survival risk.  
   - Clinical relevance validated by expert panels covering 90+ diseases and 3,000+ codes.

3. **Advancing Precision Medicine**  
   - Facilitates knowledge-grounded patient modeling, enhancing personalization in treatment and care recommendations.  
   - Enables embedding-based similarity, prediction, and stratification tasks with high clinical fidelity.

### 📊 Relevance to Humana

Why this matters for a health insurance provider like Humana:

- **Predictive Modeling & Risk Stratification**  
  Improve identification of high-risk patients for early intervention.  
  Enhance care management and reduce preventable utilization costs.

- **Interoperability & Standardization**  
  Normalize diverse clinical vocabularies across healthcare partners.  
  Streamline integration of external EHR data into Humana's systems.

- **Data-Driven Personalization**  
  Power member-facing applications (e.g., care navigation, preventive alerts) with clinically meaningful embeddings.  
  Improve value-based care outcomes with knowledge-informed analytics.

---

## 📂 Project Structure

```
her2-rag-chatbot/
├── app/                         # Streamlit UI
│   └── rag_app.py
├── data/
│   ├── her2_paper.pdf
│   └── qa/
│       ├── her2_qa_dataset_v2.json
│       └── her2_eval_predictions.csv
├── her2_faiss_db/
│   └── index.faiss
│   └── index.pkl
├── notebook/
│   ├── build_vectorstore.ipynb
│   └── evaluate_qa_model.ipynb
├── environment.yml
├── requirements.txt
├── README.md
└── .gitignore
```

