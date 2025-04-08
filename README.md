# HER2 Q&A RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot for answering questions based on a [seminal breast cancer paper (HER2)](/data/her2_paper.pdf).  

## 🛠 Setup Instructions

### Create Conda Environment
```bash
conda env create -f environment.yml
conda activate her2-rag-env
```

### Create the Vector Database using the paper
```
jupyter notebook notebook/build_vectorstore.ipynb
```

### Run the Chatbot App
```
streamlit run app/rag_app.py --server.fileWatcherType none
```


## 📂 Project Structure (after generating the vector database)
```
her2-rag-chatbot/
├── app/                          # Streamlit web interface for chatbot
│   └── rag_app.py                # Main app: handles query input, retrieval, and response generation
│
├── data/                         # Project data directory
│   ├── her2_paper.pdf            # Source document used for RAG
│   └── qa/                       # Evaluation assets
│       ├── her2_qa_dataset.json  # Gold-standard QA pairs for evaluation
│       └── her2_predictions.json # Model-predicted answers for comparison
│
├── her2_faiss_db/                # Saved FAISS vectorstore (retriever index)
│   └── index.faiss               # FAISS binary index
│   └── index.pkl                 # Metadata for the FAISS store (LangChain-compatible)
│
├── notebook/                     # Jupyter notebooks for development and evaluation
│   ├── build_vectorstore.ipynb   # Extracts text from PDF and builds FAISS vector store
│   └── evaluate_qa_model.ipynb   # Evaluates chatbot using F1 score against gold QA dataset
│
├── environment.yml               # Conda environment definition (dependencies for setup)
├── requirements.txt              # Optional: pip-based environment (can be auto-generated)
├── README.md                     # Project overview, instructions, and evaluation methodology
├── .gitignore                    # Excludes unneeded files (e.g., .pyc, __pycache__, FAISS temp files)

```

## 🔬 Source
Paper: Slamon et al., "Human breast cancer: correlation of relapse and survival with amplification of the HER-2/neu oncogene"  
[ResearchGate PDF](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf)


---

## User Interface

The HER2 Q&A chatbot is built using **Streamlit** for a lightweight, interactive interface. Users can type questions related to the Her2 publication.

### Example UI:
- A text input field at the top allows users to type natural language questions.
- Below the input, the app displays the chatbot’s answer in markdown format.
- A spinner shows progress while the model processes the query.

![Chatbot UI](data/HER2%20Q&A%20Chatbot%20-%20screenshot.png)

---

## Evaluation Approach

### ◦ Business Metrics (KPIs)

To ensure the chatbot delivers practical value in a biomedical research context, we define the following KPIs:

1. **Response Accuracy (Primary KPI):**
   - Measured using **F1 Score** between the chatbot-generated answer and a human-annotated gold answer.
   - Target: Maintain average F1 > **0.70** across test sets.
   - Weak answers (F1 < 0.5) are flagged for review.
   **F1 Score Definition:**
   The F1 score is the harmonic mean of precision and recall. It measures the overlap between the model-generated answer and the gold answer.
   
   - **Precision** = proportion of predicted words that are correct
   - **Recall** = proportion of gold words that are captured in the prediction
   - **F1 = 2 × (Precision × Recall) / (Precision + Recall)**

   **Example Calculation:**
   - **Gold answer:** "HER-2/neu gene amplification was found in 19 of 103 tumors."
   - **Predicted answer:** "Amplification of HER2/neu was seen in 19 out of 103 cases."
   - **Normalized and tokenized gold:** ["her2neu", "gene", "amplification", "was", "found", "in", "19", "of", "103", "tumors"]
   - **Normalized and tokenized prediction:** ["amplification", "of", "her2neu", "was", "seen", "in", "19", "out", "103", "cases"]
   - **Overlap tokens:** ["amplification", "her2neu", "was", "in", "19", "103"] (6 shared)
   - **Precision:** 6 / 10 = 0.60
   - **Recall:** 6 / 10 = 0.60
   - **F1 Score:** 2 × (0.60 × 0.60) / (0.60 + 0.60) = **0.60**

   
2. **User Satisfaction (Secondary KPI):**
   - Collected via post-interaction feedback (Likert scale or thumbs up/down in app).
   - Optional future enhancement: integrate feedback logging into `rag_app.py`.

3. **Response Time:**
   - Measure time to return an answer from query submission.
   - Acceptable threshold: ≤ **2 seconds** for 90% of queries.

---

### ◦ Testing

We perform rigorous evaluation using a two-tier testing strategy:

#### 1. **Curated QA Dataset Evaluation (Offline)**
   - Built from the HER2 paper. (By Human expert or ChatGPT/Gemini)
   - Each QA pair includes a reference answer and is evaluated using:
     - F1 score (for content overlap)
     - Color-coded analysis of weak answers
   - Can be scaled to include adversarial or paraphrased questions.

#### 2. **Live User Testing (Online)**
   - Log user queries and chatbot responses via Streamlit or backend logging.
   - Periodically review real-user queries for:
     - Coverage gaps
     - Hallucinated or inaccurate answers
     - Opportunities to add QA pairs to the evaluation set

---

### ◦ Continuous Improvement Process

1. **Monitoring**
   - Maintain a rolling log of user interactions and weak responses.
   - Track F1 score trends on benchmark questions.

2. **Human-in-the-loop Feedback Loop**
   - Domain experts or QA reviewers validate low-F1 answers.
   - Approved corrections are added to the gold dataset to expand evaluation coverage.

3. **Model & Retrieval Refinement**
   - Improve context retrieval using embedding model tuning (e.g., better sentence transformers).
   - Fine-tune or switch to domain-specific LLMs (e.g., BioMedLM, BioGPT) based on weak answer patterns.

4. **Dashboard (Optional)**
   - Display metrics like average F1, count of weak responses, and response latency.

---

### Summary Table

| KPI                | Metric               | Target       | Method                          |
|--------------------|----------------------|--------------|----------------------------------|
| Response Accuracy  | F1 Score              | > 70%        | Offline QA dataset              |
| User Satisfaction  | Feedback score        | Qualitative  | Streamlit thumbs/scale (future) |
| Response Time      | Latency               | < 2 sec      | Streamlit logging                |
| Improvement Cycle  | Flagged low-F1 answers| ↓ over time  | QA review & retraining           |

