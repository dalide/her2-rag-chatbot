# HER2 Q&A RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot for answering questions based on a seminal breast cancer paper (HER2).

## 🛠 Setup Instructions

```bash
conda env create -f environment.yml
conda activate rag-chatbot
jupyter notebook notebook/build_vectorstore.ipynb
streamlit run app/rag_app.py
```

## 🐳 Docker (optional)

```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

## 📂 Project Structure
```
├── app/                  # Streamlit interface
│   └── rag_app.py
├── data/                 # Contains HER2 paper
│   └── her2_paper.pdf    # Source: Slamon et al. (1987)
├── notebook/             # For vector DB creation
│   └── build_vectorstore.ipynb
├── her2_faiss_db/        # Saved FAISS vector store
├── environment.yml       # Conda environment
├── Dockerfile            # Docker setup
├── requirements.txt      # For Docker image
├── README.md             # Project instructions
└── .gitignore            # Ignored files
```

## 🔬 Source
Paper: Slamon et al., "Human breast cancer: correlation of relapse and survival with amplification of the HER-2/neu oncogene"  
[ResearchGate PDF](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf)
