import os
import sys

# Add project root to sys.path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from langchain_community.vectorstores import FAISS
from utils.pdf_vector_utils import load_vector_store

st.set_page_config(page_title="HER2 Q&A Chatbot")
st.title("🔬 HER2 Q&A Chatbot (with Chat History)")

# Always use CPU for compatibility and stability
DEVICE = "cpu"

def build_prompt(context: str, history: list, question: str) -> str:
    history_text = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history
    )
    prompt = (
        "You are a biomedical research assistant. Use the provided paper context "
        "and conversation history to answer the user's question accurately and in detail.\n\n"
        f"Context:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n"
        f"User: {question}\nAssistant:"
    )
    return prompt

@st.cache_resource
def load_vectorstore():
    db_path = os.path.abspath("her2_faiss_db")
    return load_vector_store(persist_directory=db_path, model_name="sentence-transformers/allenai-specter")

@st.cache_resource
def load_phi2_pipeline():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return tokenizer, pipe

@st.cache_resource
def load_reranker():
    model_id = "BAAI/bge-reranker-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to("cpu")
    return tokenizer, model

def rerank_chunks(query: str, docs: list, tokenizer, model, top_k: int = 5) -> list:
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        scores = logits.tolist() if logits.ndim > 0 else [logits.item()]

    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]

def get_answer(query: str, history: list) -> str:
    docs = vectorstore.similarity_search(query, k=5)
    reranker_tokenizer, reranker_model = load_reranker()
    top_docs = rerank_chunks(query, docs, reranker_tokenizer, reranker_model, top_k=3)

    context = "\n\n".join(doc.page_content[:300] for doc in top_docs)
    prompt = build_prompt(context, history, query)

    result = llm_pipeline(prompt, max_new_tokens=256, do_sample=False, temperature=0.3)
    return result[0]["generated_text"].split("Assistant:")[-1].strip()

# Load resources
vectorstore = load_vectorstore()
llm_tokenizer, llm_pipeline = load_phi2_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something about the HER2 paper...")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = get_answer(query, st.session_state.chat_history)
            st.session_state.chat_history.append({"user": query, "assistant": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display chat history
for turn in st.session_state.chat_history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['assistant']}")