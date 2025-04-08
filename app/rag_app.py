import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

import os 


st.set_page_config(page_title="HER2 Q&A Chatbot")

# Cache the vector store load to improve performance
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "her2_faiss_db"))
    vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

@st.cache_resource
def load_local_llm_pipeline():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

vectorstore = load_vectorstore()
llm_pipeline = load_local_llm_pipeline()

# Format context + question into a FLAN-style prompt
def build_prompt(context: str, question: str) -> str:
    return f"""You are a biomedical research assistant. Read the context and answer the question in a detailed, informative way suitable for a graduate-level researcher.

Context:
{context}

Question:
{question}

Answer:"""


# Retrieve + format + answer
def get_answer(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=15)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(context, query)
    result = llm_pipeline(prompt, max_new_tokens=512, temperature=0.2)
    return result[0]["generated_text"]


st.title("🔬 HER2 Q&A Chatbot")
query = st.text_input("Ask something about the HER2 paper...")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = get_answer(query)
            st.markdown("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

