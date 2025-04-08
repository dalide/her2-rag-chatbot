import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

st.set_page_config(page_title="HER2 Q&A Chatbot")

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/allenai-specter")
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "her2_faiss_db"))
    vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

@st.cache_resource
def load_local_llm_pipeline():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cpu')
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

vectorstore = load_vectorstore()
llm_pipeline = load_local_llm_pipeline()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def build_prompt(context: str, question: str) -> str:
    return f"""You are a biomedical research assistant. Use the context to write a detailed, factual, and specific answer suitable for a graduate-level researcher.

Context:
{context}

Question:
{question}

Answer:"""

# def build_prompt(context: str, question: str) -> str:
#     return f"""You are a biomedical research assistant helping summarize scientific papers.

# Context:
# {context}

# Task:
# Answer the following question using exact numbers or statistics from the context when available.

# Question: {question}
# Answer:"""


def get_answer(query: str) -> str:
    # Retrieve chunks
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Debug: show retrieved chunks
    print("\n🔍 Top Retrieved Chunks:")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---\n{doc.page_content[:300]}...\n")

    # Build and truncate prompt
    prompt = build_prompt(context, query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    truncated_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    print("\n🧪 Truncated Prompt to Model:\n", truncated_prompt[:1000])

    # Generate
    result = llm_pipeline(truncated_prompt, max_new_tokens=768, do_sample=True, temperature=0.3)
    return result[0]["generated_text"]

# Streamlit app UI
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
