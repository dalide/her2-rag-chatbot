# Prevent asyncio error: "no running event loop"
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os 


st.set_page_config(page_title="HER2 Q&A Chatbot")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "her2_faiss_db"))
vectorstore = FAISS.load_local(path, embedding_model,allow_dangerous_deserialization=True)

# Use Hugging Face Inference API with flan-t5-base
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-rw-1b",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="map_reduce")

st.title("🔬 HER2 Q&A Chatbot")
query = st.text_input("Ask something about the HER2 paper...")

if query:
    result = qa.invoke({"query": query})
    st.write("**Answer:**", result)