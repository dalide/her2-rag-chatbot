import os
import fitz
import spacy
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

def spacy_sentence_tokenize(text: str) -> list:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def load_pdf_to_documents(pdf_path: str) -> list:
    documents = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().replace("-\n", "").replace("\n", " ").strip()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i}))
    return documents

def sentence_overlap_chunk(text: str, max_tokens: int = 150, overlap_sent_count: int = 2) -> list:
    sentences = spacy_sentence_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if current_len + token_count <= max_tokens:
            current_chunk.append(sentence)
            current_len += token_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sent_count:] + [sentence]
            current_len = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def analyze_chunks(chunks: list):
    token_lengths = [len(chunk.page_content.split()) for chunk in chunks]

    print(f"Total Chunks: {len(token_lengths)}")
    print(f"Avg Tokens per Chunk: {sum(token_lengths)/len(token_lengths):.2f}")
    print(f"Min Tokens: {min(token_lengths)}")
    print(f"Max Tokens: {max(token_lengths)}")

    plt.hist(token_lengths, bins=20)
    plt.title("Chunk Token Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Number of Chunks")
    plt.show()

def build_vector_store(documents: list,
                       max_tokens: int = 250,
                       overlap_sent_count: int = 3,
                       model_name: str = "sentence-transformers/allenai-specter",
                       persist_directory: str = "./vector_db") -> FAISS:

    all_chunks = []
    for doc in documents:
        chunks = sentence_overlap_chunk(doc.page_content, max_tokens=max_tokens, overlap_sent_count=overlap_sent_count)
        all_chunks.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])

    analyze_chunks(all_chunks)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(persist_directory)
    return vectorstore

def load_vector_store(persist_directory: str,
                      model_name: str = "sentence-transformers/allenai-specter") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

def query_vector_store(vectorstore: FAISS, query: str, k: int = 3, show: bool = True):
    results = vectorstore.similarity_search(query, k=k)
    if show:
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} (Page {doc.metadata.get('page')}):\n{doc.page_content[:500]}...\n")
    return results
