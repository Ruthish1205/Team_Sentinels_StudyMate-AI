import os
import json
import time
import PyPDF2
import openai
import pickle
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_CACHE_DIR = "embedding_cache"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)


def load_pdf_with_page_refs(file_path: str) -> List[Document]:
    """Extracts text from a PDF and tags each page with metadata."""
    reader = PyPDF2.PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            metadata = {"source": f"{os.path.basename(file_path)} - Page {i + 1}"}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into manageable chunks with metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)

def get_cache_path(file_list: List[str]) -> str:
    """Creates a unique cache file name based on input file names."""
    name = "_".join(os.path.basename(f).replace(".pdf", "") for f in file_list)
    return os.path.join(EMBED_CACHE_DIR, f"{name}_cache.pkl")

def cache_exists(cache_path: str) -> bool:
    return os.path.exists(cache_path)

def load_cached_vectorstore(cache_path: str) -> FAISS:
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def save_vectorstore(vectorstore: FAISS, cache_path: str):
    with open(cache_path, "wb") as f:
        pickle.dump(vectorstore, f)

def build_vectorstore(file_paths: List[str]) -> Tuple[FAISS, List[str]]:
    """Loads PDFs, splits, embeds, and builds a FAISS vectorstore."""
    all_docs = []
    loaded_files = []

    for path in file_paths:
        if not os.path.isfile(path):
            print(f"⚠️ File not found: {path}")
            continue
        try:
            docs = load_pdf_with_page_refs(path)
            all_docs.extend(docs)
            loaded_files.append(path)
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

    if not all_docs:
        raise ValueError("No valid PDF documents loaded.")

    print("🔧 Chunking and embedding...")
    chunks = chunk_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, loaded_files

def ask_question(vectorstore: FAISS, query: str) -> str:
    """Uses OpenAI to answer questions with sources."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    sources = "\n".join(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    return f"{answer}\n\n📎 Sources:\n{sources}"

def summarize_pdfs(vectorstore: FAISS) -> str:
    """Generates a general summary of the uploaded PDFs."""
    docs = vectorstore.similarity_search("Provide a summary of the document", k=5)
    llm = OpenAI(temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")
    summary = chain.run(input_documents=docs, question="Give a short academic summary.")
    return summary

def export_history(history: List[Tuple[str, str]], filename="qa_history.txt"):
    """Exports question and answer history to a text file."""
    with open(filename, "w", encoding="utf-8") as f:
        for q, a in history:
            f.write(f"Q: {q}\nA: {a}\n\n")
    print(f"📤 Q&A history saved to {filename}")



def main():
    print("📚 Welcome to StudyMate Pro!")
    input_paths = input("Enter PDF paths (comma-separated): ").split(",")
    file_paths = [f.strip() for f in input_paths if f.strip()]

    if not file_paths:
        print("❌ No PDF files specified. Exiting.")
        return

    cache_path = get_cache_path(file_paths)
    if cache_exists(cache_path):
        print("✅ Using cached embeddings.")
        vectorstore = load_cached_vectorstore(cache_path)
    else:
        print("⏳ Building knowledge base...")
        vectorstore, loaded = build_vectorstore(file_paths)
        save_vectorstore(vectorstore, cache_path)
        print(f"✅ Embeddings cached for: {', '.join(os.path.basename(f) for f in loaded)}")

    history = []

    while True:
        print("\nOptions:\n  [1] Ask a question\n  [2] Summarize PDFs\n  [3] Export Q&A history\n  [4] Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            question = input("❓ Ask your question: ").strip()
            if not question:
                print("⚠️ Question cannot be empty.")
                continue
            try:
                answer = ask_question(vectorstore, question)
                history.append((question, answer))
                print(f"\n🧠 Answer:\n{answer}")
            except Exception as e:
                print(f"❌ Error during question answering: {e}")
        elif choice == "2":
            print("📝 Generating summary...")
            try:
                summary = summarize_pdfs(vectorstore)
                print(f"\n📄 Summary:\n{summary}")
            except Exception as e:
                print(f"❌ Failed to generate summary: {e}")
        elif choice == "3":
            export_history(history)
        elif choice == "4":
            print("👋 Exiting StudyMate Pro. Goodbye!")
            break
        else:
            print("⚠️ Invalid option. Please try again.")

if __name__ == "__main__":
    main()
