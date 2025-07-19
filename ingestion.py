# ingestion.py
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_and_embed(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return texts, embeddings

def build_vectorstore(texts, embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "jain_faiss.index")
    with open("jain_texts.pkl", "wb") as f:
        pickle.dump(texts, f)

def main():
    book_text = load_pdf("jain_book_ocr_hindi.txt")
    texts, embeddings = chunk_and_embed(book_text)
    build_vectorstore(texts, embeddings)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
