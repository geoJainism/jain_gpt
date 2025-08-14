# retrieval.py
import os, pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq # change
from dotenv import load_dotenv
load_dotenv()

gemini_api = os.getenv("GEMINI_API_KEY")

def load_resources():
    index = faiss.read_index("jain_faiss.index")
    with open("jain_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return index, texts, embed_model

def query_jain_gpt(question: str, k=5):
    idx, texts, embedder = load_resources()
    q_emb = embedder.encode([question])
    D, I = idx.search(q_emb, k)
    context = "\n\n".join(texts[i] for i in I[0])
    print(context)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api)
    prompt = f"""
    आप एक सहायक बॉट हैं जिसका नाम "Jain GPT" है। निम्नलिखित अवधरणाओं के संदर्भ में प्रश्न का उत्तर दें:

    संदर्भ:
    {context}

    प्रश्न: {question}
    उत्तर (हिंदी में):"""
    resp = llm.invoke([("system", "You are Jain GPT."), ("human", prompt)])
    return resp.content, context

def debug_retrieval(question: str, k=5):
    idx, texts, embedder = load_resources()
    q_emb = embedder.encode([question])
    D, I = idx.search(q_emb, k)

    print("\n🔍 Top Matching Chunks for:", question)
    for rank, i in enumerate(I[0]):
        print(f"\n--- Chunk #{rank+1} ---")
        print(texts[i])

