from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import uuid
import re
import traceback

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.cache import InMemoryCache
from langchain_core.globals import set_llm_cache

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

# Load environment
load_dotenv()

# FastAPI init
app = FastAPI()

# Cache Langchain
set_llm_cache(InMemoryCache())

# Store untuk session
store = {}

# Request model
class TextRequest(BaseModel):
    response_text: str
    notelp: str = None

# Qdrant init
client = QdrantClient(
    url="https://b91cf208-0d2a-4ec6-8711-78a6163bbc32.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

# Embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Vector Store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="inda_collection",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
)

# Daftar GROQ API Key
GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
]

# Fungsi menghapus emoji
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Riwayat percakapan
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Chain utama
def get_conversational_chain():
    prompt_template = """
    Anda adalah INDA (Intelligent Data Assistant) yang menyediakan data dari BPS Provinsi Sumatera Utara yang hanya berfokus pada layanan penyediaan data. Anda juga dapat menjawab salam pembuka dan salam penutup selayaknya pelayan virtual.
    Jawablah dengan ringkas dan akurat.
    Jika ada pertanyaan yang tidak jelas, Anda dapat meminta tahun dan variabel penjelas lainnya.
    Konteks:\n {context}\n
    Pertanyaan Pengguna: \n{input}\n    
    Jawaban yang relevan (berdasarkan dokumen):\n
    """
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    model = get_groq_model_with_fallback()

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# Fungsi generate response
def generate_response(user_question, session_id):
    last_exception = None

    for i, api_key in enumerate(GROQ_KEYS):
        try:
            print(f"🔁 Mencoba GROQ_API_KEY_{i+1}")
            model = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                groq_api_key=api_key
            )

            prompt_template = """
            Anda adalah INDA (Intelligent Data Assistant) yang menyediakan data dari BPS Provinsi Sumatera Utara yang hanya berfokus pada layanan penyediaan data. Anda juga dapat menjawab salam pembuka dan salam penutup selayaknya pelayan virtual.
            Jawablah dengan ringkas dan akurat.
            Jika ada pertanyaan yang tidak jelas, Anda dapat meminta tahun dan variabel penjelas lainnya.
            Konteks:\n {context}\n
            Pertanyaan Pengguna: \n{input}\n    
            Jawaban yang relevan (berdasarkan dokumen):\n
            """

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            config = {"configurable": {"session_id": session_id}}

            response = conversational_rag_chain.invoke(
                {"input": user_question},
                config=config
            )

            response_text = response.get("answer", "")
            return remove_emojis(response_text)

        except Exception as e:
            error_msg = str(e).lower()
            print(f"[‼️ Gagal GROQ_API_KEY_{i+1}] {e}")
            last_exception = e

            # Jika error karena rate limit, lanjut ke key berikutnya
            if any(msg in error_msg for msg in ["rate limit", "quota", "429"]):
                continue
            elif "503" in error_msg or "unavailable" in error_msg:
                return "Maaf, layanan sementara tidak tersedia. Mohon coba beberapa saat lagi."
            else:
                # Jika error bukan karena rate limit, hentikan
                break

    return f"Terjadi kesalahan: {str(last_exception)}"


# Endpoint utama
@app.post("/process_text")
async def process_text(data: TextRequest):
    response_text = data.response_text
    notelp = data.notelp or str(uuid.uuid4())

    if not response_text:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Teks pertanyaan kosong"})

    processed_text = generate_response(response_text, notelp)
    return {
        "status": "success",
        "processed_text": processed_text,
        "notelp": notelp
    }
