import os
import pandas as pd
import io
from uuid import uuid4
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

load_dotenv()

def load_csv_files_with_metadata(csv_files):
    """Load CSV files, enrich with metadata, and return Document objects."""
    all_documents = []

    for file in csv_files:
        try:
            file_bytes = file.read()
            file.seek(0)
            df = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')))

            file_name = os.path.basename(file.name).split('.')[0]

            for _, row in df.iterrows():
                subject = str(row.get('label_subject', 'Subjek tidak diketahui'))
                vervar = str(row.get('label_vervar', 'Wilayah tidak diketahui'))
                var = str(row.get('label_var', 'Variabel tidak diketahui'))
                turvar = str(row.get('label_turvar', '')).strip()
                tahun = str(row.get('label_tahun', 'Tahun tidak diketahui'))
                nilai = str(row.get('nilai', 'Nilai tidak tersedia'))

                if turvar and turvar != "-":
                    content = (
                        f"Pada {turvar} tahun {tahun}, dalam topik '{subject}', indikator '{var}' untuk kategori '{vervar}' "
                        f"memiliki nilai sebesar {nilai}."
                    )
                else:
                    content = (
                        f"Pada tahun {tahun}, dalam topik '{subject}', indikator '{var}' untuk kategori '{vervar}' "
                        f"memiliki nilai sebesar {nilai}."
                    )

                metadata = {
                    "source": file_name,
                    "subject": subject,
                    "vervar": vervar,
                    "var": var,
                    "turvar": turvar if turvar != "-" else None,
                    "tahun": tahun,
                    "nilai": nilai,
                }

                document = Document(page_content=content, metadata=metadata)
                all_documents.append(document)

        except Exception as e:
            print(f"[ERROR] Gagal memproses file {file.name}: {e}")

    return all_documents


def create_or_update_vector_store(documents, batch_size):
    """Create or update a vector store with the given documents in batches."""
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    try:
        if client.collection_exists("inda_collection"):
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="inda_collection",
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
            print("📁 Koleksi vector store sudah ada. Melanjutkan update...")
        else:
            client.create_collection(
                collection_name="inda_collection",
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                }
            )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="inda_collection",
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
            print("🆕 Koleksi vector store baru berhasil dibuat.")

        # Batch processing
        total_docs = len(documents)
        success_count = 0

        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            uuids = [str(uuid4()) for _ in batch_docs]

            try:
                print(f"🔄 Memproses batch {i // batch_size + 1}: dokumen {i} - {i + len(batch_docs) - 1}")
                vector_store.add_documents(documents=batch_docs, ids=uuids)
                success_count += len(batch_docs)
                print(f"✅ Batch sukses: {len(batch_docs)} dokumen")
            except Exception as e:
                print(f"❌ Gagal di batch {i // batch_size + 1}: {e}")
                break

        print(f"🎯 Total dokumen berhasil dimasukkan: {success_count}/{total_docs}")

    except Exception as e:
        print(f"🚫 Error saat membuat atau memperbarui vector store: {e}")
        return None

    return vector_store
