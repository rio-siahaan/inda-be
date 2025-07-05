from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()

# client = QdrantClient(path="/tmp/langchain_qdrant")
try:
    client = QdrantClient(
            url="https://b91cf208-0d2a-4ec6-8711-78a6163bbc32.europe-west3-0.gcp.cloud.qdrant.io",
            api_key=os.getenv("QDRANT_API_KEY")
        )
    client.delete_collection('csv_collection')
    print("csv_collection sudah dihapus")
except Exception as e:
    print(e)