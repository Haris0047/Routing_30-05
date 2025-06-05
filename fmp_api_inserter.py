import os
import json
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid
import logging
from datetime import datetime

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
COLLECTION_NAME = "tools"

openai.api_key = OPENAI_API_KEY

# Setup logger
logging.basicConfig(
    filename="insert_tools.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setup Qdrant client and collection
qdrant_client = QdrantClient(url=QDRANT_URL)
try:
    collections = qdrant_client.get_collections()
    collection_names = [col.name for col in collections.collections]
    if COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️  Collection {COLLECTION_NAME} already exists")
except Exception as e:
    print(f"❌ Error setting up collection: {e}")
    logging.error(f"Error setting up collection: {e}")

def create_embedding(text: str) -> list:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        logging.error(f"Error creating embedding for text: {text[:50]}... | Error: {e}")
        return [0.0] * 1536

def insert_apis_from_catalog(catalog_path="new_catalog.json"):
    """Insert all APIs from the given JSON catalog into Qdrant, embedding only the description."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        apis_data = json.load(f)

    print(f"🚀 Starting API insertion from {catalog_path} with OpenAI embeddings (description only)...")
    for api_data in apis_data:
        status = "success"
        try:
            embedding = create_embedding(api_data["description"])
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=api_data
            )
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
            print(f"✅ Inserted API: {api_data['name']}")
        except Exception as e:
            status = f"failure: {e}"
            print(f"❌ Failed to insert API: {api_data['name']} | Error: {e}")
        # Log the insertion
        logging.info(f"Tool: {api_data['name']} | Status: {status}")
    print(f"✅ Successfully inserted {len(apis_data)} APIs into Qdrant!")

if __name__ == "__main__":
    print("🏗️  Inserting/syncing APIs from new_catalog.json into Qdrant...")
    insert_apis_from_catalog()
    print("✅ API insertion complete!") 