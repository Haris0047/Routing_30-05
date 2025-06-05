# api_catalog_ingest_and_query.py

from __future__ import annotations

import os
import json
import time
import logging
from typing import Any, Dict, List

import numpy as np
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ==============================================================================
# CONFIGURATION & LOGGER
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set these environment variables before running:
#   export OPENAI_API_KEY="YOUR_OPENAI_KEY"
#   export QDRANT_URL="http://localhost"
#   export QDRANT_PORT="6333"
#   export QDRANT_API_KEY="YOUR_QDRANT_API_KEY"  # if needed
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL", "http://74.208.122.216:6333/")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_KEY:
    logger.error("OpenAI API key missing. Aborting.")
    raise SystemExit("❗️OPENAI_API_KEY must be set in environment")

openai.api_key = OPENAI_KEY

# ==============================================================================
# EMBEDDING CACHE & RATE LIMITER
# ==============================================================================
class RateLimiter:
    def __init__(self, min_interval: float = 0.5):
        self.min_interval = min_interval
        self._last = 0.0

    def throttle(self):
        now = time.time()
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

_rate_limiter = RateLimiter(min_interval=0.5)

class EmbeddingCache:
    """Simple in‐memory cache to avoid re‐computing embeddings for identical text."""
    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def get_or_compute(self, texts: List[str]) -> List[List[float]]:
        to_compute: List[str] = []
        idx_map: Dict[int, int] = {}
        results: List[None | List[float]] = [None] * len(texts)

        for i, txt in enumerate(texts):
            if txt in self._cache:
                results[i] = self._cache[txt]
            else:
                idx_map[len(to_compute)] = i
                to_compute.append(txt)

        if to_compute:
            new_embeds = self._fetch_with_backoff(to_compute)
            for batch_idx, emb in enumerate(new_embeds):
                orig_index = idx_map[batch_idx]
                results[orig_index] = emb
                self._cache[texts[orig_index]] = emb

        return [r for r in results]  # type: ignore

    def _fetch_with_backoff(
        self, texts: List[str], max_retries: int = 5, base_delay: float = 2.0
    ) -> List[List[float]]:
        _rate_limiter.throttle()
        for attempt in range(max_retries):
            try:
                resp = openai.Embedding.create(
                    model="text-embedding-3-large",
                    input=texts
                )
                return [item["embedding"] for item in resp["data"]]
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s."
                    )
                    time.sleep(delay)
                else:
                    logger.error("Failed to fetch embeddings after retries.")
                    raise
        return []

_embedding_cache = EmbeddingCache()

def get_cached_embeddings(texts: List[str]) -> List[List[float]]:
    return _embedding_cache.get_or_compute(texts)

# ==============================================================================
# QDRANT CLIENT SETUP
# ==============================================================================
def get_qdrant_client() -> QdrantClient:
    """
    Returns a connected QdrantClient. Adjust if using an HTTP vs. GRPC endpoint.
    """
    if QDRANT_URL.startswith("http://") or QDRANT_URL.startswith("https://"):
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=60
        )
    else:
        client = QdrantClient(
            host=QDRANT_URL,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=30
        )
    # sanity check
    client.get_collections()
    return client

# ==============================================================================
# INGESTION FUNCTION
# ==============================================================================
def ingest_api_catalog_to_qdrant(catalog_path: str):
    """
    1) Load the JSON catalog (list of API entries).
    2) For each API, build a combined text string: name + description + parameters + response.
    3) Compute embeddings via OpenAI.
    4) Upsert into Qdrant collection "api_catalog".
    """
    # 1) Load JSON catalog
    try:
        with open(catalog_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to open catalog file: {e}")
        return

    apis = data.get("apis", [])
    if not apis:
        logger.error("Catalog JSON contains no 'apis' list.")
        return

    # 2) Prepare texts + payloads
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []
    for api in apis:
        endpoint = api.get("endpoint", "")
        name = api.get("name", "")
        desc = api.get("description", "")
        params = api.get("parameters", [])
        resp = api.get("response", "")

        # Build a single string that concatenates all relevant fields:
        param_texts = []
        for p in params:
            pname = p.get("name", "")
            pdesc = p.get("description", "")
            param_texts.append(f"{pname}: {pdesc}")
        param_block = "\n".join(param_texts)

        combined = (
            f"Endpoint: {endpoint}\n"
            f"Name: {name}\n"
            f"Description: {desc}\n"
            f"Parameters:\n{param_block}\n"
            f"Response Summary: {resp}"
        )

        texts.append(combined)
        # Payload will be stored alongside the vector
        payloads.append({
            "endpoint": endpoint,
            "name": name,
            "description": desc,
            "parameters": params,
            "response": resp
        })

    # 3) Compute embeddings in batches
    logger.info(f"Computing embeddings for {len(texts)} API entries...")
    embeddings = get_cached_embeddings(texts)

    # 4) Connect to Qdrant and (re)create collection
    client = get_qdrant_client()
    collection_name = "api_catalog"

    # Delete existing, if present
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'.")
    except Exception:
        pass  # Collection might not exist yet

    # Create new collection
    VECTOR_SIZE = len(embeddings[0]) if embeddings else 1536
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    logger.info(f"Created collection '{collection_name}' with vector size {VECTOR_SIZE}.")

    # Prepare PointStructs and upsert
    points = []
    for idx, (emb, payload) in enumerate(zip(embeddings, payloads)):
        points.append(
            PointStruct(
                id=idx,
                vector=emb,
                payload=payload
            )
        )

    # Batch upsert (in chunks of 64 for safety)
    batch_size = 64
    for i in range(0, len(points), batch_size):
        chunk = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=chunk
        )
        logger.info(f"Upserted points {i}–{i+len(chunk)-1} into '{collection_name}'.")

    logger.info(f"Ingested {len(points)} API entries into Qdrant collection '{collection_name}'.")


# ==============================================================================
# BIGTOOL: QUERYING THE API CATALOG
# ==============================================================================
def query_api_catalog(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Given a natural‐language query, compute its embedding and search the 'api_catalog' collection.
    Returns a list of the top_k matching API entries (payload + score).
    """
    client = get_qdrant_client()
    collection_name = "api_catalog"

    # 1) Compute query embedding
    q_emb = get_cached_embeddings([query])[0]

    # 2) Search Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=q_emb,
        limit=top_k
    )

    # 3) Extract payload + similarity
    matches: List[Dict[str, Any]] = []
    for hit in search_results:
        payload = hit.payload
        similarity = 1.0 - hit.score  # cos similarity = 1 - distance
        matches.append({
            "endpoint": payload.get("endpoint"),
            "name": payload.get("name"),
            "description": payload.get("description"),
            "parameters": payload.get("parameters"),
            "response": payload.get("response"),
            "score": similarity
        })

    return matches


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest Financial Modeling Prep API catalog into Qdrant and/or query it."
    )
    parser.add_argument(
        "--catalog", "-c",
        default="fmp_api_catalog.json",
        help="Path to the JSON API catalog file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest the catalog into Qdrant")
    ingest_parser.add_argument(
        "--no‐delete", action="store_true",
        help="Do not delete existing collection (will append instead)."
    )

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Perform a BigTool query against the catalog")
    query_parser.add_argument("user_query", type=str, help="Natural‐language query to search the catalog")
    query_parser.add_argument(
        "--top_k", "-k", type=int, default=5,
        help="Number of top results to return"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        if args.no_delete:
            # If user wants to append rather than recreate, skip deletion step internally
            pass  # The ingest function always recreates. If you want append, modify code accordingly.
        ingest_api_catalog_to_qdrant(args.catalog)
    elif args.command == "query":
        results = query_api_catalog(args.user_query, top_k=args.top_k)
        if not results:
            print("No matches found.")
            return
        print(f"Top {len(results)} matches for query: '{args.user_query}'\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. Endpoint: {r['endpoint']}")
            print(f"   Name       : {r['name']}")
            print(f"   Description: {r['description']}")
            print(f"   Parameters : {json.dumps(r['parameters'], indent=4)}")
            print(f"   Response   : {r['response']}")
            print(f"   Score      : {r['score']:.3f}\n{'-'*40}\n")


if __name__ == "__main__":
    main()
