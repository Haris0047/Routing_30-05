# api_catalog_bigtool.py

from __future__ import annotations

import os
import json
import time
import uuid
import logging
from typing import Any, Dict, List

import numpy as np
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from langchain_core.tools import BaseTool, tool  # LangChain's tool decorator :contentReference[oaicite:9]{index=9}
from langgraph_bigtool import create_agent  # BigTool registry from langgraph-bigtool :contentReference[oaicite:10]{index=10}
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# ==============================================================================
# LOGGER SETUP
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# ENVIRONMENT & CONFIGURATION
# ==============================================================================
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL", "http://74.208.122.216:6333/")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_KEY:
    logger.error("Missing OPENAI_API_KEY; aborting.")
    raise SystemExit("❗️OPENAI_API_KEY must be set.")

openai.api_key = OPENAI_KEY

# ==============================================================================
# EMBEDDING HELPER (CACHED + RATE LIMIT)
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
    """
    Caches embeddings to avoid recomputing.
    Uses OpenAI's text-embedding-3-large under the hood.
    """
    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def get_or_compute(self, texts: List[str]) -> List[List[float]]:
        to_compute: List[str] = []
        idx_map: Dict[int, int] = {}
        results: List[None|List[float]] = [None] * len(texts)

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
                        f"OpenAI embedding error (retry {attempt+1}/{max_retries}): {e}. Retrying in {delay}s."
                    )
                    time.sleep(delay)
                else:
                    logger.error("Exceeded OpenAI embedding retries. Aborting.")
                    raise
        return []

_embedding_cache = EmbeddingCache()


def get_cached_embeddings(texts: List[str]) -> List[List[float]]:
    return _embedding_cache.get_or_compute(texts)

# ==============================================================================
# QDRANT CLIENT UTILITY
# ==============================================================================
def get_qdrant_client() -> QdrantClient:
    """
    Returns a QdrantClient connected to QDRANT_URL and QDRANT_PORT.
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
# FUNCTION TO LOAD FMP CATALOG JSON
# ==============================================================================
def load_api_catalog(catalog_path: str) -> List[Dict[str, Any]]:
    """
    Load and return the list of APIs from fmp_api_catalog.json.
    """
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            apis = data.get("apis", [])
            if not isinstance(apis, list):
                raise ValueError("'apis' field is missing or not a list")
            return apis
    except Exception as e:
        logger.error(f"Failed to load catalog JSON: {e}")
        raise

# ==============================================================================
# STEP 1: REGISTER EACH API AS A LANGCHAIN TOOL
# ==============================================================================
def build_langchain_tool(api_entry: Dict[str, Any]) -> BaseTool:
    """
    Given one API entry dict with keys:
      - endpoint
      - name
      - description
      - parameters (list of {name,type,required,description})
      - response (string)
    Return a LangChain Tool with:
      - name = api_entry['name']
      - description = api_entry['description']
      - args_json_schema derived from 'parameters'
    """
    name = api_entry["name"]
    description = api_entry["description"]
    params = api_entry.get("parameters", [])

    # Build JSON Schema for function arguments
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []
    for p in params:
        pname = p["name"]
        ptype = p.get("type", "string")
        prop_schema: Dict[str, Any] = {"type": ptype}
        prop_schema["description"] = p.get("description", "")
        properties[pname] = prop_schema
        if p.get("required", False):
            required.append(pname)

    args_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }

    # We don't actually implement the function, because BigTool just uses metadata.
    # But LangChain's BaseTool requires an 'invoke' method, so we stub it to pass.
    @tool(name=name, description=description, args_json_schema=args_schema)
    def dummy_tool(**kwargs) -> str:
        """
        This dummy tool is a placeholder. In practice, your agent would replace
        this with the actual HTTP call logic (e.g. calling the FMP endpoint).
        """
        return f"Invoked {name} with args: {kwargs}"

    return dummy_tool

# ==============================================================================
# STEP 2: INGEST TOOLS INTO QDRANT (FOR SEMANTIC SEARCH) + REGISTER WITH BigTool
# ==============================================================================
def ingest_into_qdrant_and_bigtool(catalog_path: str):
    """
    1) Load catalog JSON.
    2) Build a LangChain Tool object for each API entry.
    3) Compute a combined 'metadata text' string for each tool: name + description + param names + response.
    4) Get embeddings (via OpenAI) and upsert into Qdrant collection "bigtool_api_catalog".
    5) Register each tool with a BigToolRegistry, pointing to our QdrantStore as the embedding backend.
    """
    # 1) Load catalog
    api_entries = load_api_catalog(catalog_path)
    if not api_entries:
        logger.error("No API entries found. Exiting ingestion.")
        return

    # 2) Build LangChain Tool objects + metadata texts
    tools: Dict[str, BaseTool] = {}
    texts_for_embedding: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for entry in api_entries:
        endpoint = entry["endpoint"]
        api_tool = build_langchain_tool(entry)
        tool_id = str(uuid.uuid4())  # unique ID for each tool in our registry
        tools[tool_id] = api_tool

        # Build a metadata text for embedding: include name, desc, param names & response
        param_names = [p["name"] for p in entry.get("parameters", [])]
        text = (
            f"{entry['name']}. {entry['description']}\n"
            f"Endpoint: {entry['endpoint']}\n"
            f"Parameters: {', '.join(param_names)}\n"
            f"Response Summary: {entry.get('response', '')}"
        )
        texts_for_embedding.append(text)

        # Prepare payload for Qdrant
        payloads.append({
            "tool_id": tool_id,
            "name": entry["name"],
            "description": entry["description"],
            "endpoint": entry["endpoint"],
            "parameters": entry.get("parameters", []),
            "response": entry.get("response", "")
        })

    # 3) Compute embeddings in one batch
    logger.info(f"Computing embeddings for {len(texts_for_embedding)} tools...")
    embeddings = get_cached_embeddings(texts_for_embedding)

    # 4) Connect to Qdrant and create/recreate collection
    client = get_qdrant_client()
    collection_name = "bigtool_api_catalog"

    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing Qdrant collection '{collection_name}'.")
    except Exception:
        pass  # If it does not exist, that's fine

    vector_size = len(embeddings[0]) if embeddings else 3072
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    logger.info(f"Created Qdrant collection '{collection_name}' (vector_size={vector_size}).")

    # Upsert points in batches
    batch_size = 64
    points: List[PointStruct] = []
    for idx, (emb, payload) in enumerate(zip(embeddings, payloads)):
        points.append(PointStruct(id=idx, vector=emb, payload=payload))

    for i in range(0, len(points), batch_size):
        chunk = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=chunk)
        logger.info(f"Upserted points {i}–{i+len(chunk)-1} into '{collection_name}'.")

    # 5) Build a Qdrant‐backed LangChain vector store (via langchain-qdrant)
    qdrant_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings()
    )

    # 6) Create a BigTool registry backed by our QdrantStore
    bigtool_registry = BigToolRegistry(vector_store=qdrant_store)

    # Register each tool with BigTool
    for tool_id, lc_tool in tools.items():
        bigtool_registry.register_tool(tool_id=tool_id, tool=lc_tool)

    logger.info(f"Registered {len(tools)} tools with BigToolRegistry.")
    return bigtool_registry


# ==============================================================================
# STEP 3: QUERY FUNCTION (THE "BigTool" USAGE)
# ==============================================================================
def find_api_tools(query: str, top_k: int = 5):
    """
    Given a user query string, use the BigToolRegistry (with Qdrant vector store)
    to find the top_k relevant API tools. Returns a list of (tool_id, similarity_score).
    """
    # If no registry is built, prompt user to ingest
    if not hasattr(find_api_tools, "_bigtool"):
        raise RuntimeError("BigToolRegistry not initialized—run ingestion first.")

    registry: BigToolRegistry = find_api_tools._bigtool

    # BigTool's `retrieve_tools_by_query` returns a list of (tool_id, score) tuples
    results = registry.retrieve_tools_by_query(query_text=query, k=top_k)
    return results

# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest FMP API catalog into Qdrant + BigTool, then optionally query."
    )
    parser.add_argument(
        "--catalog", "-c",
        default="fmp_api_catalog.json",
        help="Path to the JSON API catalog"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest catalog into BigTool/Qdrant")
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query BigTool for best matching APIs")
    query_parser.add_argument("user_query", type=str, help="Natural‐language query")
    query_parser.add_argument(
        "--top_k", "-k", type=int, default=5,
        help="Number of top results to return"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        registry = ingest_into_qdrant_and_bigtool(args.catalog)
        # Store registry on function for subsequent calls
        find_api_tools._bigtool = registry
        print("✅ Ingestion complete. You can now run 'query'.")
    elif args.command == "query":
        # Ensure ingestion was run in this process
        if not hasattr(find_api_tools, "_bigtool"):
            print("Error: You must first run 'ingest' in this process to initialize BigTool.")
            return

        matches = find_api_tools(args.user_query, top_k=args.top_k)
        if not matches:
            print("No matching tools found.")
            return

        print(f"\nTop {len(matches)} API matches for '{args.user_query}':\n")
        for i, (tool_id, score) in enumerate(matches, start=1):
            # We can fetch the payload from Qdrant to show details
            resp = find_api_tools._bigtool.vector_store.client.get_point(
                collection_name="bigtool_api_catalog",
                id=int(tool_id)  # Qdrant point ID matches the index
            )
            payload = resp.payload
            print(f"{i}. Name        : {payload['name']}")
            print(f"   Endpoint    : {payload['endpoint']}")
            print(f"   Description : {payload['description']}")
            print(f"   Parameters  : {json.dumps(payload['parameters'], indent=4)}")
            print(f"   Response    : {payload['response']}")
            print(f"   Similarity  : {score:.3f}\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
