# catalog_ingest.py

from __future__ import annotations

import os
import json
import time
import logging

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ==============================================================================
# CONFIGURATION & LOGGER
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables (must be set externally)
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_KEY:
    logger.error("OpenAI API key missing. Aborting ingestion.")
    raise SystemExit("❗️OpenAI key missing.")

openai.api_key = OPENAI_KEY

# ==============================================================================
# DATA CLASSES
# ==============================================================================
@dataclass
class Profile:
    id: str
    name: str
    description: str
    tools: List[str] = field(default_factory=list)


@dataclass
class Tool:
    id: str
    profile_id: str
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    input_schema: Optional[str] = None


# ==============================================================================
# EMBEDDING CACHE & RATE LIMITER
# ==============================================================================
class EmbeddingCache:
    """Cache for embeddings to avoid redundant OpenAI calls."""
    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def get_or_compute_batch(self, texts: List[str]) -> List[List[float]]:
        to_compute = []
        idx_map: Dict[int, int] = {}
        results: List[Optional[List[float]]] = [None] * len(texts)

        for i, txt in enumerate(texts):
            if txt in self._cache:
                results[i] = self._cache[txt]
            else:
                idx_map[len(to_compute)] = i
                to_compute.append(txt)

        if to_compute:
            new_embeds = _fetch_embeddings_with_backoff(to_compute)
            for batch_idx, embed in enumerate(new_embeds):
                orig_index = idx_map[batch_idx]
                results[orig_index] = embed
                self._cache[texts[orig_index]] = embed

        return [r for r in results]  # type: ignore


class RateLimiter:
    """Ensures minimum interval between OpenAI API calls."""
    def __init__(self, min_interval: float = 0.5):
        self.min_interval = min_interval
        self._last = 0.0

    def throttle(self):
        now = time.time()
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()


_embedding_cache = EmbeddingCache()
_rate_limiter = RateLimiter(min_interval=0.5)


def _fetch_embeddings_with_backoff(
    texts: List[str], max_retries: int = 5, base_delay: float = 2.0
) -> List[List[float]]:
    _rate_limiter.throttle()

    for attempt in range(max_retries):
        try:
            resp = openai.embeddings.create(model="text-embedding-3-large", input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"OpenAI embedding error (retry {attempt+1}/{max_retries}): {e}. Sleeping {delay}s."
                )
                time.sleep(delay)
            else:
                logger.error("Exceeded OpenAI embedding retries. Aborting.")
                raise
    return []


def get_cached_embeddings(texts: List[str]) -> List[List[float]]:
    return _embedding_cache.get_or_compute_batch(texts)


# ==============================================================================
# CATALOG LOADING
# ==============================================================================
def load_catalog(file_path: str) -> Tuple[List[Profile], List[Tool]]:
    try:
        raw = json.load(open(file_path, "r"))
        profiles = [
            Profile(id=p["id"], name=p["name"], description=p["description"])
            for p in raw.get("profiles", [])
        ]
        profile_index = {p.id: p for p in profiles}

        tools: List[Tool] = []
        for t in raw.get("tools", []):
            tool = Tool(
                id=t["id"],
                profile_id=t["profile_id"],
                name=t["name"],
                description=t["description"],
                examples=t.get("examples", []),
                input_schema=t.get("input_schema"),
            )
            tools.append(tool)
            if tool.profile_id in profile_index:
                profile_index[tool.profile_id].tools.append(tool.id)

        return profiles, tools
    except Exception as e:
        logger.error(f"Failed to load catalog JSON: {e}")
        return [], []


# ==============================================================================
# QDRANT CLIENT + COLLECTION MANAGEMENT
# ==============================================================================
def get_qdrant_client_with_retry(
    url: str, port: int, api_key: Optional[str] = None, retries: int = 3
) -> QdrantClient:
    for attempt in range(retries):
        try:
            client = (
                QdrantClient(url=url, api_key=api_key, timeout=60)
                if url.startswith(("http://", "https://"))
                else QdrantClient(host=url, port=port, api_key=api_key, timeout=30, prefer_grpc=False)
            )
            client.get_collections()  # sanity check
            logger.info(f"Connected to Qdrant at {url}:{port}")
            return client
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"Qdrant connect failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s."
                )
                time.sleep(wait)
            else:
                logger.error("Qdrant connection retries exhausted. Aborting ingestion.")
                raise
    raise RuntimeError("Unreachable code in Qdrant init.")


def recreate_collections(client: QdrantClient, vector_size: int):
    """Delete & recreate both collections: semantic_profiles + semantic_tools."""
    for collection in ("semantic_profiles", "semantic_tools"):
        try:
            client.delete_collection(collection_name=collection)
            logger.debug(f"Deleted existing Qdrant collection: {collection}")
        except Exception:
            pass  # Wasn't there; no big deal.

    client.create_collection(
        collection_name="semantic_profiles",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_collection(
        collection_name="semantic_tools",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("Recreated Qdrant collections: semantic_profiles, semantic_tools.")


# ==============================================================================
# CORE INGESTION WORKFLOW
# ==============================================================================
def ingest_catalog_to_qdrant(catalog_path: str):
    profiles, tools = load_catalog(catalog_path)
    if not profiles or not tools:
        logger.error("No profiles/tools found in catalog. Exiting ingestion.")
        return

    # Embed dimensionality for text-embedding-3-large
    VECTOR_SIZE = 3072

    client = get_qdrant_client_with_retry(QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY)
    recreate_collections(client, VECTOR_SIZE)

    # Step 1: Embed & Upsert Profiles
    logger.info("Embedding & upserting profiles...")
    profile_texts: List[str] = []
    profile_payloads: List[Dict[str, Any]] = []
    for p in profiles:
        desc = f"Profile: {p.name}. {p.description}"
        tools_list = [t.name for t in tools if t.id in p.tools]
        if tools_list:
            desc += f" Contains tools: {', '.join(tools_list)}."
        profile_texts.append(desc)
        profile_payloads.append({
            "id": p.id,
            "name": p.name,
            "description": p.description
        })

    profile_embeddings = get_cached_embeddings(profile_texts)
    client.upsert(
        collection_name="semantic_profiles",
        points=[
            PointStruct(id=i, vector=embed, payload=payload)
            for i, (embed, payload) in enumerate(zip(profile_embeddings, profile_payloads))
        ],
    )

    # Step 2: Embed & Upsert Tools (use examples from catalog.json directly)
    logger.info("Embedding & upserting tools...")
    tool_texts: List[str] = []
    tool_payloads: List[Dict[str, Any]] = []
    for t in tools:
        profile_ctx = ""
        prof = next((p for p in profiles if p.id == t.profile_id), None)
        if prof:
            profile_ctx = f"Part of {prof.name}: {prof.description}"

        # Build enriched semantics using examples from catalog.json
        enriched = (
            f"Tool: {t.name}\n\n"
            f"Description: {t.description}\n\n"
            f"{profile_ctx}\n\n"
            "Example use cases:\n"
            + "\n".join(f"- {ex}" for ex in t.examples)
        )

        tool_texts.append(enriched)
        tool_payloads.append({
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "profile_id": t.profile_id,
            "examples": t.examples
        })

    tool_embeddings = get_cached_embeddings(tool_texts)
    client.upsert(
        collection_name="semantic_tools",
        points=[
            PointStruct(id=i, vector=embed, payload=payload)
            for i, (embed, payload) in enumerate(zip(tool_embeddings, tool_payloads))
        ],
    )

    logger.info(f"Ingested {len(profiles)} profiles and {len(tools)} tools into Qdrant.")


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    CATALOG_PATH = "catalog.json"  # adjust path if necessary
    ingest_catalog_to_qdrant(CATALOG_PATH)
