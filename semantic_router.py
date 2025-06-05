# semantic_router.py

from __future__ import annotations

import os
import time
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import openai

from qdrant_client import QdrantClient

# ==============================================================================
# CONFIGURATION & LOGGER
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_KEY:
    logger.error("OpenAI API key missing. Cannot start router.")
    raise SystemExit("❗️OpenAI key missing.")

openai.api_key = OPENAI_KEY

# ==============================================================================
# DATA CLASSES & ENUMS
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
    input_schema: Optional[str] = None


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RoutingResult:
    profile_id: str
    tool_id: str
    confidence: float
    confidence_level: ConfidenceLevel
    alternative_tools: List[Tuple[str, str, float]] = field(default_factory=list)


# ==============================================================================
# EMBEDDING CACHE & RATE LIMITER
# ==============================================================================
class EmbeddingCache:
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
# QDRANT CLIENT
# ==============================================================================
def get_qdrant_client(url: str, port: int, api_key: Optional[str] = None) -> QdrantClient:
    try:
        client = (
            QdrantClient(url=url, api_key=api_key, timeout=60)
            if url.startswith(("http://", "https://"))
            else QdrantClient(host=url, port=port, api_key=api_key, timeout=30, prefer_grpc=False)
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


# ==============================================================================
# SEMANTIC ROUTER CLASS
# ==============================================================================
class SemanticRouter:
    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        qdrant_port: int = QDRANT_PORT,
        qdrant_api_key: Optional[str] = QDRANT_API_KEY,
        high_thresh: float = 0.70,
        med_thresh: float = 0.50,
        top_k: int = 10,
    ):
        self.client = get_qdrant_client(qdrant_url, qdrant_port, qdrant_api_key)
        self.tools_collection = "semantic_tools"
        self.high_thresh = high_thresh
        self.med_thresh = med_thresh
        self.top_k = top_k

        # In-memory caches (populated via build())
        self.profiles: Dict[str, Profile] = {}
        self.tools: Dict[str, Tool] = {}
        self.tool_vectors: Dict[str, List[float]] = {}
        self.example_embeddings: Dict[str, Dict[str, List[float]]] = {}

        # Embedding dimension for "text-embedding-3-large"
        self.vector_size = 3072

    def build(self, catalog_path: str):
        """
        1) Load catalog (profiles + tools)
        2) Generate examples + embeddings for each tool (no Qdrant writes)
        3) Populate in-memory caches for routing
        """
        profiles, tools = load_catalog(catalog_path)
        if not profiles or not tools:
            logger.error("Catalog load failed. Cannot build router.")
            raise SystemExit(1)

        self.profiles = {p.id: p for p in profiles}
        self.tools = {t.id: t for t in tools}

        # Step 1: Build example embeddings
        all_examples: List[str] = []
        example_to_tool: Dict[str, str] = {}

        for t in tools:
            desc_words = re.findall(r"\b\w{5,}\b", t.description.lower())
            nouns = [w for w in desc_words if w not in {"about", "using", "based", "information"}]
            examples: List[str] = [f"Tell me about {t.name}"]
            if nouns:
                key = " ".join(nouns[:2])
                examples.append(f"Provide {key} information")
                examples.append(f"I need {key} details")
                verb = ["Show", "Get", "Analyze", "Display"][hash(t.id) % 4]
                examples.append(f"{verb} {nouns[0]} for AAPL stock")
                examples.append(f"{verb} latest {nouns[0]}")
            examples = examples[:5]

            self.example_embeddings[t.id] = {}
            for ex in examples:
                example_to_tool[ex] = t.id
                all_examples.append(ex)

        example_emb_list = get_cached_embeddings(all_examples)
        for ex, emb in zip(all_examples, example_emb_list):
            t_id = example_to_tool[ex]
            self.example_embeddings.setdefault(t_id, {})[ex] = emb

        # Step 2: Build tool vectors
        tool_texts: List[str] = []
        ordered_tool_ids: List[str] = []

        for t in tools:
            profile_ctx = ""
            prof = self.profiles.get(t.profile_id)
            if prof:
                profile_ctx = f"Part of {prof.name}: {prof.description}"

            enriched = (
                f"Tool: {t.name}\n\n"
                f"Description: {t.description}\n\n"
                f"{profile_ctx}\n\n"
                "Example use cases:\n"
                + "\n".join(f"- {ex}" for ex in self.example_embeddings[t.id].keys())
            )

            tool_texts.append(enriched)
            ordered_tool_ids.append(t.id)

        tool_emb_list = get_cached_embeddings(tool_texts)
        self.tool_vectors = {
            t_id: emb for t_id, emb in zip(ordered_tool_ids, tool_emb_list)
        }

        logger.info("Router build complete: in-memory caches are ready.")

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        v1, v2 = np.array(a), np.array(b)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1 / n1, v2 / n2))

    def route(self, query: str) -> RoutingResult:
        """
        1) Embed query
        2) Qdrant.search → top_k tool hits
        3) Compute: direct vector sim + Qdrant sim + example sim
        4) Weight → final score, pick top
        5) Compute confidence + alternatives
        """
        q_embed = get_cached_embeddings([query])[0]

        search_results = self.client.search(
            collection_name=self.tools_collection, query_vector=q_embed, limit=self.top_k
        )
        if not search_results:
            return RoutingResult(
                profile_id="web-search",
                tool_id="web_search_deep",
                confidence=40.0,
                confidence_level=ConfidenceLevel.LOW,
            )

        candidates: List[Dict[str, Any]] = []
        for hit in search_results:
            payload = hit.payload
            t_id = payload["id"]
            p_id = payload["profile_id"]
            qdrant_sim = 1.0 - hit.score
            direct_sim = self._cosine_similarity(q_embed, self.tool_vectors.get(t_id, []))

            ex_score = 0.0
            if t_id in self.example_embeddings:
                sims = [
                    self._cosine_similarity(q_embed, emb)
                    for emb in self.example_embeddings[t_id].values()
                ]
                ex_score = max(sims) if sims else 0.0

            final_sim = 0.6 * direct_sim + 0.4 * qdrant_sim + 0.2 * ex_score
            print("name: ", payload["name"])
            print(f"direct_sim: {direct_sim}")
            print(f"qdrant_sim: {qdrant_sim}")
            print(f"ex_score: {ex_score}")
            print(f"final_sim: {final_sim}")
            candidates.append({"tool_id": t_id, "profile_id": p_id, "score": final_sim})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        gap = top["score"] - second_score

        base_conf = int(top["score"] * 100)
        bonus = int(gap * 100)
        raw_conf = min(95, max(40, base_conf + bonus))
        if raw_conf >= self.high_thresh * 100:
            level = ConfidenceLevel.HIGH
        elif raw_conf >= self.med_thresh * 100:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        alternatives: List[Tuple[str, str, float]] = []
        if level != ConfidenceLevel.HIGH and len(candidates) > 1:
            for alt in candidates[1:4]:
                if alt["score"] > 0.30:
                    alt_conf = min(95, max(40, int(alt["score"] * 100)))
                    alternatives.append((alt["profile_id"], alt["tool_id"], alt_conf / 100.0))

        return RoutingResult(
            profile_id=top["profile_id"],
            tool_id=top["tool_id"],
            confidence=float(raw_conf),
            confidence_level=level,
            alternative_tools=alternatives,
        )


# ==============================================================================
# INTERACTIVE AGENTIC REPL
# ==============================================================================
def main():
    CATALOG_PATH = "catalog.json"  # Ensure this matches the file used by catalog_ingest.py

    # Initialize router and build in-memory caches
    router = SemanticRouter(
        qdrant_url=QDRANT_URL, qdrant_port=QDRANT_PORT, qdrant_api_key=QDRANT_API_KEY
    )
    router.build(CATALOG_PATH)

    print("Agentic RAG Router is ready. Type your query and press Enter.")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_query = input(">> ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye.")
            break
        if not user_query:
            continue

        # Perform routing
        try:
            result = router.route(user_query)
            print(f"\n→ Routed to Profile: {result.profile_id}")
            print(f"            Tool: {result.tool_id}")
            print(f"      Confidence: {result.confidence:.1f}% ({result.confidence_level.value.upper()})")

            if result.alternative_tools:
                print("\n  Alternative candidates considered:")
                for prof, tool, conf in result.alternative_tools:
                    print(f"    • {prof} → {tool} ({conf*100:.1f}%)")
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to route query: {e}\n")


if __name__ == "__main__":
    main()
