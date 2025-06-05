from __future__ import annotations

import os
import json
import time
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

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
    logger.error("OpenAI API key missing. Exiting.")
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
        idx_map = {}
        results: List[Optional[List[float]]] = [None] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                idx_map[len(to_compute)] = i
                to_compute.append(text)

        if to_compute:
            new_embeds = _fetch_embeddings_with_backoff(to_compute)
            for i, embed in enumerate(new_embeds):
                orig_index = idx_map[i]
                results[orig_index] = embed
                self._cache[texts[orig_index]] = embed

        # By now, every slot in results is non-None
        return [r for r in results]  # type: ignore

_embedding_cache = EmbeddingCache()

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

def _fetch_embeddings_with_backoff(texts: List[str], max_retries: int = 5, base_delay: float = 2.0) -> List[List[float]]:
    _rate_limiter.throttle()
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(model="text-embedding-3-large", input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"OpenAI embedding error (retry {attempt + 1}/{max_retries}): {e}. Waiting {delay}s.")
                time.sleep(delay)
            else:
                logger.error("Failed to retrieve embeddings after retries.")
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
        profiles = [Profile(id=p["id"], name=p["name"], description=p["description"]) for p in raw.get("profiles", [])]
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
        logger.error(f"Failed to load catalog: {e}")
        return [], []

def load_sample_queries(file_path: str) -> List[Dict[str, Any]]:
    try:
        data = json.load(open(file_path, "r"))
        return data.get("sample_queries", [])
    except Exception as e:
        logger.error(f"Failed to load sample queries: {e}")
        return []

# ==============================================================================
# QDRANT CLIENT MANAGEMENT
# ==============================================================================
def get_qdrant_client_with_retry(url: str, port: int, api_key: Optional[str] = None, retries: int = 5) -> QdrantClient:
    for attempt in range(retries):
        try:
            client = (QdrantClient(url=url, api_key=api_key, timeout=60)
                      if url.startswith(("http://", "https://"))
                      else QdrantClient(host=url, port=port, api_key=api_key, timeout=30, prefer_grpc=False))
            client.get_collections()  # quick connectivity check
            logger.info(f"Connected to Qdrant at {url}:{port}")
            return client
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Qdrant connection failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s.")
                time.sleep(wait)
            else:
                logger.error("Exceeded Qdrant connection retries.")
                raise
    raise RuntimeError("Unreachable: Qdrant init failed.")

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
        self.client = get_qdrant_client_with_retry(qdrant_url, qdrant_port, qdrant_api_key)
        self.tools_collection = "semantic_tools"
        self.profiles_collection = "semantic_profiles"
        self.high_thresh = high_thresh
        self.med_thresh = med_thresh
        self.top_k = top_k

        self.profiles: Dict[str, Profile] = {}
        self.tools: Dict[str, Tool] = {}
        self.tool_vectors: Dict[str, List[float]] = {}
        self.tool_examples: Dict[str, List[str]] = {}
        self.example_embeddings: Dict[str, Dict[str, List[float]]] = {}

        # Embed dimension hardcoded for text-embedding-3-large
        self.vector_size = 3072

    def _cleanup_and_create_collections(self):
        for col in (self.tools_collection, self.profiles_collection):
            try:
                self.client.delete_collection(col)
                logger.debug(f"Deleted existing collection: {col}")
            except Exception:
                pass
        self.client.create_collection(
            collection_name=self.profiles_collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        self.client.create_collection(
            collection_name=self.tools_collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info("Initialized Qdrant collections for profiles and tools.")

    def build(self, profiles: List[Profile], tools: List[Tool]):
        # STEP 1: Clean and recreate collections
        self._cleanup_and_create_collections()
        self.profiles = {p.id: p for p in profiles}
        self.tools = {t.id: t for t in tools}

        # STEP 2: Embed & upload profiles
        logger.info("Embedding profiles...")
        profile_texts, profile_payloads = [], []
        for p in profiles:
            text = f"Profile: {p.name}. {p.description}"
            tools_in_profile = [self.tools[tid].name for tid in p.tools if tid in self.tools]
            if tools_in_profile:
                text += f" Contains tools: {', '.join(tools_in_profile)}."
            profile_texts.append(text)
            profile_payloads.append({"id": p.id, "name": p.name, "description": p.description})

        profile_embeds = get_cached_embeddings(profile_texts)
        self.client.upsert(
            collection_name=self.profiles_collection,
            points=[
                PointStruct(id=i, vector=embed, payload=payload)
                for i, (embed, payload) in enumerate(zip(profile_embeds, profile_payloads))
            ],
        )

        # STEP 3: Embed & upload tools
        logger.info("Generating examples and embedding tools...")
        tool_texts, tool_payloads = [], []
        for t in tools:
            examples = self._create_tool_examples(t)
            self.tool_examples[t.id] = examples

            profile_ctx = ""
            if t.profile_id in self.profiles:
                p = self.profiles[t.profile_id]
                profile_ctx = f"Part of {p.name}: {p.description}"

            enriched = (
                f"Tool: {t.name}\n\n"
                f"Description: {t.description}\n\n"
                f"{profile_ctx}\n\n"
                "Example use cases:\n"
                + "\n".join(f"- {ex}" for ex in examples)
            )

            tool_texts.append(enriched)
            tool_payloads.append({"id": t.id, "name": t.name, "description": t.description, "profile_id": t.profile_id, "examples": examples})

        tool_embeds = get_cached_embeddings(tool_texts)
        for t_id, embed in zip([t.id for t in tools], tool_embeds):
            self.tool_vectors[t_id] = embed

        self.client.upsert(
            collection_name=self.tools_collection,
            points=[
                PointStruct(id=i, vector=embed, payload=payload)
                for i, (embed, payload) in enumerate(zip(tool_embeds, tool_payloads))
            ],
        )

        # STEP 4: Precompute example embeddings
        logger.info("Precomputing example embeddings...")
        all_examples = []
        example_to_tool: Dict[str, str] = {}
        for t_id, ex_list in self.tool_examples.items():
            for ex in ex_list:
                example_to_tool[ex] = t_id
                all_examples.append(ex)

        example_embeds = get_cached_embeddings(all_examples)
        for ex, embed in zip(all_examples, example_embeds):
            t_id = example_to_tool[ex]
            self.example_embeddings.setdefault(t_id, {})[ex] = embed

        logger.info("SemanticRouter build completed.")

    def _create_tool_examples(self, tool: Tool) -> List[str]:
        desc_words = re.findall(r"\b\w{5,}\b", tool.description.lower())
        nouns = [w for w in desc_words if w not in {"about", "using", "based", "information"}]
        examples: List[str] = []

        # Always include basic “tell me about” query
        examples.append(f"Tell me about {tool.name}")

        if nouns:
            key = " ".join(nouns[:2])
            examples.append(f"Provide {key} information")
            examples.append(f"I need {key} details")
            verb = ["Show", "Get", "Analyze", "Display"][hash(tool.id) % 4]
            examples.append(f"{verb} {nouns[0]} for AAPL stock")
            examples.append(f"{verb} latest {nouns[0]}")
        return examples[:5]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        v1, v2 = np.array(a), np.array(b)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if not norm1 or not norm2:
            return 0.0
        return float(np.dot(v1 / norm1, v2 / norm2))

    def route(self, query: str) -> RoutingResult:
        # Embed query
        q_embed = get_cached_embeddings([query])[0]

        # Retrieve top-K tools
        hits = self.client.search(collection_name=self.tools_collection, query_vector=q_embed, limit=self.top_k)
        if not hits:
            return RoutingResult("web-search", "web_search_deep", 0.40, ConfidenceLevel.LOW, [])

        # Score each hit: combine direct vector sim, Qdrant similarity, and example sim
        candidates: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload
            t_id = payload["id"]
            p_id = payload["profile_id"]
            qdrant_sim = 1.0 - hit.score
            direct_sim = self._cosine_similarity(q_embed, self.tool_vectors.get(t_id, []))

            # Example-based scoring
            ex_score = 0.0
            if t_id in self.example_embeddings:
                ex_sims = [self._cosine_similarity(q_embed, emb) for emb in self.example_embeddings[t_id].values()]
                ex_score = max(ex_sims) if ex_sims else 0.0

            final_sim = 0.5 * direct_sim + 0.3 * qdrant_sim + 0.2 * ex_score
            candidates.append({"tool_id": t_id, "profile_id": p_id, "score": final_sim})

        # Rank
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        score_gap = top["score"] - second_score

        # Confidence calculation
        base_conf = int(top["score"] * 100)
        gap_bonus = int(score_gap * 100)
        raw_conf = min(95, max(40, base_conf + gap_bonus))
        if raw_conf >= self.high_thresh * 100:
            level = ConfidenceLevel.HIGH
        elif raw_conf >= self.med_thresh * 100:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        # Gather alternatives if score is low
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
# EVALUATION
# ==============================================================================
def _execute_tool(result: RoutingResult, query: str):
    badge = {
        ConfidenceLevel.HIGH: "HIGH CONFIDENCE",
        ConfidenceLevel.MEDIUM: "MEDIUM CONFIDENCE",
        ConfidenceLevel.LOW: "LOW CONFIDENCE",
    }
    logger.info(f"[EXECUTE] {result.profile_id} → {result.tool_id} | '{query}' ({badge[result.confidence_level]}: {result.confidence:.1f}%)")
    if result.confidence_level != ConfidenceLevel.HIGH and result.alternative_tools:
        logger.info("[ALTERNATIVES]")
        for prof, tool, conf in result.alternative_tools:
            logger.info(f"  - {prof} → {tool} ({conf * 100:.1f}%)")

def evaluate(router: SemanticRouter, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        logger.warning("No sample queries provided.")
        return {"summary": {}, "detailed_results": []}

    total = len(samples)
    correct_tool = correct_prof = correct_both = 0
    total_time = total_conf = 0.0
    records: List[Dict[str, Any]] = []

    for entry in samples:
        q = entry["query"]
        exp = entry["expected"]
        start = time.time()
        try:
            result = router.route(q)
            duration = time.time() - start
            total_time += duration
            total_conf += result.confidence

            tool_ok = result.tool_id == exp["tool_id"]
            prof_ok = result.profile_id == exp["profile_id"]
            if tool_ok: correct_tool += 1
            if prof_ok: correct_prof += 1
            if tool_ok and prof_ok: correct_both += 1

            records.append({
                "query": q,
                "expected_tool": exp["tool_id"],
                "expected_profile": exp["profile_id"],
                "routed_tool": result.tool_id,
                "routed_profile": result.profile_id,
                "tool_correct": tool_ok,
                "profile_correct": prof_ok,
                "confidence": result.confidence,
                "time_seconds": round(duration, 4),
            })

            _execute_tool(result, q)
            logger.info(f"[EXPECTED] {exp['profile_id']} → {exp['tool_id']}")
            logger.info(f"[RESULT] {'✓' if tool_ok and prof_ok else '✗'} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start
            logger.error(f"[ERROR] Query processing failed: {e}")
            records.append({
                "query": q,
                "expected_tool": exp["tool_id"],
                "expected_profile": exp["profile_id"],
                "error": str(e),
                "time_seconds": round(duration, 4),
            })

    avg_time = total_time / total
    avg_conf = total_conf / total
    summary = {
        "total_queries": total,
        "avg_time": round(avg_time, 4),
        "avg_confidence": round(avg_conf, 1),
        "tool_accuracy": round((correct_tool / total) * 100, 1),
        "profile_accuracy": round((correct_prof / total) * 100, 1),
        "full_accuracy": round((correct_both / total) * 100, 1),
    }

    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    for k, v in summary.items():
        logger.info(f"{k}: {v}")

    return {"summary": summary, "detailed_results": records}

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    profiles, tools = load_catalog("catalog.json")
    if not profiles or not tools:
        logger.error("Catalog loading failed. Exiting.")
        return

    samples = load_sample_queries("sample_queries.json")
    if not samples:
        logger.error("Sample queries loading failed. Exiting.")
        return

    router = SemanticRouter(qdrant_url=QDRANT_URL, qdrant_port=QDRANT_PORT, qdrant_api_key=QDRANT_API_KEY)
    router.build(profiles, tools)
    results = evaluate(router, samples)

    try:
        with open("semantic_routing_results.json", "w") as fout:
            json.dump(results, fout, indent=2)
        logger.info("Results saved to semantic_routing_results.json")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
