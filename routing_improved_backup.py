from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import re
from enum import Enum

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Helper to fetch environment variables
# ---------------------------------------------------------------------------
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
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
    input_schema: str | None = None


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
    alternative_tools: List[Tuple[str, str, float]] = field(default_factory=list)  # [(profile_id, tool_id, confidence)]


# ---------------------------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""
    def __init__(self):
        self.cache = {}  # text -> embedding vector

    def get(self, text: str) -> List[float] | None:
        """Get embedding for text if it exists in cache."""
        return self.cache.get(text)

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding for text in cache."""
        self.cache[text] = embedding

    def get_or_add_batch(self, texts: List[str], embedding_func) -> List[List[float]]:
        """Get embeddings for a batch of texts, computing only what's needed."""
        # Check which texts are not in cache
        missing_texts = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            if text not in self.cache:
                missing_texts.append(text)
                missing_indices.append(i)
        
        # Get all cached embeddings first
        results = [None] * len(texts)
        for i, text in enumerate(texts):
            if text in self.cache:
                results[i] = self.cache[text]
                
        # Compute embeddings only for missing texts
        if missing_texts:
            new_embeddings = embedding_func(missing_texts)
            
            # Update cache with new embeddings
            for text, embedding in zip(missing_texts, new_embeddings):
                self.cache[text] = embedding
                
            # Insert new embeddings into results
            for i, embedding in zip(missing_indices, new_embeddings):
                results[i] = embedding
                
        return results


# Create a global embedding cache instance
_embedding_cache = EmbeddingCache()

# ---------------------------------------------------------------------------
# Catalog loading functions
# ---------------------------------------------------------------------------
def load_catalog_from_json(file_path: str) -> Tuple[List[Profile], List[Tool]]:
    """Load profiles and tools from the catalog JSON file."""
    try:
        with open(file_path, 'r') as f:
            catalog_data = json.load(f)
        
        profiles = [Profile(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            tools=[]  # We'll populate this later after loading tools
        ) for p in catalog_data.get("profiles", [])]
        
        profile_map = {p.id: p for p in profiles}
        
        tools = []
        for t in catalog_data.get("tools", []):
            tool = Tool(
                id=t["id"],
                profile_id=t["profile_id"],
                name=t["name"],
                description=t["description"],
                input_schema=t.get("input_schema")
            )
            tools.append(tool)
            
            # Add tool id to the corresponding profile's tools list
            if tool.profile_id in profile_map:
                profile_map[tool.profile_id].tools.append(tool.id)
        
        return profiles, tools
    except Exception as e:
        print(f"Error loading catalog from {file_path}: {e}")
        return [], []


def load_sample_queries_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load sample queries from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            queries_data = json.load(f)
        
        return queries_data.get("sample_queries", [])
    except Exception as e:
        print(f"Error loading sample queries from {file_path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Rate limiter for API calls
# ---------------------------------------------------------------------------
class APIRateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, min_time_between_calls: float = 0.5):
        self.min_time_between_calls = min_time_between_calls
        self.last_call_time = 0.0
        
    def wait_if_needed(self):
        """Wait if needed to respect the minimum time between calls."""
        import time
        
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_time_between_calls:
            wait_time = self.min_time_between_calls - time_since_last_call
            time.sleep(wait_time)
            
        self.last_call_time = time.time()


# Create a global rate limiter instance
_rate_limiter = APIRateLimiter(min_time_between_calls=0.5)

# ---------------------------------------------------------------------------
# OpenAI Embedding Helper
# ---------------------------------------------------------------------------
def get_embeddings(texts: List[str], max_retries: int = 5, retry_delay: int = 2) -> List[List[float]]:
    """Get OpenAI embeddings for a list of texts with retry logic."""
    import openai
    import time
    
    if not _OPENAI_KEY:
        raise EnvironmentError("OpenAI key missing – set via Colab userdata or env var.")
    
    openai.api_key = _OPENAI_KEY
    
    # Apply rate limiting
    _rate_limiter.wait_if_needed()
    
    # Use exponential backoff for retries
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"OpenAI API connection error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                print(f"Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed to get embeddings after {max_retries} attempts.")
                print("This could be due to network issues or OpenAI API rate limits.")
                raise

# Get embeddings with caching
def get_cached_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from cache or compute them if missing."""
    return _embedding_cache.get_or_add_batch(texts, get_embeddings)

# ---------------------------------------------------------------------------
# Qdrant initialization
# ---------------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-large"
VECTOR_SIZE = 3072  # Size for text-embedding-3-large model

def _get_qdrant_client(url: str = "localhost", port: int = 6333, api_key: str | None = None):
    """Initialize a Qdrant client."""
    # Handle full URLs (like http://74.208.122.216:6333)
    if url.startswith("http://") or url.startswith("https://"):
        # For remote URLs, use the full URL and set timeout
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,  # 60 second timeout for remote connections
        )
    else:
        # For localhost or IP without protocol, use host:port format
        return QdrantClient(
            host=url,
            port=port,
            api_key=api_key,
            timeout=30,  # 30 second timeout for local connections
            prefer_grpc=False
        )


def _get_qdrant_client_with_retry(url: str = "localhost", port: int = 6333, api_key: str | None = None, max_retries: int = 5, retry_delay: int = 2):
    """Initialize a Qdrant client with retry logic."""
    import time
    
    for attempt in range(max_retries):
        try:
            client = _get_qdrant_client(url=url, port=port, api_key=api_key)
            # Test the connection
            client.get_collections()
            print(f"Successfully connected to Qdrant at {url}:{port}")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Connection to Qdrant failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                print(f"Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed to connect to Qdrant after {max_retries} attempts.")
                print("Please ensure Qdrant is running, using Docker command:")
                print("docker run -p 6333:6333 -p 6334:6334 -v C:/path/to/Routing/qdrant_storage:/qdrant/storage qdrant/qdrant")
                raise


class SemanticRouter:
    """Pure semantic router for financial queries using embeddings."""

    def __init__(self, 
                 qdrant_url: str = "localhost", 
                 qdrant_port: int = 6333, 
                 qdrant_api_key: str | None = None,
                 high_confidence_threshold: float = 0.70,
                 medium_confidence_threshold: float = 0.50,
                 top_k_tools: int = 10):
        # Connect to Qdrant
        self.client = _get_qdrant_client_with_retry(
            url=qdrant_url, 
            port=qdrant_port, 
            api_key=qdrant_api_key, 
            max_retries=3, 
            retry_delay=5
        )
        
        # Collection names
        self.tools_collection = "semantic_tools"
        self.profiles_collection = "semantic_profiles"
        
        # Confidence thresholds
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.top_k_tools = top_k_tools
        
        # Store tools and profiles for lookup
        self.tools = {}
        self.profiles = {}
        
        # Store vectors for fast similarity computation
        self.tool_vectors = {}
        self.tool_examples = {}
        
        # Cache for example embeddings
        self.example_embeddings = {}  # tool_id -> {example_text -> embedding}

    def initialize_collections(self):
        """Create or recreate vector collections."""
        # Delete existing collections if they exist
        try:
            self.client.delete_collection(self.tools_collection)
            self.client.delete_collection(self.profiles_collection)
        except Exception:
            pass  # Collections might not exist
        
        # Create new collections
        self.client.create_collection(
            collection_name=self.tools_collection,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        self.client.create_collection(
            collection_name=self.profiles_collection,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        print(f"Initialized collections: {self.tools_collection}, {self.profiles_collection}")

    def build(self, profiles: List[Profile], tools: List[Tool]):
        """Build the semantic router database."""
        # Initialize collections
        self.initialize_collections()
        
        # Store profiles and tools for lookup
        self.profiles = {p.id: p for p in profiles}
        self.tools = {t.id: t for t in tools}
        
        # Process profiles
        print("Processing profiles...")
        profile_texts = []
        profile_points = []
        
        for i, profile in enumerate(profiles):
            # Create a rich semantic profile description
            profile_text = f"Profile: {profile.name}. {profile.description}"
            
            # Add tools info to profile for context
            tools_in_profile = [self.tools[tool_id] for tool_id in profile.tools if tool_id in self.tools]
            tool_names = [t.name for t in tools_in_profile]
            
            if tool_names:
                profile_text += f" Contains tools: {', '.join(tool_names)}."
            
            profile_texts.append(profile_text)
            profile_points.append({
                "id": profile.id,
                "name": profile.name,
                "description": profile.description
            })
        
        # Get profile embeddings
        profile_embeddings = get_cached_embeddings(profile_texts)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.profiles_collection,
            points=[
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                )
                for i, (embedding, payload) in enumerate(zip(profile_embeddings, profile_points))
            ]
        )
        
        # Process tools with rich semantic context
        print("Processing tools...")
        tool_texts = []
        tool_points = []
        tool_examples = {}
        
        for i, tool in enumerate(tools):
            # Generate high-quality examples for each tool
            examples = self._generate_semantic_examples(tool)
            tool_examples[tool.id] = examples
            
            # Create enriched tool description with metadata
            profile = self.profiles.get(tool.profile_id)
            profile_context = f"Part of {profile.name} profile: {profile.description}" if profile else ""
            
            # Build a rich semantic representation
            tool_text = f"""
Tool: {tool.name}

Description: {tool.description}

{profile_context}

Example use cases:
{chr(10).join(f"- {example}" for example in examples)}
            """
            
            # Save for embedding
            tool_texts.append(tool_text)
            tool_points.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "profile_id": tool.profile_id,
                "examples": examples
            })
        
        # Store example queries for lookup
        self.tool_examples = tool_examples
        
        # Get tool embeddings with caching
        tool_embeddings = get_cached_embeddings(tool_texts)
        
        # Store for fast similarity calculations
        for tool_id, embedding in zip([t.id for t in tools], tool_embeddings):
            self.tool_vectors[tool_id] = embedding
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.tools_collection,
            points=[
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                )
                for i, (embedding, payload) in enumerate(zip(tool_embeddings, tool_points))
            ]
        )
        
        # Pre-compute and cache embeddings for all examples
        print("Pre-computing example embeddings...")
        all_examples = []
        example_tool_map = {}
        
        for tool_id, examples in tool_examples.items():
            for example in examples:
                all_examples.append(example)
                example_tool_map[example] = tool_id
        
        # Get all example embeddings in one batch
        example_embeddings = get_cached_embeddings(all_examples)
        
        # Organize in tool-specific caches
        self.example_embeddings = {}
        for example, embedding in zip(all_examples, example_embeddings):
            tool_id = example_tool_map[example]
            if tool_id not in self.example_embeddings:
                self.example_embeddings[tool_id] = {}
            self.example_embeddings[tool_id][example] = embedding
        
        print("Build complete!")

    def _generate_semantic_examples(self, tool: Tool) -> List[str]:
        """Generate high-quality example queries for a tool based on its name and description."""
        # Extract the core purpose of the tool
        tool_name = tool.name.lower()
        description = tool.description.lower()
        
        # Generate focused examples without relying on hardcoded patterns
        examples = []
        
        # Example 1: Direct information request
        examples.append(f"Tell me about {tool.name}")
        
        # Example 2-3: Extract key topics from description
        words = description.split()
        nouns = []
        
        # Simple noun extraction (can be improved with NLP)
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            if len(clean_word) > 4 and clean_word not in {"about", "these", "those", "them", "their", "other", "using", "based"}:
                nouns.append(clean_word)
        
        if nouns:
            key_topics = " ".join(nouns[:2])  # Use first 2 nouns as key topics
            examples.append(f"Provide {key_topics} information")
            examples.append(f"I need {key_topics} for my research")
            
        # Example 4-5: Action-oriented examples
        action_verbs = ["Show", "Analyze", "Get", "Display", "Calculate"]
        if nouns:
            action = action_verbs[hash(tool.id) % len(action_verbs)]  # Deterministic selection
            examples.append(f"{action} {nouns[0]} for AAPL stock")
            examples.append(f"{action} latest {nouns[0]}")
            
        return examples[:5]  # Return up to 5 examples

    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Convert to numpy arrays and normalize
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        return np.dot(v1 / norm1, v2 / norm2)

    def route(self, query: str) -> RoutingResult:
        """Route a query to the most appropriate tool using semantic similarity."""
        # Get query embedding with caching
        query_embedding = get_cached_embeddings([query])[0]
        
        # Find semantically similar tools
        search_results = self.client.search(
            collection_name=self.tools_collection,
            query_vector=query_embedding,
            limit=self.top_k_tools  # Return top matches
        )
        
        if not search_results:
            # Handle case with no results
            return RoutingResult(
                profile_id="web-search",
                tool_id="web_search_deep",
                confidence=50.0,
                confidence_level=ConfidenceLevel.LOW,
                alternative_tools=[]
            )
        
        # Process search results to determine best tool
        candidates = []
        
        for hit in search_results:
            tool_id = hit.payload["id"]
            profile_id = hit.payload["profile_id"]
            initial_score = 1.0 - hit.score  # Convert distance to similarity
            
            # Extract examples for this tool
            examples = hit.payload.get("examples", [])
            
            # Compare query directly to example queries for this tool using cached embeddings
            example_score = 0.0
            if examples and tool_id in self.example_embeddings:
                example_similarities = []
                
                # Use pre-computed example embeddings
                for example in examples:
                    if example in self.example_embeddings[tool_id]:
                        example_embedding = self.example_embeddings[tool_id][example]
                        sim = self._compute_similarity(query_embedding, example_embedding)
                        example_similarities.append(sim)
                
                # Use max similarity as example score
                example_score = max(example_similarities) if example_similarities else 0.0
            
            # Compute additional score component - direct vector similarity
            direct_similarity = 0.0
            if tool_id in self.tool_vectors:
                tool_vector = self.tool_vectors[tool_id]
                direct_similarity = self._compute_similarity(query_embedding, tool_vector)
            
            # Calculate final score as weighted combination
            final_score = (
                0.5 * direct_similarity +  # Direct vector similarity
                0.3 * initial_score +      # Qdrant search score
                0.2 * example_score        # Example query similarity
            )
            print("query", query)
            print("direct_similarity", direct_similarity)
            print("initial_score", initial_score)
            print("example_score", example_score)
            print("final_score", final_score)
            
            # Add to candidates
            candidates.append({
                "tool_id": tool_id,
                "profile_id": profile_id,
                "score": final_score
            })
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top candidate
        top_candidate = candidates[0]
        top_score = top_candidate["score"]
        
        # Calculate confidence based on score distribution
        if len(candidates) > 1:
            score_gap = top_score - candidates[1]["score"]  # Gap to second best
            confidence = min(95, max(40, int(top_score * 100) + int(score_gap * 100)))
        else:
            confidence = min(95, max(40, int(top_score * 100)))
        
        # Determine confidence level
        if confidence >= self.high_confidence_threshold * 100:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= self.medium_confidence_threshold * 100:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        # Get alternative tools (up to 3 with reasonable confidence)
        alternative_tools = []
        for candidate in candidates[1:4]:  # Next 3 candidates
            if candidate["score"] > 0.3:  # Reasonable confidence threshold
                alt_confidence = min(95, max(40, int(candidate["score"] * 100)))
                alternative_tools.append((candidate["profile_id"], candidate["tool_id"], alt_confidence / 100.0))
        
        print(top_candidate)


        # Create result
        return RoutingResult(
            profile_id=top_candidate["profile_id"],
            tool_id=top_candidate["tool_id"],
            confidence=float(confidence),
            confidence_level=confidence_level,
            alternative_tools=alternative_tools
        )


# ---------------------------------------------------------------------------
# Evaluation and Demo
# ---------------------------------------------------------------------------
def _execute_tool(result: RoutingResult, query: str):
    """Execute the selected tool and show confidence information."""
    confidence_text = {
        ConfidenceLevel.HIGH: "HIGH CONFIDENCE",
        ConfidenceLevel.MEDIUM: "MEDIUM CONFIDENCE",
        ConfidenceLevel.LOW: "LOW CONFIDENCE"
    }

    # Primary tool execution
    print(f"[EXECUTE] {result.profile_id} → {result.tool_id} | '{query}' ({confidence_text[result.confidence_level]}: {result.confidence:.1f}%)")

    # Alternative tools if confidence is low
    if result.confidence_level != ConfidenceLevel.HIGH and result.alternative_tools:
        print(f"[ALTERNATIVES CONSIDERED]")
        for alt_profile, alt_tool, alt_conf in result.alternative_tools:
            print(f"  - {alt_profile} → {alt_tool} ({alt_conf*100:.1f}%)")


def evaluate(router: SemanticRouter, sample_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate router performance on sample queries."""
    if not sample_queries:
        print("No sample queries to evaluate.")
        return {}
    
    # Tracking metrics
    total_queries = len(sample_queries)
    correct_tool_count = 0
    correct_profile_count = 0
    correct_both_count = 0
    total_time = 0
    confidence_sum = 0
    
    # Store detailed results
    results = []
    
    # Process each query
    for sample in sample_queries:
        query = sample["query"]
        expected = sample["expected"]
        
        print("\nQuery:", query)
        
        try:
            # Measure routing time
            start_time = time.time()
            result = router.route(query)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            confidence_sum += result.confidence
            
            # Check accuracy
            tool_correct = result.tool_id == expected["tool_id"]
            profile_correct = result.profile_id == expected["profile_id"]
            
            if tool_correct:
                correct_tool_count += 1
            if profile_correct:
                correct_profile_count += 1
            if tool_correct and profile_correct:
                correct_both_count += 1
            
            # Store detailed result
            results.append({
                "query": query,
                "expected_tool": expected["tool_id"],
                "expected_profile": expected["profile_id"],
                "routed_tool": result.tool_id,
                "routed_profile": result.profile_id,
                "tool_correct": tool_correct,
                "profile_correct": profile_correct,
                "confidence": result.confidence,
                "time_seconds": query_time
            })
            
            # Print result
            accuracy = "✓" if tool_correct and profile_correct else "✗"
            _execute_tool(result, query)
            print(f"[EXPECTED] {expected['profile_id']} → {expected['tool_id']}")
            print(f"[ACCURACY] {accuracy} | Time: {query_time:.3f}s")
        except Exception as e:
            print(f"[ERROR] Failed to process query: {query}")
            print(f"Error details: {e}")
            # Record error
            results.append({
                "query": query,
                "expected_tool": expected["tool_id"],
                "expected_profile": expected["profile_id"],
                "error": str(e)
            })
            print("Continuing with next query...\n")
            continue
    
    # Calculate summary metrics
    avg_time = total_time / total_queries if total_queries > 0 else 0
    avg_confidence = confidence_sum / total_queries if total_queries > 0 else 0
    
    tool_accuracy = (correct_tool_count / total_queries) * 100 if total_queries > 0 else 0
    profile_accuracy = (correct_profile_count / total_queries) * 100 if total_queries > 0 else 0
    full_accuracy = (correct_both_count / total_queries) * 100 if total_queries > 0 else 0
    
    # Display summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Queries: {total_queries}")
    print(f"Average Processing Time: {avg_time:.3f} seconds")
    print(f"Average Confidence: {avg_confidence:.1f}%")
    print(f"Tool Accuracy: {tool_accuracy:.1f}% ({correct_tool_count}/{total_queries})")
    print(f"Profile Accuracy: {profile_accuracy:.1f}% ({correct_profile_count}/{total_queries})")
    print(f"Full Accuracy: {full_accuracy:.1f}% ({correct_both_count}/{total_queries})")
    
    # Return summary and details
    return {
        "summary": {
            "total_queries": total_queries,
            "avg_time": avg_time,
            "avg_confidence": avg_confidence,
            "tool_accuracy": tool_accuracy,
            "profile_accuracy": profile_accuracy,
            "full_accuracy": full_accuracy
        },
        "detailed_results": results
    }


def main():
    """Main function for running the semantic router."""
    if not _OPENAI_KEY:
        raise SystemExit("❗️OpenAI key missing.")

    # Load profiles and tools from catalog
    profiles, tools = load_catalog_from_json("catalog.json")
    
    # Load sample queries
    sample_queries = load_sample_queries_from_json("sample_queries.json")
    
    # Check if data was loaded successfully
    if not profiles or not tools:
        print("Failed to load profiles and tools from catalog.json.")
        return
    
    if not sample_queries:
        print("Failed to load sample queries from sample_queries.json.")
        return

    # Initialize router
    router = SemanticRouter(
        qdrant_url=_QDRANT_URL,
        qdrant_port=_QDRANT_PORT,
        qdrant_api_key=_QDRANT_API_KEY
    )
    
    # Build router
    router.build(profiles, tools)
    
    # Evaluate performance
    evaluation_results = evaluate(router, sample_queries)
    
    # Save results
    try:
        with open("semantic_routing_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        print("Detailed results saved to semantic_routing_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main() 