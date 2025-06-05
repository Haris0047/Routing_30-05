import os
import json
from routing_improved import (
    load_catalog_from_json,
    SemanticRouter,
    _OPENAI_KEY,
    _QDRANT_URL,
    _QDRANT_PORT,
    _QDRANT_API_KEY
)

def rag_query(query: str):
    # Load profiles and tools
    profiles, tools = load_catalog_from_json("catalog.json")
    if not profiles or not tools:
        print("Failed to load profiles and tools from catalog.json.")
        return

    # Initialize router
    router = SemanticRouter(
        qdrant_url=_QDRANT_URL,
        qdrant_port=_QDRANT_PORT,
        qdrant_api_key=_QDRANT_API_KEY
    )
    # Build router (if not already built)
    router.build(profiles, tools)

    # Route the query
    result = router.route(query)
    print(f"\n[ROUTER] Routed to: {result.profile_id} → {result.tool_id} (Confidence: {result.confidence:.1f}%, Level: {result.confidence_level.value})")
    if result.alternative_tools:
        print("[ALTERNATIVES]")
        for alt_profile, alt_tool, alt_conf in result.alternative_tools:
            print(f"  - {alt_profile} → {alt_tool} ({alt_conf*100:.1f}%)")

    # Simulate retrieval (mock)
    print(f"\n[MOCK RAG] Retrieved answer using tool '{result.tool_id}':")
    print(f"This is a mock answer for query: '{query}' using tool '{result.tool_id}'.")

if __name__ == "__main__":
    if not _OPENAI_KEY:
        raise SystemExit("❗️OpenAI key missing.")
    user_query = input("Enter your query: ")
    rag_query(user_query) 