import json
import requests
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict, Any
import os
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

class APIRankingLearner:
    def __init__(self, performance_file="api_performance.pkl"):
        self.performance_file = performance_file
        self.api_performance = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total_queries': 0})
        self.intent_api_success = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'wrong': 0}))
        self.query_history = []
        self.load_performance()
    
    def load_performance(self):
        """Load historical performance data"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'rb') as f:
                    data = pickle.load(f)
                    self.api_performance = data.get('api_performance', self.api_performance)
                    self.intent_api_success = data.get('intent_api_success', self.intent_api_success)
                    self.query_history = data.get('query_history', [])
                print(f"📚 Loaded performance data from {self.performance_file}")
        except Exception as e:
            print(f"⚠️  Could not load performance data: {e}")
    
    def save_performance(self):
        """Save performance data to file"""
        try:
            data = {
                'api_performance': dict(self.api_performance),
                'intent_api_success': dict(self.intent_api_success),
                'query_history': self.query_history[-1000:]  # Keep last 1000 queries
            }
            with open(self.performance_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved performance data to {self.performance_file}")
        except Exception as e:
            print(f"⚠️  Could not save performance data: {e}")
    
    def record_feedback(self, query: str, intent: str, selected_api: str, api_candidates: list, was_correct: bool = None):
        """Record feedback on API selection"""
        timestamp = datetime.now().isoformat()
        
        # Auto-determine correctness based on intent and API name
        if was_correct is None:
            was_correct = self.evaluate_api_selection(intent, selected_api)
        
        # Record general API performance
        key = selected_api
        self.api_performance[key]['total_queries'] += 1
        if was_correct:
            self.api_performance[key]['correct'] += 1
        else:
            self.api_performance[key]['wrong'] += 1
        
        # Record intent-specific performance
        intent_key = f"{intent}_{selected_api}"
        if was_correct:
            self.intent_api_success[intent][selected_api]['correct'] += 1
        else:
            self.intent_api_success[intent][selected_api]['wrong'] += 1
        
        # Record query history
        self.query_history.append({
            'timestamp': timestamp,
            'query': query,
            'intent': intent,
            'selected_api': selected_api,
            'candidates': api_candidates,
            'was_correct': was_correct
        })
        
        # Save every 10 queries
        if len(self.query_history) % 10 == 0:
            self.save_performance()
    
    def evaluate_api_selection(self, intent: str, selected_api: str) -> bool:
        """Evaluate if the API selection was appropriate for the intent"""
        # Define correct API patterns for different intents
        correct_apis = {
            'current_price': [
                'Stock Quote Short API', 'Stock Quote API', 'Real Time Price API',
                'Quote Short', 'Quote', 'Real Time', 'Current Price'
            ],
            'historical_data': [
                'Historical Stock Data API', 'Chart Data API', 'Historical Price API',
                'Historical', 'Chart', 'Price History'
            ],
            'financial_statements': [
                'Income Statement API', 'Balance Sheet API', 'Cash Flow API',
                'Financial Statements', 'Income', 'Balance', 'Cash Flow'
            ],
            'company_info': [
                'Company Profile API', 'Company Information API', 'Profile API',
                'Profile', 'Company', 'Information'
            ],
            'earnings': [
                'Earnings Report API', 'Earnings Calendar API', 'Earnings API',
                'Earnings', 'Report'
            ],
            'dividends': [
                'Dividends Company API', 'Dividend API', 'Historical Dividends API',
                'Dividends', 'Dividend'
            ]
        }
        
        if intent not in correct_apis:
            return True  # Unknown intent, assume correct
        
        # Check if selected API matches any correct pattern
        selected_lower = selected_api.lower()
        for correct_api in correct_apis[intent]:
            if any(word in selected_lower for word in correct_api.lower().split()):
                return True
        
        return False
    
    def get_learning_boost(self, api_name: str, intent: str) -> float:
        """Get boost/penalty based on historical performance"""
        # Intent-specific performance
        intent_perf = self.intent_api_success[intent][api_name]
        intent_total = intent_perf['correct'] + intent_perf['wrong']
        
        if intent_total >= 3:  # Need minimum samples for intent-specific learning
            intent_success_rate = intent_perf['correct'] / intent_total
            intent_boost = (intent_success_rate - 0.5) * 0.3  # Max ±0.15 boost
        else:
            intent_boost = 0
        
        # General API performance
        general_perf = self.api_performance[api_name]
        general_total = general_perf['correct'] + general_perf['wrong']
        
        if general_total >= 5:  # Need minimum samples for general learning
            general_success_rate = general_perf['correct'] / general_total
            general_boost = (general_success_rate - 0.5) * 0.2  # Max ±0.1 boost
        else:
            general_boost = 0
        
        return intent_boost + general_boost
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {
            'total_queries': len(self.query_history),
            'api_performance': dict(self.api_performance),
            'intent_breakdown': {}
        }
        
        # Intent breakdown
        for intent, apis in self.intent_api_success.items():
            intent_stats = {}
            for api, perf in apis.items():
                total = perf['correct'] + perf['wrong']
                if total > 0:
                    success_rate = perf['correct'] / total
                    intent_stats[api] = {
                        'success_rate': round(success_rate, 3),
                        'total_queries': total
                    }
            stats['intent_breakdown'][intent] = intent_stats
        
        return stats

class FMPAPIRag:
    def __init__(self, qdrant_url, fmp_api_key, openai_api_key):
        # Initialize clients
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = "tools"
        self.fmp_api_key = fmp_api_key
        
        # Initialize OpenAI
        openai.api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize learning system
        self.learner = APIRankingLearner()
        
        # Create collection if it doesn't exist
        self.setup_collection()
    
    def setup_collection(self):
        """Setup Qdrant collection for FMP APIs with OpenAI embedding dimensions"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # OpenAI text-embedding-3-small has 1536 dimensions
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"ℹ️  Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"❌ Error setting up collection: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI's text-embedding-3-small model"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return [0.0] * 1536  # Return zero vector as fallback
    
    def insert_api(self, api_data: Dict[str, Any]):
        """Insert a single API into the vector database"""
        # Create rich text for embedding
        embedding_text = f"""
        API Name: {api_data['name']}
        Description: {api_data['description']}
        Category: {api_data['category']}
        Use Cases: {', '.join(api_data['use_cases'])}
        Required Parameters: {', '.join(api_data['parameters']['required'])}
        Optional Parameters: {', '.join(api_data['parameters']['optional'])}
        """
        
        # Generate embedding using OpenAI
        embedding = self.create_embedding(embedding_text)
        
        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=api_data
        )
        
        # Insert into Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        print(f"✅ Inserted API: {api_data['name']}")
    
    def apply_intent_boost(self, results: list, intent: str) -> list:
        """Apply intent-based boosts to search results"""
        intent_boosts = {
            'current_price': {
                'positive_keywords': ['quote', 'price', 'real', 'current', 'live', 'short'],
                'negative_keywords': ['earnings', 'dividend', 'estimate', 'historical', 'news'],
                'boost': 0.15,
                'penalty': -0.10
            },
            'historical_data': {
                'positive_keywords': ['historical', 'chart', 'data', 'price', 'history'],
                'negative_keywords': ['current', 'real', 'live', 'quote', 'earnings'],
                'boost': 0.12,
                'penalty': -0.08
            },
            'earnings': {
                'positive_keywords': ['earnings', 'report', 'financial'],
                'negative_keywords': ['quote', 'price', 'dividend', 'current'],
                'boost': 0.12,
                'penalty': -0.08
            }
        }
        
        if intent not in intent_boosts:
            return results
        
        boost_config = intent_boosts[intent]
        
        for result in results:
            api_name_lower = result['api']['name'].lower()
            
            # Apply positive boost
            if any(keyword in api_name_lower for keyword in boost_config['positive_keywords']):
                result['score'] += boost_config['boost']
            
            # Apply penalty
            elif any(keyword in api_name_lower for keyword in boost_config['negative_keywords']):
                result['score'] += boost_config['penalty']
        
        return results
    
    def apply_learning_boost(self, results: list, intent: str) -> list:
        """Apply learning-based boosts to search results"""
        for result in results:
            api_name = result['api']['name']
            learning_boost = self.learner.get_learning_boost(api_name, intent)
            result['score'] += learning_boost
            
            # Debug info
            if learning_boost != 0:
                print(f"🧠 Learning boost for {api_name}: {learning_boost:+.3f}")
        
        return results
    
    def rerank_results(self, results: list, intent: str) -> list:
        """Re-rank results using multiple signals"""
        # Apply intent-based boosts
        results = self.apply_intent_boost(results, intent)
        
        # Apply learning-based boosts
        results = self.apply_learning_boost(results, intent)
        
        # Sort by updated scores
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return results
    
    def search_apis(self, query: str, top_k: int = 5, n_paraphrases: int = 2) -> list[dict]:
        """Enhanced search with learning-based re-ranking"""
        # Generate paraphrases (existing code)
        try:
            gpt_prompt = (
                f"Generate {n_paraphrases} alternative phrasings for this financial data query. "
                "Each variant should:\n"
                "1. Maintain the exact same intent and information\n"
                "2. Use clear, professional financial terminology\n"
                "3. Be concise (under 15 words)\n"
                "4. Be suitable for a financial API search\n\n"
                f"Original query: \"{query}\"\n\n"
                "Return only the variants, one per line, with no additional text."
            )
            para_resp = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a paraphrasing engine."},
                    {"role": "user", "content": gpt_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            paraphrases = [
                line.strip("•- ").strip()
                for line in para_resp.choices[0].message.content.split("\n")
                if line.strip()
            ][:n_paraphrases]
        except Exception as e:
            print(f"⚠️  GPT paraphrase error, falling back to original only → {e}")
            paraphrases = []

        variants = [query] + paraphrases
        print("variants: ", variants)
        
        # Create embeddings and search
        vectors = [self.create_embedding(v) for v in variants]
        merged: dict[str, dict] = {}
        recall_k = max(top_k * 3, 15)

        for vec in vectors:
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=recall_k,
                with_payload=True
            )
            for h in hits:
                tool_id = h.payload.get("id", h.id)
                if (tool_id not in merged) or (h.score > merged[tool_id]["score"]):
                    merged[tool_id] = {"score": h.score, "api": h.payload}

        # Get top results
        top_hits = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k * 2]
        
        # Store paraphrases for debugging
        self._last_paraphrases = variants
        
        return top_hits[:top_k]
    
    def extract_parameters_with_gpt(self, query: str, api_data: Dict) -> Dict:
        """Use GPT-4o to extract parameters from natural language query"""
        try:
            prompt = f"""
            Extract API parameters from the user's natural language query.
            
            API Information:
            - Name: {api_data['name']}
            - Description: {api_data['description']}
            - Required Parameters: {api_data['parameters']['required']}
            - Optional Parameters: {api_data['parameters']['optional']}
            
            User Query: "{query}"
            
            Instructions:
            1. Extract stock symbols from company names (e.g., "Apple" -> "AAPL", "Tesla" -> "TSLA")
            2. Extract date ranges if mentioned (e.g., "last 30 days", "this month")
            3. Return only the parameters that are relevant to this API
            4. Use standard stock symbols (AAPL, TSLA, MSFT, GOOGL, AMZN, META, NVDA, etc.)
            
            Return ONLY a JSON object with the extracted parameters. No explanation needed.
            
            Example formats:
            {{"symbol": "AAPL"}}
            {{"symbol": "TSLA", "from": "2024-01-01", "to": "2024-01-31"}}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting API parameters from natural language queries. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            params_text = response.choices[0].message.content.strip()
            if params_text.startswith('```json'):
                params_text = params_text.replace('```json', '').replace('```', '').strip()
            
            extracted_params = json.loads(params_text)
            return extracted_params
            
        except Exception as e:
            print(f"⚠️  Error extracting parameters with GPT: {e}")
            return self.simple_parameter_extraction(query)
    
    def simple_parameter_extraction(self, query: str) -> Dict:
        """Fallback simple parameter extraction"""
        query_upper = query.upper()
        
        symbol_mapping = {
            "APPLE": "AAPL", "TESLA": "TSLA", "MICROSOFT": "MSFT", 
            "GOOGLE": "GOOGL", "AMAZON": "AMZN", "META": "META",
            "FACEBOOK": "META", "NVIDIA": "NVDA", "NETFLIX": "NFLX"
        }
        
        for company, symbol in symbol_mapping.items():
            if company in query_upper:
                return {"symbol": symbol}
        
        words = query_upper.split()
        for word in words:
            if len(word) <= 5 and word.isalpha() and word.isupper():
                return {"symbol": word}
        
        return {"symbol": "AAPL"}
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Use GPT-4o to analyze query intent and provide context"""
        try:
            prompt = f"""
            Analyze this financial data query and provide insights:
            
            Query: "{query}"
            
            Provide a JSON response with:
            1. "intent": What the user wants (e.g., "current_price", "historical_data", "company_info", "earnings", "dividends")
            2. "entities": Any financial entities mentioned (companies, symbols, dates)
            3. "confidence": How confident you are in understanding the query (0-1)
            4. "clarification": Any clarification that might be needed
            
            Return ONLY valid JSON.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial data analyst. Analyze queries and return structured insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            analysis_text = response.choices[0].message.content.strip()
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()
            
            return json.loads(analysis_text)
            
        except Exception as e:
            print(f"⚠️  Error analyzing query intent: {e}")
            return {
                "intent": "unknown",
                "entities": [],
                "confidence": 0.5,
                "clarification": "Could not analyze query"
            }
    
    def execute_api_call(self, api_data: Dict, extracted_params: Dict) -> Dict:
        """Execute the API call with extracted parameters"""
        try:
            params = {"apikey": self.fmp_api_key}
            params.update(extracted_params)
            
            response = requests.get(api_data["endpoint"], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data,
                    "api_used": api_data["name"],
                    "endpoint": api_data["endpoint"],
                    "parameters": params,
                    "data_count": len(data) if isinstance(data, list) else 1
                }
            else:
                return {
                    "success": False,
                    "error": f"API call failed with status {response.status_code}: {response.text}",
                    "api_used": api_data["name"]
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "api_used": api_data["name"]
            }
    
    def format_response_with_gpt(self, query: str, api_result: Dict) -> str:
        """Use GPT-4o to format the API response in a user-friendly way"""
        try:
            if not api_result.get("success"):
                return f"❌ Failed to get data: {api_result.get('error', 'Unknown error')}"
            
            data = api_result.get("data", {})
            prompt = f"""
            Format this financial API response in a clear, user-friendly way for the query: "{query}"
            
            API Used: {api_result.get('api_used')}
            Data: {json.dumps(data, indent=2)[:1000]}
            
            Instructions:
            1. Provide a concise, informative summary
            2. Highlight key financial metrics
            3. Use appropriate financial terminology
            4. Make it easy to understand for general users
            5. If it's a list, show a few examples and mention total count
            
            Keep the response under 200 words.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial data analyst. Format API responses clearly and professionally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️  Error formatting response: {e}")
            return f"✅ Got data from {api_result.get('api_used', 'API')}"
    
    def query(self, user_query: str) -> Dict:
        """Main query function with learning integration"""
        print(f"\n🔍 Processing query: '{user_query}'")
        
        # Step 1: Analyze query intent
        intent_analysis = self.analyze_query_intent(user_query)
        intent = intent_analysis.get('intent', 'unknown')
        print(f"🧠 Query Intent: {intent} (confidence: {intent_analysis.get('confidence', 0):.2f})")
        
        # Step 2: Search for relevant APIs
        search_results = self.search_apis(user_query, top_k=5)
        
        if not search_results:
            return {"error": "No relevant APIs found"}
        
        # Step 3: Apply learning-based re-ranking
        search_results = self.rerank_results(search_results, intent)
        
        print(f"📊 Found {len(search_results)} relevant APIs:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. {result['api']['name']} (score: {result['score']:.3f})")
        
        # Step 4: Select best API
        best_api = search_results[0]["api"]
        api_candidates = [r["api"]["name"] for r in search_results]
        print(f"🎯 Selected API: {best_api['name']}")
        
        # Step 5: Record learning feedback
        self.learner.record_feedback(
            query=user_query,
            intent=intent,
            selected_api=best_api['name'],
            api_candidates=api_candidates
        )
        
        # Step 6: Extract parameters and execute
        extracted_params = self.extract_parameters_with_gpt(user_query, best_api)
        print(f"🔧 Extracted Parameters: {extracted_params}")
        
        api_result = self.execute_api_call(best_api, extracted_params)
        formatted_response = self.format_response_with_gpt(user_query, api_result)
        
        return {
            "query": user_query,
            "intent_analysis": intent_analysis,
            "selected_api": best_api["name"],
            "api_candidates": api_candidates,
            "extracted_parameters": extracted_params,
            "api_result": api_result,
            "formatted_response": formatted_response,
            "learning_applied": True
        }
    
    def get_learning_stats(self):
        """Get learning system statistics"""
        return self.learner.get_stats()
    
    def save_learning_data(self):
        """Manually save learning data"""
        self.learner.save_performance()

if __name__ == "__main__":
    print("🏗️  Setting up FMP API RAG System with Learning Integration...")
    print("🧠 Using GPT-4o with OpenAI embeddings and learning-based ranking")
    
    rag_system = FMPAPIRag(
        qdrant_url=QDRANT_URL,
        fmp_api_key=FMP_API_KEY,
        openai_api_key=OPENAI_API_KEY
    )
    
    print("\n💬 Welcome to the Smart FMP Chatbot! (Type 'stats' for learning stats, 'exit' to quit)")
    
    while True:
        user_query = input("\nYou: ").strip()
        
        if user_query.lower() in ("exit", "quit"):
            print("💾 Saving learning data...")
            rag_system.save_learning_data()
            print("👋 Goodbye!")
            break
        
        if user_query.lower() == "stats":
            stats = rag_system.get_learning_stats()
            print(f"\n📊 Learning Statistics:")
            print(f"  Total Queries: {stats['total_queries']}")
            print(f"  Intent Breakdown:")
            for intent, apis in stats['intent_breakdown'].items():
                print(f"    {intent}:")
                for api, perf in apis.items():
                    print(f"      {api}: {perf['success_rate']:.1%} success ({perf['total_queries']} queries)")
            continue
        
        if not user_query:
            continue
        
        result = rag_system.query(user_query)
        
        print(f"\n📋 QUERY RESULTS:")
        print(f"  🎯 Selected API: {result.get('selected_api', 'N/A')}")
        print(f"  📊 Candidates: {result.get('api_candidates', [])}")
        print(f"  🔧 Parameters: {result.get('extracted_parameters', {})}")
        
        formatted_response = result.get('formatted_response', '')
        if formatted_response:
            print(f"  💬 Response: {formatted_response}")
        
        api_result = result.get('api_result', {})
        if api_result.get('success'):
            print(f"  ✅ API Status: Success")
        else:
            print(f"  ❌ API Status: Failed - {api_result.get('error', 'Unknown error')}")