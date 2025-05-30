"""
Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models
Paper-aligned implementation with proper architecture and methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, LlamaTokenizer, LlamaModel
from typing import Dict, List, Optional, Tuple, Any
import json
import re
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import random

@dataclass
class ToolCall:
    """Represents a single tool call with its context"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    confidence: float
    reasoning_step: int

@dataclass 
class ReasoningStep:
    """Represents one step in the chain-of-thought reasoning"""
    step_id: int
    thought: str
    tool_call: Optional[ToolCall]
    intermediate_result: str

class FrozenLanguageModel:
    """
    Frozen Language Model wrapper following the paper's approach
    Uses a pre-trained model without fine-tuning
    """
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "cpu"):
        self.device = device
        print(f"Loading frozen language model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Freeze all parameters as per paper methodology
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.model.config.hidden_size
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded with hidden size: {self.hidden_size}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Frozen parameters: {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}")
    
    def encode_text(self, text: str, max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and return hidden states"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
        return hidden_states, attention_mask
    
    def get_pooled_representation(self, text: str) -> torch.Tensor:
        """Get mean-pooled representation of text"""
        hidden_states, attention_mask = self.encode_text(text)
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return pooled

class ToolCallJudge(nn.Module):
    """
    Neural module to determine when to call tools during reasoning
    Following the paper's approach for tool call decision making
    """
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_size = hidden_size
        
        for i in range(num_layers - 1):
            next_size = hidden_size // 2
            layers.extend([
                nn.Linear(current_size, next_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_size = next_size
            
        layers.append(nn.Linear(current_size, 1))
        self.classifier = nn.Sequential(*layers)
        
        # Initialize with slight bias towards not calling tools
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if module == list(self.modules())[-1]:  # Last layer
                        nn.init.constant_(module.bias, -0.5)  # Bias towards no tool call
                    else:
                        nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        Returns:
            probabilities: [batch_size, seq_len] or [batch_size]
        """
        if hidden_states.dim() == 3:
            # For sequence of hidden states, apply to each position
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_size)
            logits = self.classifier(hidden_states)
            probabilities = torch.sigmoid(logits.view(batch_size, seq_len))
        else:
            # For single hidden state
            logits = self.classifier(hidden_states)
            probabilities = torch.sigmoid(logits.squeeze(-1))
            
        return probabilities

class ToolRetriever(nn.Module):
    """
    Neural tool retriever that matches queries to appropriate tools
    Implements dense retrieval as described in the paper
    """
    def __init__(self, hidden_size: int, projection_dim: int = 256):
        super().__init__()
        
        # Query encoder - encodes the current reasoning state and query
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, projection_dim)
        )
        
        # Tool encoder - encodes tool descriptions
        self.tool_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, projection_dim)
        )
        
        self.projection_dim = projection_dim
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_query(self, query_hidden: torch.Tensor) -> torch.Tensor:
        """Encode query into retrieval space"""
        encoded = self.query_encoder(query_hidden)
        return F.normalize(encoded, p=2, dim=-1)
    
    def encode_tool(self, tool_hidden: torch.Tensor) -> torch.Tensor:
        """Encode tool description into retrieval space"""
        encoded = self.tool_encoder(tool_hidden)
        return F.normalize(encoded, p=2, dim=-1)
    
    def compute_similarity(self, query_emb: torch.Tensor, tool_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and tool embeddings"""
        return torch.matmul(query_emb, tool_emb.T)

class AdvancedToolManager:
    """
    Comprehensive tool manager supporting various tool types
    Extends beyond basic arithmetic to demonstrate the paper's vision
    """
    def __init__(self):
        self.tools = {}
        self.tool_categories = defaultdict(list)
        self._setup_comprehensive_tools()
    
    def _setup_comprehensive_tools(self):
        """Setup a comprehensive set of tools across different domains"""
        
        # Mathematical tools
        self.register_tool(
            name="add",
            category="math",
            description="Add two or more numbers together. Handles integer and decimal arithmetic.",
            function=self._safe_add,
            parameters={"x": "First number", "y": "Second number"},
            examples=["5 + 3 = 8", "2.5 + 1.7 = 4.2"],
            keywords=["add", "plus", "sum", "total", "combine", "+"]
        )
        
        self.register_tool(
            name="subtract", 
            category="math",
            description="Subtract one number from another. Computes the difference between two values.",
            function=self._safe_subtract,
            parameters={"x": "Number to subtract from", "y": "Number to subtract"},
            examples=["10 - 3 = 7", "5.5 - 2.1 = 3.4"],
            keywords=["subtract", "minus", "difference", "take away", "-", "less"]
        )
        
        self.register_tool(
            name="multiply",
            category="math", 
            description="Multiply two numbers together. Computes the product of numerical values.",
            function=self._safe_multiply,
            parameters={"x": "First number", "y": "Second number"},
            examples=["4 * 5 = 20", "2.5 * 3 = 7.5"],
            keywords=["multiply", "times", "product", "*", "of"]
        )
        
        self.register_tool(
            name="divide",
            category="math",
            description="Divide one number by another. Handles division with zero checking.",
            function=self._safe_divide,
            parameters={"x": "Dividend", "y": "Divisor"},
            examples=["15 / 3 = 5", "7 / 2 = 3.5"],
            keywords=["divide", "division", "quotient", "/", "per"]
        )
        
        self.register_tool(
            name="power",
            category="math",
            description="Raise a number to a power. Computes exponential values.",
            function=self._safe_power,
            parameters={"base": "Base number", "exponent": "Power to raise to"},
            examples=["2^3 = 8", "5^2 = 25"],
            keywords=["power", "exponent", "raised to", "^", "squared", "cubed"]
        )
        
        # String manipulation tools
        self.register_tool(
            name="string_length",
            category="string",
            description="Calculate the length of a text string.",
            function=lambda text: len(str(text)),
            parameters={"text": "Input string"},
            examples=["length of 'hello' = 5"],
            keywords=["length", "count", "characters", "size"]
        )
        
        self.register_tool(
            name="string_reverse",
            category="string", 
            description="Reverse the order of characters in a string.",
            function=lambda text: str(text)[::-1],
            parameters={"text": "Input string"},
            examples=["reverse of 'hello' = 'olleh'"],
            keywords=["reverse", "backwards", "flip"]
        )
        
        # List processing tools
        self.register_tool(
            name="list_max",
            category="list",
            description="Find the maximum value in a list of numbers.",
            function=self._safe_max,
            parameters={"numbers": "List of numbers"},
            examples=["max of [1,5,3,9,2] = 9"],
            keywords=["maximum", "max", "largest", "biggest", "highest"]
        )
        
        self.register_tool(
            name="list_min",
            category="list",
            description="Find the minimum value in a list of numbers.",
            function=self._safe_min,
            parameters={"numbers": "List of numbers"},
            examples=["min of [1,5,3,9,2] = 1"],
            keywords=["minimum", "min", "smallest", "lowest"]
        )
        
        print(f"Registered {len(self.tools)} tools across {len(self.tool_categories)} categories")
    
    def register_tool(self, name: str, category: str, description: str, function: callable, 
                     parameters: Dict[str, str], examples: List[str] = None, 
                     keywords: List[str] = None):
        """Register a new tool with comprehensive metadata"""
        self.tools[name] = {
            'name': name,
            'category': category,
            'description': description,
            'function': function,
            'parameters': parameters,
            'examples': examples or [],
            'keywords': keywords or [],
            'usage_count': 0
        }
        self.tool_categories[category].append(name)
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get comprehensive description for tool encoding"""
        if tool_name not in self.tools:
            return ""
            
        tool = self.tools[tool_name]
        desc = f"Tool: {tool['name']} | Category: {tool['category']} | "
        desc += f"Description: {tool['description']} | "
        desc += f"Parameters: {', '.join(f'{k}: {v}' for k, v in tool['parameters'].items())} | "
        desc += f"Examples: {'; '.join(tool['examples'])} | "
        desc += f"Keywords: {', '.join(tool['keywords'])}"
        
        return desc
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[Any, str]:
        """Execute a tool and return result with status"""
        if tool_name not in self.tools:
            return None, f"Error: Tool '{tool_name}' not found"
            
        try:
            tool = self.tools[tool_name]
            result = tool['function'](**parameters)
            tool['usage_count'] += 1
            return result, "Success"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Safe mathematical operations
    def _safe_add(self, x: float, y: float) -> float:
        return float(x) + float(y)
        
    def _safe_subtract(self, x: float, y: float) -> float:
        return float(x) - float(y)
        
    def _safe_multiply(self, x: float, y: float) -> float:
        return float(x) * float(y)
        
    def _safe_divide(self, x: float, y: float) -> float:
        if abs(float(y)) < 1e-10:
            raise ValueError("Division by zero")
        return float(x) / float(y)
        
    def _safe_power(self, base: float, exponent: float) -> float:
        return float(base) ** float(exponent)
        
    def _safe_max(self, numbers: List[float]) -> float:
        if not numbers:
            raise ValueError("Empty list")
        return max(float(x) for x in numbers)
        
    def _safe_min(self, numbers: List[float]) -> float:
        if not numbers:
            raise ValueError("Empty list")
        return min(float(x) for x in numbers)

class ChainOfToolsPipeline:
    """
    Main Chain-of-Tools pipeline implementing the paper's methodology
    Integrates frozen LM with neural tool calling and retrieval
    """
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "cpu"):
        self.device = device
        
        # Initialize components
        self.frozen_lm = FrozenLanguageModel(model_name, device)
        self.tool_manager = AdvancedToolManager()
        
        # Neural components
        self.tool_judge = ToolCallJudge(self.frozen_lm.hidden_size).to(device)
        self.tool_retriever = ToolRetriever(self.frozen_lm.hidden_size).to(device)
        
        # Precompute tool embeddings
        self.tool_embeddings = {}
        self._precompute_tool_embeddings()
        
        # Configuration
        self.judge_threshold = 0.4
        self.max_reasoning_steps = 5
        self.max_tools_per_step = 2
        
    def _precompute_tool_embeddings(self):
        """Precompute embeddings for all available tools"""
        print("Precomputing tool embeddings...")
        
        for tool_name in self.tool_manager.tools:
            tool_desc = self.tool_manager.get_tool_description(tool_name)
            tool_hidden = self.frozen_lm.get_pooled_representation(tool_desc)
            tool_emb = self.tool_retriever.encode_tool(tool_hidden)
            self.tool_embeddings[tool_name] = tool_emb.detach()
            
        print(f"Precomputed embeddings for {len(self.tool_embeddings)} tools")
    
    def should_call_tool(self, reasoning_context: str) -> Tuple[bool, float]:
        """Determine if a tool should be called given the current reasoning context"""
        hidden_state = self.frozen_lm.get_pooled_representation(reasoning_context)
        
        with torch.no_grad():
            call_prob = self.tool_judge(hidden_state).item()
            
        should_call = call_prob > self.judge_threshold
        return should_call, call_prob
    
    def retrieve_relevant_tools(self, query: str, context: str = "", top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the most relevant tools for the given query and context"""
        
        # Combine query and context for retrieval
        retrieval_text = f"Query: {query} Context: {context}"
        query_hidden = self.frozen_lm.get_pooled_representation(retrieval_text)
        query_emb = self.tool_retriever.encode_query(query_hidden)
        
        # Compute similarities with all tools
        tool_scores = []
        for tool_name, tool_emb in self.tool_embeddings.items():
            similarity = torch.cosine_similarity(query_emb, tool_emb, dim=-1).item()
            tool_scores.append((tool_name, similarity))
        
        # Sort by similarity and return top-k
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[:top_k]
    
    def extract_parameters(self, tool_name: str, query: str, context: str = "") -> Dict[str, Any]:
        """Extract parameters for tool execution using pattern matching and heuristics"""
        
        tool_info = self.tool_manager.tools[tool_name]
        param_names = list(tool_info['parameters'].keys())
        
        # Combine query and context for parameter extraction
        text = f"{query} {context}".lower()
        
        # Extract numbers from text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        numbers = [float(n) for n in numbers if n]
        
        params = {}
        
        # Tool-specific parameter extraction
        if tool_name in ['add', 'subtract', 'multiply', 'divide']:
            if len(numbers) >= 2:
                params['x'] = numbers[0]
                params['y'] = numbers[1]
            elif len(numbers) == 1:
                params['x'] = numbers[0]
                params['y'] = 1.0  # Default second operand
            else:
                params['x'] = 0.0
                params['y'] = 0.0
                
        elif tool_name == 'power':
            if len(numbers) >= 2:
                params['base'] = numbers[0] 
                params['exponent'] = numbers[1]
            else:
                params['base'] = numbers[0] if numbers else 2.0
                params['exponent'] = 2.0  # Default to square
                
        elif tool_name in ['list_max', 'list_min']:
            params['numbers'] = numbers if numbers else [1, 2, 3]
            
        elif tool_name in ['string_length', 'string_reverse']:
            # Extract quoted strings or use the whole query
            string_matches = re.findall(r'"([^"]*)"', query)
            if string_matches:
                params['text'] = string_matches[0]
            else:
                # Use the query itself
                params['text'] = query.strip()
        
        # Fill in any missing parameters with defaults
        for param_name in param_names:
            if param_name not in params:
                params[param_name] = self._get_default_param_value(param_name)
                
        return params
    
    def _get_default_param_value(self, param_name: str) -> Any:
        """Get default value for a parameter"""
        defaults = {
            'x': 0.0, 'y': 1.0, 'base': 2.0, 'exponent': 2.0,
            'text': 'sample', 'numbers': [1, 2, 3]
        }
        return defaults.get(param_name, None)
    
    def execute_reasoning_step(self, step_id: int, query: str, context: str) -> ReasoningStep:
        """Execute one step of chain-of-thought reasoning with potential tool use"""
        
        # Generate reasoning thought
        reasoning_prompt = f"Step {step_id}: Let me analyze the problem: {query}\nContext so far: {context}\nThought:"
        
        # Determine if we should call a tool
        should_call, confidence = self.should_call_tool(reasoning_prompt)
        
        tool_call = None
        intermediate_result = ""
        
        if should_call:
            # Retrieve relevant tools
            relevant_tools = self.retrieve_relevant_tools(query, context, top_k=3)
            
            if relevant_tools:
                # Use the most relevant tool
                best_tool_name = relevant_tools[0][0]
                tool_confidence = relevant_tools[0][1]
                
                # Extract parameters
                parameters = self.extract_parameters(best_tool_name, query, context)
                
                # Execute tool
                result, status = self.tool_manager.execute_tool(best_tool_name, parameters)
                
                if status == "Success":
                    tool_call = ToolCall(
                        tool_name=best_tool_name,
                        parameters=parameters,
                        result=result,
                        confidence=tool_confidence,
                        reasoning_step=step_id
                    )
                    intermediate_result = f"Using {best_tool_name} with {parameters}, I get: {result}"
                else:
                    intermediate_result = f"Tool execution failed: {status}"
        
        if not tool_call:
            intermediate_result = f"Step {step_id}: Analyzing the problem without tools."
        
        thought = f"In step {step_id}, I need to {query}. {intermediate_result}"
        
        return ReasoningStep(
            step_id=step_id,
            thought=thought,
            tool_call=tool_call,
            intermediate_result=intermediate_result
        )
    
    def generate_with_chain_of_tools(self, query: str) -> Tuple[str, List[ReasoningStep]]:
        """
        Generate response using Chain-of-Tools reasoning
        Main entry point following the paper's methodology
        """
        print(f"\n=== Chain-of-Tools Reasoning ===")
        print(f"Query: {query}")
        
        reasoning_steps = []
        context = f"Problem to solve: {query}"
        
        # Execute reasoning steps
        for step_id in range(1, self.max_reasoning_steps + 1):
            step = self.execute_reasoning_step(step_id, query, context)
            reasoning_steps.append(step)
            
            # Update context with the step's results
            if step.tool_call:
                context += f" Step {step_id}: Used {step.tool_call.tool_name} and got {step.tool_call.result}."
            else:
                context += f" Step {step_id}: Continued reasoning."
            
            # Check if we have a definitive answer
            if step.tool_call and self._is_final_answer(step.tool_call.result):
                break
        
        # Generate final answer
        final_answer = self._generate_final_answer(query, reasoning_steps)
        
        return final_answer, reasoning_steps
    
    def _is_final_answer(self, result: Any) -> bool:
        """Check if the result represents a final answer"""
        # For mathematical problems, a numeric result is usually final
        try:
            float(result)
            return True
        except (ValueError, TypeError):
            return False
    
    def _generate_final_answer(self, query: str, steps: List[ReasoningStep]) -> str:
        """Generate final answer based on reasoning steps"""
        
        # Find the last successful tool call
        last_tool_result = None
        tool_calls_summary = []
        
        for step in steps:
            if step.tool_call:
                tool_calls_summary.append(f"Step {step.step_id}: {step.tool_call.tool_name}({step.tool_call.parameters}) = {step.tool_call.result}")
                last_tool_result = step.tool_call.result
        
        if last_tool_result is not None:
            answer = f"Based on my chain-of-tools reasoning:\n\n"
            answer += "\n".join(tool_calls_summary)
            answer += f"\n\nTherefore, the answer to '{query}' is: {last_tool_result}"
        else:
            answer = f"I analyzed the problem '{query}' through {len(steps)} reasoning steps, but couldn't find a definitive numerical answer using the available tools."
        
        return answer

def comprehensive_demo():
    """Comprehensive demonstration of the Chain-of-Tools implementation"""
    
    print("=== Chain-of-Tools: Paper Implementation Demo ===")
    
    # Initialize the pipeline
    pipeline = ChainOfToolsPipeline()
    
    # Test cases covering different scenarios
    test_cases = [
        # Basic arithmetic
        "What is 25 plus 17?",
        "Calculate 144 divided by 12",
        "Find the product of 7 and 9",
        
        # More complex scenarios  
        "If I have 100 dollars and spend 35 dollars, how much money do I have left?",
        "What is 2 raised to the power of 5?",
        "I have three test scores: 85, 92, and 78. What is my highest score?",
        
        # String operations
        'What is the length of the word "artificial"?',
        'Reverse the string "hello world"',
        
        # Edge cases
        "What is 5 divided by 0?",
        "Calculate the square root of 16",  # This might not work with current tools
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {query}")
        print('='*60)
        
        try:
            final_answer, reasoning_steps = pipeline.generate_with_chain_of_tools(query)
            
            print("\n--- Reasoning Steps ---")
            for step in reasoning_steps:
                print(f"Step {step.step_id}: {step.thought}")
                if step.tool_call:
                    print(f"  → Tool: {step.tool_call.tool_name}")
                    print(f"  → Parameters: {step.tool_call.parameters}")
                    print(f"  → Result: {step.tool_call.result}")
                    print(f"  → Confidence: {step.tool_call.confidence:.3f}")
            
            print(f"\n--- Final Answer ---")
            print(final_answer)
            
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # Print tool usage statistics
    print(f"\n{'='*60}")
    print("Tool Usage Statistics")
    print('='*60)
    for tool_name, tool_info in pipeline.tool_manager.tools.items():
        print(f"{tool_name}: {tool_info['usage_count']} uses")

if __name__ == "__main__":
    comprehensive_demo()