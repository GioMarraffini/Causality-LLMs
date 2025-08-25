#!/usr/bin/env python3
"""
LLM Evaluator script to test causal reasoning prompts with different language models.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Import your existing OpenAI client
import sys
sys.path.append('/home/gio/Documents/Repos/Causality-LLMs/src')
from clients.openai_client import OpenAIClient, process_batch
from clients.consts import DEFAULT_MODEL, DEFAULT_TEMPERATURE

@dataclass
class LLMResponse:
    """Data class for storing LLM responses."""
    scenario_id: str
    variation_type: str
    model_name: str
    system_prompt: str
    user_prompt: str
    response_text: str
    extracted_reasoning: str
    extracted_rating: Optional[int]
    response_time: float
    timestamp: str
    expected_bias: str

class LLMEvaluator:
    """Evaluate LLM responses to causal reasoning scenarios."""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
        self.responses = []
        
    def extract_rating_from_response(self, response_text: str) -> Optional[int]:
        """Extract numerical rating from LLM response."""
        import re
        
        # Look for patterns like "Rating: 85" or "Rating: [85]"
        rating_patterns = [
            r'Rating:\s*\[?(\d+)\]?',
            r'rating:\s*\[?(\d+)\]?',
            r'Rating\s*=\s*(\d+)',
            r'rating\s*=\s*(\d+)',
            r'(\d+)/100',
            r'(\d+)\s*out\s*of\s*100',
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    rating = int(match.group(1))
                    if 0 <= rating <= 100:
                        return rating
                except ValueError:
                    continue
        
        # Look for any number between 0-100 near the end of the response
        numbers = re.findall(r'\b(\d+)\b', response_text)
        for num_str in reversed(numbers):  # Check from end backwards
            try:
                num = int(num_str)
                if 0 <= num <= 100:
                    return num
            except ValueError:
                continue
                
        return None
    
    def extract_reasoning_from_response(self, response_text: str) -> str:
        """Extract reasoning section from LLM response."""
        import re
        
        # Look for reasoning section
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=Rating:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        
        # If no explicit reasoning section, return first part before rating
        rating_match = re.search(r'Rating:', response_text, re.IGNORECASE)
        if rating_match:
            return response_text[:rating_match.start()].strip()
        
        # Return full response if no clear structure
        return response_text.strip()
    
    async def evaluate_prompt(self, prompt_variation: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> LLMResponse:
        """Evaluate a single prompt variation with an LLM."""
        start_time = time.time()
        
        try:
            # Make API call
            response = await self.openai_client.chat_completion(
                messages=[
                    {"role": "system", "content": prompt_variation["system_prompt"]},
                    {"role": "user", "content": prompt_variation["user_prompt"]}
                ],
                model=model_name,
                temperature=DEFAULT_TEMPERATURE
            )
            
            response_text = response['choices'][0]['message']['content']
            response_time = time.time() - start_time
            
            # Extract reasoning and rating
            reasoning = self.extract_reasoning_from_response(response_text)
            rating = self.extract_rating_from_response(response_text)
            
            llm_response = LLMResponse(
                scenario_id=prompt_variation["scenario_id"],
                variation_type=prompt_variation["variation_type"],
                model_name=model_name,
                system_prompt=prompt_variation["system_prompt"],
                user_prompt=prompt_variation["user_prompt"],
                response_text=response_text,
                extracted_reasoning=reasoning,
                extracted_rating=rating,
                response_time=response_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                expected_bias=prompt_variation["expected_bias"]
            )
            
            return llm_response
            
        except Exception as e:
            print(f"Error evaluating prompt: {e}")
            # Return error response
            return LLMResponse(
                scenario_id=prompt_variation["scenario_id"],
                variation_type=prompt_variation["variation_type"],
                model_name=model_name,
                system_prompt=prompt_variation["system_prompt"],
                user_prompt=prompt_variation["user_prompt"],
                response_text=f"ERROR: {str(e)}",
                extracted_reasoning="",
                extracted_rating=None,
                response_time=time.time() - start_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                expected_bias=prompt_variation["expected_bias"]
            )
    
    def evaluate_all_prompts(self, prompt_variations: List[Dict[str, Any]], 
                           models: List[str] = ["gpt-4", "gpt-3.5-turbo"],
                           max_prompts: Optional[int] = None,
                           delay_between_calls: float = 1.0) -> List[LLMResponse]:
        """Evaluate all prompt variations with specified models."""
        
        if max_prompts:
            prompt_variations = prompt_variations[:max_prompts]
        
        total_evaluations = len(prompt_variations) * len(models)
        print(f"Starting evaluation of {total_evaluations} prompt-model combinations...")
        
        all_responses = []
        
        with tqdm(total=total_evaluations, desc="Evaluating prompts") as pbar:
            for prompt_variation in prompt_variations:
                for model in models:
                    response = self.evaluate_prompt(prompt_variation, model)
                    all_responses.append(response)
                    self.responses.append(response)
                    
                    pbar.set_description(f"Evaluating {model} on {prompt_variation['variation_type']}")
                    pbar.update(1)
                    
                    # Delay to respect rate limits
                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)
        
        return all_responses
    
    def save_responses_to_csv(self, output_path: str):
        """Save all responses to CSV."""
        if not self.responses:
            print("No responses to save.")
            return
        
        df_data = [asdict(response) for response in self.responses]
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(self.responses)} responses to {output_path}")
    
    def save_responses_to_json(self, output_path: str):
        """Save all responses to JSON."""
        if not self.responses:
            print("No responses to save.")
            return
        
        data = {
            "responses": [asdict(response) for response in self.responses],
            "metadata": {
                "total_responses": len(self.responses),
                "models_used": list(set(r.model_name for r in self.responses)),
                "variation_types": list(set(r.variation_type for r in self.responses)),
                "scenarios_evaluated": list(set(r.scenario_id for r in self.responses))
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.responses)} responses to {output_path}")
    
    def analyze_responses(self) -> Dict[str, Any]:
        """Analyze the collected responses for patterns and biases."""
        if not self.responses:
            return {}
        
        analysis = {
            "total_responses": len(self.responses),
            "successful_extractions": len([r for r in self.responses if r.extracted_rating is not None]),
            "average_response_time": sum(r.response_time for r in self.responses) / len(self.responses),
            "models_analyzed": {},
            "variation_type_analysis": {},
            "scenario_analysis": {}
        }
        
        # Analyze by model
        for model in set(r.model_name for r in self.responses):
            model_responses = [r for r in self.responses if r.model_name == model]
            ratings = [r.extracted_rating for r in model_responses if r.extracted_rating is not None]
            
            analysis["models_analyzed"][model] = {
                "total_responses": len(model_responses),
                "successful_extractions": len(ratings),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "rating_std": pd.Series(ratings).std() if ratings else None
            }
        
        # Analyze by variation type
        for var_type in set(r.variation_type for r in self.responses):
            var_responses = [r for r in self.responses if r.variation_type == var_type]
            ratings = [r.extracted_rating for r in var_responses if r.extracted_rating is not None]
            
            analysis["variation_type_analysis"][var_type] = {
                "total_responses": len(var_responses),
                "successful_extractions": len(ratings),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "expected_bias": var_responses[0].expected_bias if var_responses else None
            }
        
        return analysis

    async def evaluate_all_scenarios(self) -> Dict[str, Any]:
        """Evaluate all scenarios with all variations using async processing."""
        # Load prompt variations
        data_path = Path("/home/gio/Documents/Repos/Causality-LLMs/data/causal_scenarios.json")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenarios = data["scenarios"]
        prompt_variations = data["prompt_variations"]
        
        print(f"🎯 Loaded {len(scenarios)} scenarios with {len(prompt_variations)} total prompt variations")
        print(f"🤖 Using model: {DEFAULT_MODEL}")
        print(f"🌡️  Temperature: 1.0 (default - gpt-5-mini only supports default temperature)")
        print()
        
        # Group variations by scenario
        scenarios_dict = {s["id"]: s for s in scenarios}
        variations_by_scenario = {}
        for var in prompt_variations:
            scenario_id = var["scenario_id"]
            if scenario_id not in variations_by_scenario:
                variations_by_scenario[scenario_id] = []
            variations_by_scenario[scenario_id].append(var)
        
        # Initialize results structure
        start_time = time.time()
        all_results = {
            "model": DEFAULT_MODEL,
            "temperature": 1.0,  # gpt-5-mini only supports default temperature
            "total_scenarios": len(scenarios),
            "total_variations": len(prompt_variations),
            "start_time": datetime.now().isoformat(),
            "scenarios": []
        }
        
        error_count = 0
        
        # Process each scenario
        for scenario_id in tqdm(scenarios_dict.keys(), desc="Processing scenarios"):
            scenario = scenarios_dict[scenario_id]
            variations = variations_by_scenario[scenario_id]
            
            print(f"\n📖 Processing {scenario_id}: {scenario['paper'][:50]}...")
            
            # Process all variations for this scenario in parallel
            scenario_start = time.time()
            
            try:
                # Create batch requests for this scenario's variations
                batch_requests = []
                for var in variations:
                    # Note: gpt-5-mini only supports temperature=1 (default), so we omit temperature parameter
                    batch_requests.append({
                        "messages": [
                            {"role": "system", "content": var["system_prompt"]},
                            {"role": "user", "content": var["user_prompt"]}
                        ],
                        "model": DEFAULT_MODEL,
                        "max_completion_tokens": 4096
                    })
                
                # Execute all variations in parallel
                batch_results = await process_batch(self.openai_client, batch_requests, max_concurrent=4)
                
                # Process results
                scenario_results = []
                for i, (variation, result) in enumerate(zip(variations, batch_results)):
                    if isinstance(result, Exception):
                        print(f"   ❌ Error in variation {variation['variation_type']}: {result}")
                        error_count += 1
                        scenario_results.append({
                            "variation_type": variation["variation_type"],
                            "expected_bias": variation["expected_bias"],
                            "rating": None,
                            "reasoning": "",
                            "prompt": variation["user_prompt"],
                            "error": str(result)
                        })
                    else:
                        response_text = result['choices'][0]['message']['content']
                        rating = self.extract_rating_from_response(response_text)
                        reasoning = self.extract_reasoning_from_response(response_text)
                        
                        if rating is None:
                            error_count += 1
                            print(f"   ⚠️  Could not extract rating from {variation['variation_type']}")
                        
                        scenario_results.append({
                            "variation_type": variation["variation_type"],
                            "expected_bias": variation["expected_bias"],
                            "rating": rating,
                            "reasoning": reasoning,
                            "prompt": variation["user_prompt"],
                            "full_response": response_text
                        })
                
                scenario_time = time.time() - scenario_start
                print(f"   ✅ Completed in {scenario_time:.2f}s")
                
                # Add scenario to results
                all_results["scenarios"].append({
                    "id": scenario["id"],
                    "paper": scenario["paper"],
                    "causal_structure": scenario["causal_structure"],
                    "original_question": scenario["original_question"],
                    "scenario_text": scenario["scenario_text"],
                    "results": scenario_results
                })
                
            except Exception as e:
                print(f"   ❌ Failed to process scenario {scenario_id}: {e}")
                error_count += len(variations)
                # Add error scenario to results
                all_results["scenarios"].append({
                    "id": scenario["id"],
                    "paper": scenario["paper"],
                    "causal_structure": scenario["causal_structure"],
                    "original_question": scenario["original_question"],
                    "scenario_text": scenario["scenario_text"],
                    "results": [],
                    "error": str(e)
                })
        
        # Finalize results
        total_time = time.time() - start_time
        all_results.update({
            "total_time": total_time,
            "errors": error_count,
            "end_time": datetime.now().isoformat(),
            "success_rate": (len(prompt_variations) - error_count) / len(prompt_variations) if prompt_variations else 0
        })
        
        print(f"\n🎉 Evaluation completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"❌ Errors: {error_count}/{len(prompt_variations)}")
        print(f"✅ Success rate: {all_results['success_rate']:.1%}")
        
        return all_results

async def main():
    """Main function to run the evaluation."""
    print("🧪 LLM Causality Evaluation - All Scenarios")
    print("=" * 50)
    
    # Initialize evaluator
    async with OpenAIClient(cache_enabled=True) as openai_client:
        evaluator = LLMEvaluator(openai_client)
        
        try:
            # Run evaluation
            results = await evaluator.evaluate_all_scenarios()
            
            # Save results
            output_dir = Path("/home/gio/Documents/Repos/Causality-LLMs/results")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / "all_scenarios_evaluation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
