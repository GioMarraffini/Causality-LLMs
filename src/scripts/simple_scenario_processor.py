#!/usr/bin/env python3
"""
Simple, clean scenario processor that focuses on the core task.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class CausalScenario:
    """Data class for a causal scenario."""

    id: str
    paper: str
    scenario_text: str
    causal_structure: str
    original_question: str
    link: str


@dataclass
class PromptVariation:
    """Data class for a prompt variation."""

    scenario_id: str
    variation_type: str
    system_prompt: str
    user_prompt: str
    expected_bias: str
    scale_meaning: str


class SimpleScenarioProcessor:
    """Simple processor focused on core functionality."""

    def __init__(self):
        self.scenarios = []
        self.prompt_variations = []

    def parse_excel_file(self, excel_path: str) -> List[CausalScenario]:
        """Parse the Excel file and extract clean scenarios."""
        df = pd.read_excel(excel_path)

        scenarios = []
        current_paper = ""

        for idx, row in df.iterrows():
            if (
                pd.isna(row.get("Scenario", ""))
                or str(row.get("Scenario", "")).strip() == ""
            ):
                continue

            if pd.notna(row.get("Paper", "")):
                current_paper = str(row["Paper"]).strip()

            scenario_text = str(row.get("Scenario", "")).strip()
            if not scenario_text or scenario_text.lower() in ["scenario", "nan"]:
                continue

            original_question = str(row.get("Questions ", "")).strip()
            if original_question.lower() == "nan":
                original_question = ""

            scenario = CausalScenario(
                id=f"scenario_{len(scenarios) + 1}",
                paper=current_paper,
                scenario_text=scenario_text,
                causal_structure=str(row.get("Causal Structure", "")).strip(),
                original_question=original_question,
                link=str(row.get("Link ", "")).strip(),
            )

            scenarios.append(scenario)

        self.scenarios = scenarios
        return scenarios

    def generate_system_prompt(self) -> str:
        """Generate the system prompt that will be consistent across all scenarios."""
        return """You are an expert in causal reasoning and philosophy. Your task is to analyze causal scenarios and determine the degree of causal relationship between actions and outcomes.

When evaluating causality, consider:
- Direct causal connections between actions and outcomes
- The role of intervening factors and alternative causes
- Counterfactual reasoning (what would have happened otherwise)
- The distinction between correlation and causation
- Whether the action was necessary, sufficient, or both for the outcome

Important guidelines:
- Focus strictly on causal relationships, not moral responsibility, blame, or guilt
- Avoid judgments about intentions, fault, or ethical considerations
- Consider both proximate and ultimate causes
- Be precise about the specific causal claim being evaluated
- Base your assessment on the logical structure of events

You must provide your reasoning step by step, then give a numerical assessment."""

    def generate_prompt_variations(
        self, scenario: CausalScenario
    ) -> List[PromptVariation]:
        """Generate different prompt variations for a scenario."""
        variations = []

        # Use the original question if available, otherwise create generic ones
        if scenario.original_question:
            base_question = scenario.original_question
        else:
            base_question = (
                "What is the degree of causal relationship in this scenario?"
            )

        # Variation configurations
        variation_configs = [
            {
                "type": "positive_causal_assessment",
                "question": f"Based on the scenario: {base_question} Rate the strength of the causal relationship.",
                "bias": "None - neutral causal assessment",
                "scale": "0 = No causal relationship, 100 = Very strong causal relationship",
            },
            {
                "type": "negative_causal_assessment",
                "question": f"Based on the scenario: To what degree is there NO causal relationship? {base_question}",
                "bias": "Negative framing may bias toward lower causal attribution",
                "scale": "0 = Strong causal relationship, 100 = No causal relationship",
            },
            {
                "type": "counterfactual_reasoning",
                "question": f"Based on the scenario: If the key action had not occurred, would the outcome still have happened? {base_question}",
                "bias": "Focuses on necessity; may underestimate overdetermination",
                "scale": "0 = Outcome definitely would not occur, 100 = Outcome definitely would occur",
            },
            {
                "type": "causal_strength_direct",
                "question": f"Based on the scenario: How strong is the causal connection described? {base_question}",
                "bias": "None - direct assessment of causal strength",
                "scale": "0 = No causal connection, 100 = Very strong causal connection",
            },
        ]

        for config in variation_configs:
            user_prompt = f"""Scenario: {scenario.scenario_text}

Question: {config["question"]}

Please provide your reasoning step by step. Consider:
1. The sequence of events and their relationships
2. Whether actions were necessary for outcomes
3. Whether actions were sufficient for outcomes  
4. Any intervening factors or alternative causes
5. What would have happened in different circumstances

After your reasoning, provide a numerical rating from 0 to 100.

Scale meaning: {config["scale"]}

Your response format:
Reasoning: [Your step-by-step causal analysis]
Rating: [Number from 0-100]"""

            variation = PromptVariation(
                scenario_id=scenario.id,
                variation_type=config["type"],
                system_prompt=self.generate_system_prompt(),
                user_prompt=user_prompt,
                expected_bias=config["bias"],
                scale_meaning=config["scale"],
            )

            variations.append(variation)

        return variations

    def process_all_scenarios(self) -> List[PromptVariation]:
        """Process all scenarios and generate all prompt variations."""
        all_variations = []

        for scenario in self.scenarios:
            variations = self.generate_prompt_variations(scenario)
            all_variations.extend(variations)

        self.prompt_variations = all_variations
        return all_variations

    def save_to_json(self, output_path: str):
        """Save scenarios and prompt variations to JSON."""
        data = {
            "scenarios": [asdict(s) for s in self.scenarios],
            "prompt_variations": [asdict(v) for v in self.prompt_variations],
            "metadata": {
                "total_scenarios": len(self.scenarios),
                "total_variations": len(self.prompt_variations),
                "variations_per_scenario": len(self.prompt_variations)
                // len(self.scenarios)
                if self.scenarios
                else 0,
                "variation_types": list(
                    set(v.variation_type for v in self.prompt_variations)
                ),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_to_csv(self, output_path: str):
        """Save prompt variations to CSV for easy review."""
        if not self.prompt_variations:
            return

        df_data = []
        for variation in self.prompt_variations:
            df_data.append(asdict(variation))

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)


def main():
    processor = SimpleScenarioProcessor()

    # Parse the Excel file
    excel_path = "/home/gio/Documents/Repos/Causality-LLMs/Causal Scenarios.xlsx"
    scenarios = processor.parse_excel_file(excel_path)

    print(f"Parsed {len(scenarios)} scenarios:")
    for i, scenario in enumerate(scenarios[:3]):  # Show first 3
        print(f"\nScenario {i + 1}:")
        print(f"Paper: {scenario.paper[:50]}...")
        print(f"Text: {scenario.scenario_text[:100]}...")
        if scenario.original_question:
            print(f"Original Q: {scenario.original_question}")

    # Generate prompt variations
    variations = processor.process_all_scenarios()
    print(f"\nGenerated {len(variations)} prompt variations")
    print(f"Variation types: {set(v.variation_type for v in variations)}")

    # Save outputs
    output_dir = Path("/home/gio/Documents/Repos/Causality-LLMs/data")
    output_dir.mkdir(exist_ok=True)

    processor.save_to_json(output_dir / "causal_scenarios.json")
    processor.save_to_csv(output_dir / "prompt_variations.csv")

    print("\nSaved data to:")
    print(f"- {output_dir / 'causal_scenarios.json'}")
    print(f"- {output_dir / 'prompt_variations.csv'}")

    # Show example of a complete prompt variation
    if variations:
        print("\nExample prompt variation:")
        example = variations[0]
        print(f"Type: {example.variation_type}")
        print(f"Expected bias: {example.expected_bias}")
        print("\nComplete user prompt:")
        print(example.user_prompt)


if __name__ == "__main__":
    main()
