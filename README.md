# Causality-LLMs: Evaluating Large Language Models on Causal Reasoning

This project evaluates how different Large Language Models (LLMs) perform on causal reasoning tasks using scenarios extracted from academic papers. The system generates multiple prompt variations to test for biases and consistency in causal judgment.

## Overview

The project processes causal scenarios from research papers and generates systematic prompt variations to evaluate LLM responses. Each scenario is tested with different framings to identify potential biases in causal reasoning.

## Features

- **Scenario Processing**: Parses Excel files containing 27 causal scenarios from academic papers
- **Prompt Generation**: Creates 4 variations of prompts for each scenario with different causal framings
- **Async LLM Client**: High-performance OpenAI client with caching and batch processing
- **Response Extraction**: Automatically extracts numerical ratings (0-100) and reasoning from LLM responses
- **Bias Analysis**: Compares responses across different prompt variations to identify systematic biases
- **Multi-Model Support**: Tests scenarios with different LLM models for comparison

## Project Structure

```
Causality-LLMs/
├── src/
│   ├── clients/
│   │   ├── openai_client.py      # Async OpenAI API client with caching
│   │   └── consts.py             # API constants and configuration
│   └── scripts/
│       ├── simple_scenario_processor.py  # Main scenario processor
│       └── llm_evaluator.py      # LLM evaluation pipeline
├── data/
│   ├── causal_scenarios.json     # Processed scenarios (auto-generated)
│   └── prompt_variations.csv     # Generated prompt variations (auto-generated)
├── results/                      # Created when running evaluator
│   ├── test_responses.csv        # LLM responses (auto-generated)
│   ├── test_responses.json       # Detailed response data (auto-generated)
│   └── test_analysis.json        # Analysis results (auto-generated)
├── Causal Scenarios.xlsx         # Original Excel file (27 scenarios)
├── causal_scenarios_raw.csv      # Raw CSV export of Excel data
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API keys)
└── README.md                     # This file
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Causality-LLMs
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

## Usage

### 1. Process Scenarios

Run the scenario processor to extract and clean scenarios from the Excel file:

```bash
python src/scripts/simple_scenario_processor.py
```

This will:
- Parse the Excel file containing 27 causal scenarios from research papers
- Extract scenario text, causal structure, and original questions
- Generate 4 prompt variations per scenario with different causal framings
- Save processed data to `data/causal_scenarios.json` and `data/prompt_variations.csv`

### 2. Evaluate with LLMs

Run the LLM evaluator to test scenarios with language models:

```bash
python src/scripts/llm_evaluator.py
```

This will:
- Load processed scenarios
- Send prompts to LLMs via API
- Extract reasoning and numerical ratings
- Save responses and analysis to `results/` directory

## Prompt Variations

The system generates 4 types of prompt variations for each scenario to test different causal reasoning approaches:

1. **Positive Causal Assessment**: Direct evaluation of causal relationships
   - Neutral framing focusing on causal contribution
   - Scale: 0-100 rating of causal strength

2. **Negative Causal Absence**: Inverse framing to detect bias
   - Tests for consistency by asking about lack of causation
   - May reveal framing effects in causal judgment

3. **Counterfactual Necessity**: Tests necessity-based reasoning
   - Focuses on "what would have happened otherwise"
   - Evaluates whether the action was necessary for the outcome

4. **Sufficient Causation**: Tests sufficiency-based reasoning
   - Evaluates whether the action alone was sufficient
   - May reveal different causal intuitions than necessity

5. **Causal Strength Direct**: Direct strength assessment
   - Straightforward evaluation of causal connection strength
   - Baseline for comparison with other framings

Each variation uses the original research question when available, or generates appropriate causal questions for the scenario.

## Key Design Principles

- **Avoid Moral Language**: Prompts avoid terms like "guilty," "blame," or "fault" to focus purely on causal relationships
- **Systematic Variations**: Each scenario tested with multiple framings to identify bias patterns
- **Structured Responses**: LLMs required to provide step-by-step reasoning before numerical ratings
- **Bias Documentation**: Each prompt variation documents potential biases it might introduce

## Data Format

The processed scenarios include:
- **Scenario Text**: The full causal scenario description from research papers
- **Causal Structure**: Type of causation (e.g., conjunctive, disjunctive, causal chain)
- **Original Question**: Question from the source paper (when available)
- **Paper Source**: Academic paper the scenario was extracted from (e.g., MoCa paper, causal judgment studies)
- **Link**: URL to the original research paper

The system processes 26 scenarios from papers including:
- "MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks"
- "Judgments of cause and blame: The effects of intentionality and foreseeability"

## Analysis Features

The evaluation system provides:
- Response time analysis
- Rating extraction accuracy
- Cross-model comparison
- Variation type analysis
- Bias pattern identification

## Technical Details

### OpenAI Client Features
- **Async Processing**: Uses `aiohttp` for concurrent API requests
- **Response Caching**: Avoids duplicate API calls with SHA256-based cache keys
- **Batch Processing**: Processes multiple requests with concurrency control
- **Retry Logic**: Exponential backoff for failed requests
- **Error Handling**: Comprehensive error handling for API failures

### Response Processing
- **Rating Extraction**: Automatically extracts 0-100 numerical ratings using regex patterns
- **Reasoning Extraction**: Separates step-by-step reasoning from final ratings
- **Response Analysis**: Tracks response times, success rates, and model performance

## Contributing

To add new scenarios or prompt variations:
1. Update the Excel file with new scenarios
2. Modify the scenario parsing logic in `simple_scenario_processor.py`
3. Add new variation types in the `generate_prompt_variations` method
4. Update the `LLMEvaluator` class for new analysis features
5. Document any new biases or considerations

## Research Applications

This framework can be used to:
- Study consistency in LLM causal reasoning
- Identify systematic biases in different prompt formulations
- Compare causal reasoning across different model architectures
- Validate LLM performance against human causal judgments
- Test robustness of causal reasoning to prompt variations