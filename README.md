# Tathya: Advanced Agentic Fact-Checking System
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by LangChain](https://img.shields.io/badge/Powered%20By-LangChain-blue.svg)](https://www.langchain.com/)

An advanced fact-checking system utilizing multi-agent architecture, LangGraph orchestration, and confidence scoring to provide transparent and reliable fact verification.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

Tathya (Sanskrit for "fact" or "truth") is a sophisticated fact-checking system designed to verify claims through a multi-agent approach. It decomposes complex claims, gathers evidence from multiple sources (web search, news APIs, knowledge bases), cross-verifies findings, and synthesizes a final verdict with detailed confidence scoring.

The system was built with the following design principles:
- **Transparency**: Clear reasoning paths and evidence for each verification step.
- **Reliability**: Multi-source evidence gathering and potential for multi-agent verification.
- **Confidence Quantification**: Detailed confidence scoring based on evidence.
- **Modularity**: Extensible architecture to easily incorporate new verification agents and evidence sources.

## Architecture

Tathya currently follows a pipeline architecture where specialized components work sequentially:

1.  **Claim Processing**:
    *   Claim decomposition into searchable components using the primary LLM.

2.  **Evidence Retrieval**:
    *   Gathering information from multiple sources using various tools:
        *   Web search engines (Tavily, DuckDuckGo).
        *   Dedicated Search-Augmented LLM calls (Primary LLM with Search).
        *   Knowledge bases (Wikidata).
        *   News APIs (NewsAPI).

3.  **Evidence Processing**:
    *   Parsing and structuring results (e.g., using a secondary LLM for Search-Augmented LLM output).
    *   Deduplication of retrieved information based on source URLs.
    *   Verification of relevance (currently placeholder logic using the primary LLM).

4.  **Synthesis & Verdict**:
    *   Formatting collected evidence for the primary LLM.
    *   Generating a final verdict (TRUE, FALSE, UNCERTAIN, etc.) and explanation using the primary LLM based on the synthesized evidence.
    *   Assigning a confidence score to the final verdict.


## Key Features

- **Claim Decomposition**: Breaks down claims into searchable components using an LLM.
- **Multi-Source Evidence Retrieval**: Gathers evidence from Tavily, DuckDuckGo, Wikidata, NewsAPI, and a Search-Augmented LLM.
- **LLM-Powered Parsing**: Uses a secondary LLM to structure output from specific tools.
- **Evidence Deduplication**: Removes redundant information based on source URLs.
- **LLM-Powered Synthesis**: Generates a final verdict and explanation based on aggregated evidence using the primary LLM.
- **Confidence Scoring**: Provides a confidence score alongside the verdict.

## Installation

### Prerequisites

- Python 3.9 or higher.
- API keys for desired services (see Configuration).

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kaos599/tathya-fact-checking-system.git
    cd tathya-fact-checking-system
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Set up required environment variables (see Configuration).

## Configuration

Configure the system by setting environment variables. Create a `.env` file in the project root or set them directly in your system environment.

**Required:**

*   **Primary LLM Configuration**: Define the primary language model used for core reasoning tasks like claim decomposition and synthesis. (Specific variables depend on the chosen provider, e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
*   **Primary LLM Model Name**: Specify the model identifier (e.g., `PRIMARY_MODEL_NAME=gpt-4o`).

**Optional (for Tools):**

*   **Search-Augmented LLM Configuration**:
    *   `GEMINI_API_KEY`: API Key for Google Gemini (used by `perform_gemini_google_search`).
    *   `GEMINI_MODEL`: Model name (e.g., `gemini-pro`).
*   **Parsing LLM Configuration**: Define the model used for parsing structured data, if different from the primary LLM.
*   **Search Tool Keys**:
    *   `TAVILY_API_KEY`: API Key for Tavily search.
    *   `NEWS_API_KEY`: API Key for NewsAPI.
    *   *(DuckDuckGo and Wikidata tools do not require API keys)*

**Optional (Behavioral):**

*   **Confidence Thresholds**:
    *   `HIGH_CONFIDENCE_THRESHOLD`: (Default: 0.8)
    *   `MEDIUM_CONFIDENCE_THRESHOLD`: (Default: 0.5)
    *   `LOW_CONFIDENCE_THRESHOLD`: (Default: 0.3)

*Example `.env` file structure:*
```dotenv
# Primary LLM (Example: OpenAI)
OPENAI_API_KEY=your_openai_api_key
PRIMARY_MODEL_NAME=gpt-4o

# Search-Augmented LLM (Example: Gemini)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash

# Tool Keys
TAVILY_API_KEY=your_tavily_api_key
NEWS_API_KEY=your_newsapi_key

# Optional Thresholds
# HIGH_CONFIDENCE_THRESHOLD=0.85
# MEDIUM_CONFIDENCE_THRESHOLD=0.6
```

## Usage

The main entry point is the `run_fact_check` function.

```python
# Example usage (ensure environment variables are set)
from fact_check_system.fact_checker import run_fact_check
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

claim = "The Eiffel Tower is located in Berlin."
result = run_fact_check(claim)

print(f"Claim: {result.claim}")
print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Explanation:\n{result.explanation}")
print("\nEvidence Sources:")
if result.evidence_sources:
    for evidence in result.evidence_sources:
        print(f"- [{evidence.source_tool}] {evidence.title or evidence.url}")
else:
    print("- No evidence sources found.")

```

## Development

### Project Structure

```
.
├── fact_check_system/      # Main package directory
│   ├── config.py           # Environment loading, API clients
│   ├── models.py           # Core Pydantic data models
│   ├── schemas.py          # Additional Pydantic schemas (potential future use)
│   ├── tools.py            # Wrappers for external APIs/tools
│   ├── prompts.py          # LangChain prompt templates
│   └── fact_checker.py     # Main orchestration logic
├── .env.example            # Example environment file
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

### Running Tests (Placeholder)

*(Add instructions for running tests once implemented)*

```bash
# Example: pytest
# pytest tests/
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Ensure code follows project style guidelines (consider adding linters like Ruff/Black).
5.  Write tests for new functionality.
6.  Commit your changes (`git commit -am 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Create a new Pull Request.

Please ensure your PR includes a clear description of the changes and addresses any relevant issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming MIT, create a LICENSE file if needed).

