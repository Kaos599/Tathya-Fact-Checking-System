# Tathya: Advanced Agentic Fact-Checking System

An advanced fact-checking system utilizing multi-agent architecture, LangGraph orchestration, and confidence scoring to provide transparent and reliable fact verification.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)

## Overview

Tathya (Sanskrit for "fact" or "truth") is a sophisticated fact-checking system designed to verify claims through a multi-agent approach. It decomposes complex claims into atomic verifiable units, gathers evidence from multiple sources, cross-verifies through parallel agent reasoning, and synthesizes a final verdict with detailed confidence scoring.

The system was built with the following design principles:
- **Transparency**: Clear reasoning paths and evidence for each verification step
- **Reliability**: Multi-agent verification to mitigate individual agent biases
- **Confidence Quantification**: Detailed confidence scoring based on evidence quality, verification consensus, and claim complexity
- **Modularity**: Extensible architecture to easily incorporate new verification agents and evidence sources

## Architecture

Tathya follows a multi-agent architecture where specialized agents work collaboratively through a precisely orchestrated workflow:

### Core Components

1. **Claim Processing Pipeline**
   - Claim extraction and normalization
   - Claim decomposition into atomic verifiable units
   - Priority ranking of claims based on complexity and importance

2. **Verification Framework**
   - Multi-source evidence retrieval system (Web, Wikidata, News APIs)
   - Cross-verification through parallel agent reasoning using different LLMs (Azure OpenAI and Google Gemini)
   - Evidence evaluation and synthesis

3. **Orchestration Layer**
   - State management and workflow coordination using LangGraph
   - Human feedback integration points
   - Execution monitoring and error handling

4. **Confidence Scoring**
   - Evidence quality and quantity assessment
   - Verification consensus measurement
   - Claim complexity evaluation
   - Historical accuracy adjustments

## Key Features

- **Claim Decomposition**: Breaks down complex claims into atomic, verifiable units
- **Multi-Source Evidence Retrieval**: Gathers evidence from web searches, knowledge bases, and news sources
- **Cross-Verification**: Uses multiple verification agents with different LLMs (Azure OpenAI and Google Gemini) to independently verify claims
- **Detailed Confidence Scoring**: Provides comprehensive confidence metrics for transparent assessment
- **API-First Design**: RESTful API for easy integration with other systems
- **Background Processing**: Support for asynchronous verification of complex claims

## Installation

### Prerequisites

- Python 3.9+
- API keys for:
  - Google Gemini (for cross-verification)
  - News API (optional, for news-based verification)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Kaos599/tathya-fact-checking-system.git
   cd tathya-fact-checking-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r fact_check_system/requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   NEWS_API_KEY=your_news_api_key
   ```

## Configuration

The system requires the following API keys to be set in your `.env` file:

- **Azure OpenAI** (optional):
  ```
  AZURE_OPENAI_API_KEY=your_azure_openai_key
  AZURE_OPENAI_ENDPOINT=your_azure_endpoint
  AZURE_OPENAI_API_VERSION=your_api_version
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
  ```

- **Tavily Search API** (optional):
  ```
  TAVILY_API_KEY=your_tavily_api_key
  ```

You can also configure the system through environment variables:

- `GEMINI_API_KEY`: Google Gemini API key for cross-verification
- `GEMINI_MODEL`: The specific Gemini model to use (default: gemini-1.5-pro)
- `NEWS_API_KEY`: News API key for news-based verification
- `HIGH_CONFIDENCE_THRESHOLD`: Threshold for high confidence (default: 0.8)
- `MEDIUM_CONFIDENCE_THRESHOLD`: Threshold for medium confidence (default: 0.5)
- `LOW_CONFIDENCE_THRESHOLD`: Threshold for low confidence (default: 0.3)
- `PORT`: Port for the API server (default: 8000)

## Usage

### Starting the API Server

```bash
python -m fact_check_system.main
```

The API will start on http://localhost:8000 by default.

### API Endpoints

#### Check a Factual Claim

```bash
curl -X POST "http://localhost:8000/check" \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Earth orbits the Sun", "background_processing": false}'
```

For complex claims that may take longer to process, use background processing:

```bash
curl -X POST "http://localhost:8000/check" \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Earth orbits the Sun", "background_processing": true}'
```

#### Check the Status of a Background Verification

```bash
curl -X GET "http://localhost:8000/status/req_1234567890_1234"
```

## API Reference

### `POST /check`

Checks a factual claim and returns a verification result.

**Request Body:**
- `claim` (string): The claim to verify
- `background_processing` (boolean, optional): Whether to process the claim in the background

**Response:**
```json
{
  "request_id": "req_1234567890_1234",
  "status": "completed",
  "result": {
    "original_claim": "The Earth orbits the Sun",
    "atomic_verdicts": {
      "claim_id_1": {
        "claim_id": "claim_id_1",
        "claim_text": "The Earth orbits the Sun",
        "verdict": "TRUE",
        "confidence": 0.95,
        "reasoning": "Multiple reliable sources confirm...",
        "evidence_summary": "Scientific consensus and observational evidence..."
      }
    },
    "overall_verdict": "TRUE",
    "overall_confidence": 0.95,
    "reasoning": "The claim is supported by scientific consensus...",
    "summary": "This claim is verified as true with high confidence..."
  },
  "processing_time": 12.34
}
```

### `GET /status/{request_id}`

Checks the status of a background verification request.

**Path Parameters:**
- `request_id` (string): The unique request ID

**Response:**
```json
{
  "request_id": "req_1234567890_1234",
  "status": "completed",
  "result": { ... },
  "processing_time": 12.34
}
```

## Development

### Project Structure

- `config.py`: Environment variables and configuration settings
- `schemas.py`: Pydantic models for all data structures
- `agents.py`: Specialized agent functions with single responsibilities
- `tools.py`: Custom tools for accessing external APIs or services
- `workflow.py`: LangGraph StateGraph definitions
- `main.py`: FastAPI application and API endpoints

### Adding a New Evidence Source

To add a new evidence source:

1. Add a new tool function in `tools.py`
2. Update the `retrieve_evidence` function in `workflow.py` to use the new tool
3. Update the confidence scoring logic if necessary

### Adding a New Verification Agent

To add a new verification agent:

1. Add a new agent function in `agents.py`
2. Update the `verify_claims` function in `workflow.py` to use the new agent
3. Update the adjudication logic to incorporate the new agent's results

### Testing

Run the tests with:

```bash
pytest
``` 