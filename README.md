# Tathya Fact Checking System

![Tathya Logo](Logo.png)

## Overview

Tathya is a comprehensive fact-checking system designed to verify claims by gathering evidence from multiple sources and providing a structured analysis with confidence scores. The system uses a combination of search engines, knowledge bases, and LLM-powered analysis to deliver accurate verdicts on factual claims.

## Features

- **Multi-source Evidence Collection**: Gathers information from Google Search, DuckDuckGo, Wikidata, NewsAPI, and Tavily
- **Claim Decomposition**: Breaks complex claims into simpler, verifiable components using LLMs
- **Confidence Scoring**: Provides a confidence score with each verdict
- **Detailed Explanation**: Offers a comprehensive summary explaining the reasoning behind the verdict
- **Source Attribution**: Transparently shows all sources consulted during fact-checking
- **Modern Dark Mode Interface**: Clean, user-friendly interface with dark mode for comfortable reading
- **Multi-step Verification Process**: Shows the user a step-by-step verification process
- **Source Tool Categorization**: Displays the tools used for evidence gathering

## System Architecture

The Tathya Fact Checking System follows a pipeline architecture with these key components:

1. **Claim Analyzer**: Decomposes complex claims into verifiable units using LLMs
2. **Evidence Gatherer**: Collects information from multiple sources:
   - Tavily API (search)
   - Google AI (Gemini) for web search and analysis
   - Wikidata for factual knowledge base queries
   - DuckDuckGo for independent web search
   - NewsAPI for current events and news articles
3. **Evidence Verifier**: Assesses the credibility and relevance of evidence
4. **Synthesis Module**: Combines all evidence to form a final verdict with explanation
5. **REST API**: Provides programmatic access to the fact-checking capabilities
6. **Streamlit UI**: Presents the results in an intuitive web interface

![mermaid-diagram-2025-04-19-123124](https://github.com/user-attachments/assets/c249b14b-9a13-4e32-a03a-27bdaca1e8dd)

## Getting Started

### Prerequisites

- Python 3.8+
- Required API keys:
  - OpenAI API key (or Azure OpenAI API key)
  - Google AI (Gemini) API key
  - Tavily API key
  - NewsAPI key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tathya-fact-checking-system.git
   cd tathya-fact-checking-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file:
   ```
   LLM_KEYS=your_llm_api_key
   
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-pro
   NEWS_API_KEY=your_news_api_key
   ```

### Running the Application

1. Start the backend API server:
   ```bash
   cd fact_check_system/api
   python main.py
   ```
   The API will be available at `http://127.0.0.1:8000`.

2. Start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```
   The app will be available at `http://localhost:8501`.

## How the System Works

1. **User Input**: The user enters a factual claim in the Streamlit interface.
2. **API Request**: The frontend sends the claim to the backend API.
3. **Claim Decomposition**: The system breaks down complex claims into simpler, verifiable parts.
4. **Evidence Collection**: Multiple sources are queried in parallel to gather relevant information.
5. **Evidence Verification**: Each piece of information is assessed for relevance and credibility.
6. **Synthesis**: All evidence is combined to form a final verdict with a confidence score.
7. **Presentation**: The results are displayed in an intuitive interface with a verdict, explanation, and source list.

## Example Claims to Try

- "Does India have the largest population?"
- "Is water boiling point 100 degrees Celsius?"
- "Did the COVID-19 pandemic start in 2019?"
- "Is Mount Everest the tallest mountain on Earth?"

## API Usage

The system provides a REST API for programmatic access:

### Check a Claim
```
POST /check
{
  "claim": "Your claim text here",
  "language": "en"
}
```

Response:
```json
{
  "claim": "Your claim text here",
  "result": "True",
  "confidence_score": 0.85,
  "explanation": "Detailed explanation...",
  "sources": [
    {
      "url": "https://example.com/source1",
      "title": "Source Title",
      "snippet": "Relevant excerpt...",
      "tool": "Tavily Search"
    }
  ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the API providers that make this system possible
- Built with Streamlit for the web interface
- Powered by LangChain for orchestration

