# Tathya Fact Checking System

![Tathya Logo](Logo.png)

## Overview

Tathya is a comprehensive fact-checking system designed to verify claims by gathering evidence from multiple sources and providing a structured analysis with confidence scores. The system uses a combination of search engines, knowledge bases, and LLM-powered analysis to deliver accurate verdicts on factual claims.

## Features

- **Multi-source Evidence Collection**: Gathers information from Google Search, DuckDuckGo, Wikidata, NewsAPI, and Tavily
- **Claim Decomposition**: Breaks complex claims into simpler, verifiable components
- **Confidence Scoring**: Provides a confidence score with each verdict
- **Detailed Explanation**: Offers a 150-word summary explaining the reasoning behind the verdict
- **Source Attribution**: Transparently shows all sources consulted during fact-checking
- **Modern Dark Mode Interface**: Clean, user-friendly interface with dark mode for comfortable reading

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
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
   
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-pro
   NEWS_API_KEY=your_news_api_key
   ```

### Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## How to Use

1. Enter a factual claim in the search bar
2. Wait for the system to analyze the claim and gather evidence
3. Review the verdict, confidence score, and detailed explanation
4. Examine the evidence sources for additional information

## Example Claims to Try

- "Does India have the largest population?"
- "Is water boiling point 100 degrees Celsius?"
- "Did the COVID-19 pandemic start in 2019?"
- "Is Mount Everest the tallest mountain on Earth?"

## System Architecture

The Tathya Fact Checking System consists of the following components:

1. **Claim Analyzer**: Decomposes complex claims into verifiable units
2. **Evidence Gatherer**: Collects information from multiple sources
3. **Verification Engine**: Assesses the credibility and relevance of evidence
4. **Synthesis Module**: Combines all evidence to form a final verdict
5. **User Interface**: Presents the results in an intuitive way

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the API providers that make this system possible
- Built with Streamlit for the web interface

