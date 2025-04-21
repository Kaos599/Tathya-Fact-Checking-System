# Tathya Fact Checking System 🕵️‍♀️

![Tathya Logo](Logo.png)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangChain-purple.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Overview

Tathya is a comprehensive fact-checking system designed to verify claims by **autonomously gathering and analyzing evidence** from multiple sources. The name "Tathya" (तथ्य) comes from Sanskrit, meaning "truth" or "reality" - perfectly embodying the system's purpose of discovering factual accuracy through a rigorous, agent-driven process. It uses a sophisticated agent powered by LLMs and LangChain to dynamically select tools, conduct research, and synthesize findings, ultimately delivering a verdict with a confidence score and detailed explanation.

## 🚀 Features

- **🤖 Agentic Workflow**: Employs an AI agent to manage the entire fact-checking process, from claim analysis to final synthesis.
- **🛠️ Dynamic Tool Selection**: The agent intelligently chooses the best tools (Search Engines, Wikidata, News APIs, Web Scrapers) based on the claim and intermediate findings.
- **🔍 Multi-source Evidence Collection**: Gathers information from diverse sources like Tavily, Google Search (via Gemini), DuckDuckGo, Wikidata, and NewsAPI.
- **🧩 Claim Decomposition**: Automatically breaks down complex claims into simpler, verifiable sub-questions using LLMs.
- **📊 Confidence Scoring**: Provides a numerical confidence score (0.0-1.0) alongside the final verdict (TRUE, FALSE, PARTIALLY TRUE/MIXTURE, UNCERTAIN).
- **📝 Detailed Explanation**: Offers a comprehensive summary explaining the agent's reasoning, citing the evidence gathered.
- **🔗 Source Attribution**: Transparently lists all sources consulted and the tools used to access them.
- **🖥️ Modern Dark Mode Interface**: Clean, user-friendly Streamlit interface with dark mode support.
- **🪜 Multi-step Verification Process**: Shows the user the agent's step-by-step reasoning and evidence gathering process.

## 🏗️ System Architecture

Tathya leverages an **agentic architecture**, orchestrated using principles often found in frameworks like LangGraph. Instead of a fixed pipeline, a central **Fact-Checking Agent** dynamically plans and executes tasks using a suite of available tools:

1.  **Core Agent**: An LLM-based agent responsible for:
    *   Understanding the claim.
    *   Planning the verification strategy.
    *   Selecting and invoking appropriate tools.
    *   Analyzing tool outputs (evidence).
    *   Synthesizing findings into a final verdict and explanation.
2.  **Tool Suite**: Functions the agent can call upon:
    *   `claim_decomposition_tool`: Breaks down complex claims.
    *   `tavily_search`, `gemini_google_search_tool`, `duckduckgo_search`: General web search tools.
    *   `news_search`: Queries NewsAPI for recent articles.
    *   `wikidata_entity_search`: Retrieves structured data from Wikidata.
    *   `scrape_webpages_tool`: Extracts content from specific URLs identified during search.
    *   _(Other potential tools)_
3.  **State Manager**: Maintains the context of the investigation, including the original claim, gathered evidence, agent's thoughts, and past actions.
4.  **REST API**: Exposes the agent's fact-checking capabilities.
5.  **Streamlit UI**: Provides the user interface for interaction and result presentation.

*The diagram below represents a high-level overview of the components the agent interacts with, rather than a strict linear pipeline.*

![mermaid-diagram-2025-04-19-123124](https://github.com/user-attachments/assets/c249b14b-9a13-4e32-a03a-27bdaca1e8dd)

## ⚙️ How the System Works (Agentic Flow)

The fact-checking process is now driven by the agent's autonomous reasoning:

1.  **User Input**: A user submits a factual claim via the Streamlit UI.
2.  **API Request**: The frontend sends the claim to the backend API, initiating the agent.
3.  **Phase 1: Initial Analysis & First Search**:
    *   The agent analyzes the claim. If complex, it uses the `claim_decomposition_tool` to break it down.
    *   It plans and executes an initial broad search using a tool like `tavily_search` or `gemini_google_search_tool`.
    *   The agent evaluates the initial results for relevance and credibility.
4.  **Phase 2: Deep Investigation**:
    *   Based on the initial findings, the agent plans its next step.
    *   It iteratively selects and uses tools (`duckduckgo_search`, `news_search`, `wikidata_entity_search`, `scrape_webpages_tool`, etc.) to gather more specific evidence, analyze contradictions, or explore different angles.
    *   After each tool call, the agent analyzes the new evidence and refines its plan. This continues until sufficient evidence (typically from at least 3 distinct sources) is gathered.
5.  **Phase 3: Final Synthesis**:
    *   Once the agent determines it has enough high-quality evidence, it concludes the investigation.
    *   It synthesizes all gathered information, determines the final verdict (TRUE, FALSE, etc.), calculates a confidence score, and writes a detailed explanation justifying the conclusion, referencing key evidence.
6.  **Presentation**: The final verdict, confidence score, explanation, step-by-step agent trace (intermediate thoughts and actions), and list of sources are presented to the user in the Streamlit interface.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required API keys stored securely (e.g., in a `.env` file):
  - OpenAI API key (or Azure OpenAI endpoint details)
  - Google AI (Gemini) API key
  - Tavily API key
  - NewsAPI key

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kaos599/tathya-fact-checking-system.git
    cd tathya-fact-checking-system
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables by creating a `.env` file in the root directory:
    ```dotenv
    # Example .env structure
    # Choose your primary LLM provider (OpenAI or Azure)
    # OPENAI_API_KEY=your_openai_api_key
    # AZURE_OPENAI_ENDPOINT=your_azure_endpoint
    # AZURE_OPENAI_API_KEY=your_azure_api_key
    # AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
    # AZURE_OPENAI_API_VERSION=api_version

    TAVILY_API_KEY=your_tavily_api_key
    GEMINI_API_KEY=your_gemini_api_key
    # Optional: Specify Gemini model if not default
    # GEMINI_MODEL=gemini-1.5-flash
    NEWS_API_KEY=your_news_api_key

    # You might need other keys depending on specific tool integrations
    ```
    *Ensure you have the necessary keys for the tools you intend the agent to use.*

### Running the Application

1.  Start the backend API server:
    ```bash
    # Navigate to the API directory if your structure requires it
    # cd fact_check_system/api
    uvicorn fact_check_system.api.main:app --reload --host 0.0.0.0 --port 8000
    # Or if using Flask/other framework, adjust the command accordingly
    # python fact_check_system/api/main.py
    ```
    The API will typically be available at `http://127.0.0.1:8000`. Check the console output.

2.  Start the Streamlit frontend in a separate terminal:
    ```bash
    streamlit run app.py
    ```
    The app will usually be available at `http://localhost:8501`.

## 🤔 Example Claims to Try

Challenge the agent with various claims:

- "Does India have the largest population as of mid-2024?"
- "Is the boiling point of water always 100 degrees Celsius?"
- "Did the James Webb Space Telescope launch before 2022?"
- "Elon Musk is the CEO of Neuralink."
- "Which team won the last FIFA World Cup?"

## 🔌 API Usage

The system provides a REST API endpoint to trigger the fact-checking agent:

### Check a Claim

```
POST /check
Content-Type: application/json

{
  "claim": "Your claim text here",
  "language": "en" // Optional, defaults might apply
}
```

**Example Response:**

```json
{
  "claim": "Your claim text here",
  "verdict": "PARTIALLY TRUE/MIXTURE", // Or TRUE, FALSE, UNCERTAIN
  "confidence_score": 0.75,
  "explanation": "Detailed explanation generated by the agent, summarizing the evidence and reasoning...",
  "intermediate_steps": [ // Optional: Could include agent's thought process
    { "thought": "Initial thought...", "action": "ToolX", "input": "...", "observation": "..." },
    // ... more steps
  ],
  "sources": [
    {
      "url": "https://example.com/source1",
      "title": "Source Title 1",
      "snippet": "Relevant excerpt from source 1...",
      "tool_used": "tavily_search"
    },
    {
      "url": "https://newssite.com/article",
      "title": "Recent News Article",
      "snippet": "Latest developments...",
      "tool_used": "news_search"
    }
    // ... other sources
  ]
}
```
*(Note: The exact response structure might vary based on implementation details, especially regarding intermediate steps.)*

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add new tools or features, please feel free to:

1.  Open an issue to discuss the change.
2.  Fork the repository.
3.  Create a new branch (`git checkout -b feature/YourFeature`).
4.  Make your changes.
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to the branch (`git push origin feature/YourFeature`).
7.  Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- Built with ❤️ using [Python](https://www.python.org/), [Streamlit](https://streamlit.io/), and [LangChain](https://www.langchain.com/).
- Leverages powerful APIs from [Tavily AI](https://tavily.com/), [Google AI (Gemini)](https://ai.google.dev/), [NewsAPI](https://newsapi.org/), [Wikidata](https://www.wikidata.org/), and potentially others.
- Inspired by the need for reliable, automated fact-checking in the digital age.

