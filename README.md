# ⚖️ Arbitration Amount Predictor

This project is a Streamlit web application designed to predict potential arbitration awards by leveraging Large Language Models (LLMs) and web search analysis. It provides an estimated arbitration amount and insights by analyzing similar, publicly available legal cases.

The application takes a user's description of an arbitration case, enhances the query using an LLM, searches for relevant cases online, crawls the content, ranks the results for relevance, and finally uses another LLM to generate a detailed analysis and amount estimation.

## Features

- **🧠 Smart Search**: Uses Groq's LLM to expand a simple case description into optimized, targeted search queries.
- **🔍 Multi-Source Search**: Searches across legal databases, case law repositories, and legal news sites via the SERP API to find relevant precedents.
- **📄 Web Crawling**: Asynchronously crawls the top search results to fetch full case texts.
- **🤖 AI-Powered Ranking**: Employs a Legal-BERT model to rank retrieved documents based on semantic similarity to the user's case.
- **💰 AI Analysis & Estimation**: Generates a comprehensive report with case analogies, strategic insights, and an estimated arbitration amount using a powerful LLM.
- **📊 Interactive UI**: A user-friendly interface built with Streamlit that displays progress, results, and the final analysis.
- **📥 Downloadable Reports**: Allows users to download the complete analysis as a Markdown file.

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

```
.
├── .env                # For storing API keys
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Python dependencies
├── src/                # Source code directory
│   ├── __init__.py
│   ├── analysis/         # LLM query enhancement and final analysis
│   │   ├── __init__.py
│   │   └── llm_analysis.py
│   ├── crawling/         # Web crawling and HTML parsing
│   │   ├── __init__.py
│   │   └── crawler.py
│   ├── ranking/          # Document ranking using Legal-BERT
│   │   ├── __init__.py
│   │   └── ranker.py
│   ├── searching/        # SERP API search logic
│   │   ├── __init__.py
│   │   └── serpapi_search.py
│   ├── utils/            # Utility functions (e.g., embedding, model loading)
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── graph.py          # Defines the main workflow using LangGraph
│   └── state.py          # Defines the state object for the graph
└── tests/              # Test files
    └── __init__.py
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

The application requires API keys for **SERPAPI** and **Groq**.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API keys to the `.env` file as follows:

    ```env
    SERPAPI_API_KEY="your_serpapi_api_key"
    GROQ_API_KEY="your_groq_api_key"
    ```

## How to Run the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default web browser.

## How It Works

The application's workflow is managed by a `LangGraph` state machine, which proceeds through the following nodes:

1.  **`search`**: The initial user query is enhanced by an LLM. The enhanced queries are then used to perform a multi-faceted search using the SERP API.
2.  **`crawl`**: The URLs from the search results are crawled asynchronously to fetch the full HTML content, which is then parsed into clean text.
3.  **`rank`**: The text from the crawled documents is embedded using a Legal-BERT model. The documents are then ranked based on the cosine similarity between their embeddings and the user's query embedding.
4.  **`llm_analysis`**: The top-ranked documents are passed to a Groq LLM along with the original query. The LLM generates a detailed report, including case analogies, strategic insights, and an estimated arbitration amount.
5.  **Display**: The final results, including the ranked list of cases and the AI-generated analysis, are displayed to the user in the Streamlit interface.
