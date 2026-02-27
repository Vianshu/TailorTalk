# TailorTalk — Titanic Dataset Chat Agent 🚢

A conversational AI chatbot that analyzes the famous Titanic dataset. Ask questions in plain English and get both text answers and visual insights.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python + FastAPI |
| **Agent** | LangChain + Google Gemini |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib + Seaborn |

## Architecture

```
User Question (Streamlit) 
    → FastAPI /chat endpoint
        → LangChain ReAct Agent (Gemini 2.0 Flash)
            → Tools:
                • DatasetSchema  — column info (token-optimized)
                • DatasetHead    — sample rows
                • PythonAnalysis — execute pandas/matplotlib/seaborn code
            ← Tool results verified by LLM
        ← Structured response (text + optional base64 chart)
    ← Rendered in Streamlit chat UI
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Create .env file in project root
echo GOOGLE_API_KEY=your_key_here > .env
```

### 3. Download the dataset
```bash
python data/download_data.py
```

### 4. Start the FastAPI backend
```bash
uvicorn backend.api:app --reload --port 8000
```

### 5. Start the Streamlit frontend (in a new terminal)
```bash
streamlit run frontend/app.py --server.port 8501
```

### 6. Open in browser
Navigate to `http://localhost:8501`

## Example Questions
- "What percentage of passengers were male on the Titanic?"
- "Show me a histogram of passenger ages"
- "What was the average ticket fare?"
- "How many passengers embarked from each port?"
- "Compare survival rates between males and females"
- "Show a pie chart of passenger class distribution"

## Project Structure
```
TailorTalk/
├── backend/
│   ├── __init__.py
│   ├── agent.py        # LangChain agent + tools
│   └── api.py          # FastAPI server
├── frontend/
│   └── app.py          # Streamlit chat UI
├── data/
│   ├── download_data.py
│   └── titanic.csv     # (auto-downloaded)
├── .env                # API key (not committed)
├── .env.example
├── requirements.txt
└── README.md
```
