# TailorTalk — Titanic Dataset Chat Agent 🚢

A conversational AI chatbot that analyzes the famous Titanic dataset using LangChain, Google Gemini, FastAPI, and Streamlit. Ask questions in plain English and get text answers and visual insights.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
Create a `.env` file in the project root and add your Gemini API key:
```env
GOOGLE_API_KEY=your_key_here
```

### 3. Start the FastAPI backend
```bash
uvicorn backend.api:app --reload --port 8000
```

### 4. Start the Streamlit frontend (in a new terminal)
```bash
streamlit run frontend/app.py --server.port 8501
```

### 5. Open in browser
Navigate your browser to `http://localhost:8501`.

## Example Questions
- "What percentage of passengers were male on the Titanic?"
- "Show me a histogram of passenger ages"
- "What was the average ticket fare?"
- "How many passengers embarked from each port?"
- "Compare survival rates between males and females"
- "Show a pie chart of passenger class distribution"
