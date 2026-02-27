"""
LangChain Agent for Titanic Dataset Analysis.

Two-way tool approach:
  1. LLM receives dataset schema + context via lightweight info tools (token-optimized)
  2. LLM generates Python code via PythonAnalysis tool for metrics & visualizations
  3. Tool results are returned to LLM for verification and natural language summary
"""

import os
import io
import sys
import base64
import traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "titanic.csv")

_df_cache = None


def load_dataset() -> pd.DataFrame:
    """Load & cache the Titanic dataset."""
    global _df_cache
    if _df_cache is None:
        if not os.path.exists(DATA_PATH):
            from data.download_data import download_titanic_data
            download_titanic_data()
        _df_cache = pd.read_csv(DATA_PATH)
    return _df_cache.copy()


# Compact schema string — keeps tokens low while giving LLM full context
_SCHEMA_CACHE = None


def _build_schema() -> str:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    df = load_dataset()
    lines = [f"{len(df)} rows × {len(df.columns)} cols", ""]
    col_desc = {
        "PassengerId": "Unique ID",
        "Survived": "0=No 1=Yes",
        "Pclass": "1st/2nd/3rd class",
        "Name": "Passenger name",
        "Sex": "male/female",
        "Age": "Age in years",
        "SibSp": "Siblings/spouses aboard",
        "Parch": "Parents/children aboard",
        "Ticket": "Ticket number",
        "Fare": "Ticket fare (£)",
        "Cabin": "Cabin number",
        "Embarked": "C=Cherbourg Q=Queenstown S=Southampton",
    }
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        desc = col_desc.get(col, "")
        lines.append(f"  {col} ({dtype}, {nulls} nulls) — {desc}")

    _SCHEMA_CACHE = "\n".join(lines)
    return _SCHEMA_CACHE


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code with access to df, pd, plt, sns, np.
    Returns {'text_output': str, 'image_base64': str|None}.
    """
    df = load_dataset()

    # Dark-themed plot defaults
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#1a1a2e",
        "text.color": "#FAFAFA",
        "axes.labelcolor": "#FAFAFA",
        "xtick.color": "#FAFAFA",
        "ytick.color": "#FAFAFA",
        "axes.edgecolor": "#333366",
        "grid.color": "#333366",
        "grid.alpha": 0.3,
    })
    plt.close("all")

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    result = {"text_output": "", "image_base64": None}

    try:
        local_vars = {
            "df": df,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "np": __import__("numpy"),
        }
        exec(code, {"__builtins__": __builtins__}, local_vars)

        # Capture plot if one exists
        fig = plt.gcf()
        if fig.get_axes():
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            img_buf.seek(0)
            result["image_base64"] = base64.b64encode(img_buf.read()).decode()
            plt.close("all")

        result["text_output"] = buf.getvalue()
        if not result["text_output"] and not result["image_base64"]:
            result["text_output"] = "Code executed successfully (no output)."

    except Exception as e:
        result["text_output"] = f"Error: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        plt.close("all")

    return result


# ---------------------------------------------------------------------------
# LangChain Tool functions
# ---------------------------------------------------------------------------

def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences that LLMs often wrap code in."""
    import re
    code = code.strip()
    # Match ```python ... ``` or ``` ... ```
    pattern = r'^```(?:python|py)?\s*\n(.*?)```\s*$'
    match = re.match(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code


_last_visualization = None

def _tool_python(code: str) -> str:
    """Run analysis / charting code on the Titanic DataFrame."""
    global _last_visualization
    code = _strip_code_fences(code)
    res = execute_python_code(code)
    out = res["text_output"]
    if res["image_base64"]:
        _last_visualization = res["image_base64"]
        out += "\n[VISUALIZATION_GENERATED]"
    return out


def _tool_schema(_: str = "") -> str:
    """Return compact dataset schema."""
    return _build_schema()


def _tool_head(_: str = "") -> str:
    """Return first 5 rows (compact)."""
    df = load_dataset()
    return df.head(5).to_string(index=False)


tools = [
    Tool(
        name="PythonAnalysis",
        func=_tool_python,
        description=(
            "Execute Python code on the Titanic DataFrame. "
            "Available variables: df (DataFrame), pd, plt, sns, np. "
            "Use print() to output results. "
            "For charts, use plt/sns — plot is auto-captured. "
            "Use vibrant colors like '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'. "
            "Input: Python code string."
        ),
    ),
    Tool(
        name="DatasetSchema",
        func=_tool_schema,
        description="Get column names, dtypes, null counts, and descriptions. No input needed.",
    ),
    Tool(
        name="DatasetHead",
        func=_tool_head,
        description="See first 5 rows of the dataset. No input needed.",
    ),
]


# ---------------------------------------------------------------------------
# Prompt — kept concise for token efficiency
# ---------------------------------------------------------------------------

AGENT_PROMPT = PromptTemplate.from_template(
    """You are TitanicBot 🚢, a data analyst for the Titanic dataset.

Tools: {tools}
Tool names: {tool_names}

Rules:
1. Use DatasetSchema first if you need column info.
2. ALWAYS use PythonAnalysis to compute answers — never guess numbers.
3. Use print() in code to output results.
4. For charts: add title, axis labels, use vibrant colors on dark background.
5. Verify your code output before answering.
6. Give a concise, friendly answer with key numbers highlighted.
7. Round numbers to 2 decimal places.

Format:
Question: {{input}}
Thought: think step by step
Action: tool name
Action Input: tool input
Observation: tool result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: friendly summary with data

Question: {input}
Thought:{agent_scratchpad}"""
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agent() -> AgentExecutor:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.1,
        max_tokens=2048,
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=AGENT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=12,             # allow enough cycles for complex queries
        max_execution_time=120,        # 2 minute safety timeout
        return_intermediate_steps=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent_query(query: str) -> dict:
    """
    Run a user query through the agent.
    Returns: {'answer': str, 'visualization': str|None}
    Includes retry logic for rate-limited APIs.
    """
    global _last_visualization
    _last_visualization = None

    import time

    executor = create_agent()

    max_retries = 3
    retry_delays = [15, 30, 60]  # seconds between retries

    last_error = None
    for attempt in range(max_retries):
        try:
            result = executor.invoke({"input": query})
            break  # success
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate limit / quota errors
            if any(kw in error_str for kw in ["quota", "rate", "429", "resource_exhausted"]):
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    print(f"[Retry {attempt+1}/{max_retries}] Rate limited. Waiting {delay}s...")
                    time.sleep(delay)
                    continue
            return {"answer": f"Error: {e}", "visualization": None}
    else:
        return {"answer": f"Rate limit exceeded after {max_retries} retries. Please wait a minute and try again.\n\nDetails: {last_error}", "visualization": None}

    answer = result.get("output", "Sorry, I couldn't process that. Please try again.")

    # Extract visualisation from state (avoids re-executing LLM code)
    visualization = _last_visualization

    return {"answer": answer, "visualization": visualization}
