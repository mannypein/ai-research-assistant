import os
import json
from dotenv import load_dotenv

load_dotenv()

# Explicitly set LangSmith env vars before any other imports
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-research-assistant")

from groq import Groq
from duckduckgo_search import DDGS
from langsmith import traceable

# Initialize LangSmith client explicitly
from langsmith import Client
langsmith_client = Client()

# Initialize Groq client directly (bypass LangChain bind_tools issue)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define tool for Groq's native format
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, facts, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_web_search(query: str) -> str:
    """Run a DuckDuckGo search and return results as a string."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title', '')}\nSummary: {r.get('body', '')}\nURL: {r.get('href', '')}")
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"

@traceable(name="research_assistant_query")
def run_agent(question: str) -> dict:
    """Run the agent with a question and return the response."""
    try:
        sources_used = []

        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI research assistant.
                You have access to a web_search tool, but only use it when absolutely necessary.
                
                Answer DIRECTLY from your own knowledge when:
                - The question is about general concepts, definitions, or explanations
                - The question is about well established facts or history
                - The question involves reasoning or analysis
                
                Only use web_search when:
                - The question explicitly asks for recent news or current events
                - The question asks for something that changes frequently like prices or standings
                - You are genuinely uncertain about a specific fact
                
                Default to answering from your own knowledge first."""
            },
            {
                "role": "user",
                "content": question
            }
        ]

        # Agentic loop
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1024
                )
            except Exception as tool_error:
                # If tool calling fails, retry without tools for a direct answer
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    max_tokens=1024
                )
                sources_used = ["Direct LLM Knowledge"]
                response_message = response.choices[0].message
                break

            response_message = response.choices[0].message

            # No tool calls means we have our final answer
            if not response_message.tool_calls:
                break

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })

            # Process each tool call
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "web_search":
                    query = tool_args.get("query", "")
                    sources_used.append("WebSearch (DuckDuckGo)")
                    tool_result = run_web_search(query)
                else:
                    tool_result = f"Tool {tool_name} not found"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
        final_answer = response_message.content or "I was unable to generate a response."

        if not sources_used:
            sources_used = ["Direct LLM Knowledge"]

        return {
            "question": question,
            "answer": final_answer,
            "sources_used": sources_used
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "answer": f"An error occurred: {str(e)}",
            "sources_used": []
        }