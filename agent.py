import os
import json

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# =========================
# CALCULATOR TOOL
# =========================

def calculator(expression):
    allowed = "0123456789+-*/(). "

    if not all(char in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )
        return str(result)

    except Exception:
        return "Could not calculate the expression."


# =========================
# SEARCH TOOL
# =========================

def web_search(query):
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        results = response.get("results", [])

        if not results:
            return "No search results found."

        formatted_results = []

        for result in results:
            formatted_results.append(
                {
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", "")
                }
            )

        return json.dumps(
            formatted_results,
            ensure_ascii=False
        )

    except Exception as error:
        return f"Search error: {error}"


# =========================
# TOOLS
# =========================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or factual information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =========================
# AI AGENT
# =========================

class ResearchAgent:

    def __init__(self):
        self.name = "AI Research Agent"

    def run(self, goal):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Research Agent. "
                    "Understand the user's goal and use tools when needed. "
                    "Use the calculator for mathematical calculations. "
                    "Use web_search when current or external information "
                    "is required. After receiving tool results, provide "
                    "a clear and useful final answer."
                )
            },
            {
                "role": "user",
                "content": goal
            }
        ]

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )

        message = response.choices[0].message

        # No tool required
        if not message.tool_calls:
            return message.content

        messages.append(message)

        # Execute requested tools
        for tool_call in message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            if function_name == "calculator":

                result = calculator(
                    arguments["expression"]
                )

            elif function_name == "web_search":

                result = web_search(
                    arguments["query"]
                )

            else:
                result = "Unknown tool."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )

        # Final response
        final_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )

        return final_response.choices[0].message.content


agent = ResearchAgent()
