import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
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
                        "description": "Mathematical expression to calculate."
                    }
                },
                "required": ["expression"]
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
                    "Understand the user's goal. "
                    "Use the calculator tool whenever "
                    "accurate mathematical calculation is required."
                )
            },
            {
                "role": "user",
                "content": goal
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )

        message = response.choices[0].message

        # No tool required
        if not message.tool_calls:
            return message.content

        # Add assistant tool-call message
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

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                )

        # Generate final answer
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2
        )

        return final_response.choices[0].message.content


agent = ResearchAgent()
