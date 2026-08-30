import os
import json

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# ENVIRONMENT
# =========================

load_dotenv()


# =========================
# CLIENTS
# =========================

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
# WEB SEARCH TOOL
# =========================

def web_search(query):

    try:

        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        results = response.get(
            "results",
            []
        )

        if not results:

            return "No search results found."

        formatted_results = []

        for result in results:

            formatted_results.append(
                {
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "content": result.get(
                        "content",
                        ""
                    ),
                    "url": result.get(
                        "url",
                        ""
                    )
                }
            )

        return json.dumps(
            formatted_results,
            ensure_ascii=False
        )

    except Exception as error:

        return f"Search error: {error}"


# =========================
# RAG DOCUMENT STORE
# =========================

documents = []

vectorizer = TfidfVectorizer(
    stop_words="english"
)

document_vectors = None


# =========================
# LOAD DOCUMENT
# =========================

def load_document(file_path):

    global documents
    global document_vectors

    try:

        reader = PdfReader(
            file_path
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        if not text.strip():

            return (
                "No readable text found "
                "in the document."
            )

        # =========================
        # CHUNKING
        # =========================

        chunk_size = 1200

        documents = [
            text[i:i + chunk_size]
            for i in range(
                0,
                len(text),
                chunk_size
            )
            if text[i:i + chunk_size].strip()
        ]

        # =========================
        # CREATE VECTORS
        # =========================

        document_vectors = (
            vectorizer.fit_transform(
                documents
            )
        )

        return (
            f"Document loaded successfully. "
            f"Created {len(documents)} chunks."
        )

    except Exception as error:

        return (
            f"Document loading error: {error}"
        )


# =========================
# DOCUMENT SEARCH TOOL
# =========================

def document_search(
    query,
    top_k=3
):

    if (
        not documents
        or document_vectors is None
    ):

        return (
            "No document is currently loaded. "
            "Please load a document first."
        )

    try:

        query_vector = (
            vectorizer.transform(
                [query]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            document_vectors
        )[0]

        top_indexes = (
            similarities.argsort()[
                -top_k:
            ][::-1]
        )

        results = []

        for index in top_indexes:

            results.append(
                {
                    "score": float(
                        similarities[index]
                    ),
                    "content": documents[index]
                }
            )

        return json.dumps(
            results,
            ensure_ascii=False
        )

    except Exception as error:

        return (
            f"Document search error: {error}"
        )


# =========================
# TOOLS
# =========================

tools = [

    # =========================
    # CALCULATOR
    # =========================

    {
        "type": "function",
        "function": {

            "name": "calculator",

            "description":
                "Calculate a mathematical expression.",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description":
                            "Mathematical expression."

                    }

                },

                "required": [
                    "expression"
                ]
            }
        }
    },


    # =========================
    # WEB SEARCH
    # =========================

    {
        "type": "function",
        "function": {

            "name": "web_search",

            "description": (
                "Search the web for current "
                "or external information."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description":
                            "Search query."

                    }

                },

                "required": [
                    "query"
                ]
            }
        }
    },


    # =========================
    # DOCUMENT SEARCH
    # =========================

    {
        "type": "function",
        "function": {

            "name": "document_search",

            "description": (
                "Search the loaded PDF document "
                "for information relevant to "
                "the user's question."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description":
                            "Question to search "
                            "inside the document."

                    }

                },

                "required": [
                    "query"
                ]
            }
        }
    }
]


# =========================
# AI AGENT
# =========================

class ResearchAgent:

    def __init__(self):

        self.name = (
            "AI Research Agent"
        )

        self.conversation_history = []

        # =========================
        # LAST ACTIVITY
        # =========================

        self.last_activity = []


    # =========================
    # RUN AGENT
    # =========================

    def run(self, goal):

        # Reset activity for new request

        self.last_activity = []

        messages = [

            {
                "role": "system",

                "content": (

                    "You are an AI Research Agent. "

                    "Understand the user's goal "
                    "and complete it using the "
                    "available tools when necessary. "

                    "Use the calculator for "
                    "mathematical calculations. "

                    "Use web_search for current "
                    "or external information. "

                    "Use document_search when the "
                    "answer should come from the "
                    "loaded document. "

                    "Choose the most appropriate "
                    "tool for the user's request. "

                    "You may use multiple tools "
                    "and multiple tool calls when "
                    "needed. "

                    "Continue working until the "
                    "user's goal is fully completed. "

                    "Use previous conversation "
                    "context when it is relevant. "

                    "When web_search is used, base "
                    "your answer on the returned "
                    "search results. "

                    "Include a Sources section "
                    "when web_search is used. "

                    "Use URLs exactly as returned "
                    "by the search tool. "

                    "Never invent sources or URLs. "

                    "When document_search is used, "
                    "base the answer on the retrieved "
                    "document content. "

                    "Do not claim information comes "
                    "from a document if it was not "
                    "retrieved from the document."
                )
            }
        ]


        # =========================
        # ADD MEMORY
        # =========================

        messages.extend(
            self.conversation_history
        )


        # =========================
        # CURRENT REQUEST
        # =========================

        messages.append(
            {
                "role": "user",
                "content": goal
            }
        )


        # =========================
        # MULTI-STEP LOOP
        # =========================

        max_iterations = 5

        for iteration in range(
            max_iterations
        ):

            response = (
                groq_client.chat.completions.create(

                    model="openai/gpt-oss-20b",

                    messages=messages,

                    tools=tools,

                    tool_choice="auto",

                    parallel_tool_calls=False,

                    temperature=0.2
                )
            )


            message = (
                response.choices[0].message
            )


            # =========================
            # FINAL ANSWER
            # =========================

            if not message.tool_calls:

                final_answer = (
                    message.content
                )

                # Save memory

                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": goal
                    }
                )

                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

                # Keep last 10 messages

                self.conversation_history = (
                    self.conversation_history[-10:]
                )

                return final_answer


            # =========================
            # ADD TOOL CALL
            # =========================

            messages.append(
                message
            )


            # =========================
            # EXECUTE TOOLS
            # =========================

            for tool_call in message.tool_calls:

                function_name = (
                    tool_call.function.name
                )


                # =========================
                # TOOL ACTIVITY
                # =========================

                activity_map = {

                    "calculator": {
                        "tool": "calculator",
                        "label": "Calculator",
                        "icon": "calculator"
                    },

                    "web_search": {
                        "tool": "web_search",
                        "label": "Web Search",
                        "icon": "globe"
                    },

                    "document_search": {
                        "tool": "document_search",
                        "label": "Document Search",
                        "icon": "file"
                    }
                }


                activity = activity_map.get(
                    function_name
                )


                if activity:

                    self.last_activity.append(
                        {
                            "step": len(
                                self.last_activity
                            ) + 1,

                            "tool": activity[
                                "tool"
                            ],

                            "label": activity[
                                "label"
                            ],

                            "icon": activity[
                                "icon"
                            ],

                            "status": "completed"
                        }
                    )


                # =========================
                # PARSE ARGUMENTS
                # =========================

                try:

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                except json.JSONDecodeError:

                    result = (
                        "Invalid tool arguments."
                    )

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                result
                        }
                    )

                    continue


                # =========================
                # CALCULATOR
                # =========================

                if function_name == "calculator":

                    result = calculator(
                        arguments[
                            "expression"
                        ]
                    )


                # =========================
                # WEB SEARCH
                # =========================

                elif function_name == "web_search":

                    result = web_search(
                        arguments[
                            "query"
                        ]
                    )


                # =========================
                # DOCUMENT SEARCH
                # =========================

                elif function_name == "document_search":

                    result = document_search(
                        arguments[
                            "query"
                        ]
                    )


                # =========================
                # UNKNOWN TOOL
                # =========================

                else:

                    result = (
                        "Unknown tool."
                    )


                # =========================
                # RETURN TOOL RESULT
                # =========================

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            result
                    }
                )


        # =========================
        # MAX ITERATIONS
        # =========================

        return (
            "The agent reached the maximum "
            "number of tool steps without "
            "completing the task."
        )


# =========================
# AGENT INSTANCE
# =========================

agent = ResearchAgent()


















    

        
