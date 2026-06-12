import streamlit as st
import dotenv
import os
from openai import OpenAI
import json
from fastmcp import FastMCP
import arxiv
from pydantic import BaseModel
from typing import List, Optional
import pypdf
import re

st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)
def init_phoenix():
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor
    import webbrowser
    
    session = px.launch_app()
    webbrowser.open(session.url)
    tracer_provider = register(project_name="arxiv-research-assistant")
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

init_phoenix()

mcp = FastMCP("arXiv-Searcher")

class ArxivPaper(BaseModel):
    id: str
    title: str
    authors: List[str]
    summary: str
    pdf_url: str
    local_pdf_path: str
    download_status: str

class ArxivSearchOutput(BaseModel):
    papers: Optional[List[ArxivPaper]] = None
    error: Optional[str] = None

def search_arxiv(query: str, max_results: int) -> str:
    """
    Returns a JSON string containing the matching papers.
    Returns a JSON string of ArxivSearchOutput containing the matching papers or an error.
    """
    def dump_json(obj: BaseModel) -> str:
        return obj.model_dump_json() if hasattr(obj, 'model_dump_json') else obj.json()

    if not query or not str(query).strip():
        return dump_json(ArxivSearchOutput(error="The search query cannot be empty. Please provide a valid query string."))

    client = arxiv.Client()
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    os.makedirs("downloads", exist_ok=True)
    
    try:
        for paper in client.results(search):
            safe_name = "".join([c for c in paper.title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            filename = f"{safe_name.replace(' ', '_')}.pdf"
            filepath = os.path.join("downloads", filename)
            
            try:
                paper.download_pdf(dirpath="downloads", filename=filename)
                download_status = "Success"
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                download_status = f"Failed to download locally: {str(e)}"
                
            paper_data = {
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary.replace('\n', ' '),
                "pdf_url": paper.pdf_url,
                "local_pdf_path": filepath,
                "download_status": download_status
            }
            validated_paper = ArxivPaper(**paper_data)
            results.append(validated_paper)
    except arxiv.HTTPError as e:
        return dump_json(ArxivSearchOutput(error=f"arXiv API Rate Limit Exceeded (HTTP 429). Ask for fewer results. Details: {str(e)}"))
    except Exception as e:
        return dump_json(ArxivSearchOutput(error=f"An error occurred while searching arXiv: {str(e)}"))
        
    if not results:
        return dump_json(ArxivSearchOutput(error=f"No papers found on arXiv for the query: '{query}'"))
        
    return dump_json(ArxivSearchOutput(papers=results))

def search_local_documents(query: str, max_results: int = 5) -> str:
    """
    Returns a JSON string containing relevant chunks from local PDF files.
    """
    if not query or not str(query).strip():
        return json.dumps({"error": "The search query cannot be empty."})

    downloads_dir = "downloads"
    if not os.path.exists(downloads_dir):
        return json.dumps({"error": "Downloads directory not found."})
        
    pdf_files = [f for f in os.listdir(downloads_dir) if f.endswith('.pdf')]
    if not pdf_files:
        return json.dumps({"error": "No PDF documents found in the downloads folder."})
        
    query_terms = set(re.findall(r'\w+', query.lower()))
    if not query_terms:
        return json.dumps({"error": "Invalid search query."})
        
    results = []
    
    for pdf_file in pdf_files:
        filepath = os.path.join(downloads_dir, pdf_file)
        try:
            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
                    
            text = re.sub(r'\s+', ' ', text).strip()
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len((current_chunk + sentence).split()) > 150:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                chunk_terms = set(re.findall(r'\w+', chunk_lower))
                match_count = len(query_terms.intersection(chunk_terms))
                
                if match_count > 0:
                    results.append({
                        "file": pdf_file,
                        "chunk_index": i + 1,
                        "content": chunk,
                        "score": match_count
                    })
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:max_results]
    
    if not top_results:
        return json.dumps({"message": f"No relevant information found in local documents for '{query}'."})
        
    return json.dumps([{k: v for k, v in res.items() if k != "score"} for res in top_results], indent=2)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #808088   ;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def chat_page():
    st.title("Research System")

    dotenv.load_dotenv()

    #model_name = 'llama3.2:1b'
    #model_name = 'openrouter/owl-alpha'
    model_name = 'gpt-4o'

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": 
             "You are a helpful research assistant. "
             "When a user asks to search for papers, you MUST use the `search_arxiv` tool. "
             "When a user asks questions about downloaded documents or topics within them, use the `search_local_documents` tool. "
             "You MUST provide a valid, specific search term in the `query` parameter. NEVER leave the `query` empty. "
             "IMPORTANT: When calling a tool, do NOT output any conversational text before or after the tool call. Output ONLY the tool call."
            },
        ]
        
    client = OpenAI(
        #base_url='http://localhost:11434/v1/',
        #base_url='https://openrouter.ai/api/v1',
        #api_key=os.getenv("OPENROUTER_API_KEY"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    for message in st.session_state.messages:
        if message["role"] not in ["system", "tool"] and message.get("content"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    tools =  [{
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "searches in arxiv for papers and downloads their PDFs to local disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the query to search for on arxiv",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "the maximum number of results to return",
                    }
                },
                "required": ["query", "max_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_documents",
            "description": "Searches for topics within the locally downloaded PDF documents. Use this to read and extract information from already downloaded papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the query or topic to search for in local documents",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "the maximum number of text chunks to return",
                    }
                },
                "required": ["query"],
            },
        },
    }]
    pormpt = st.chat_input("Ask about something", key="chat_input")
    if pormpt:
        st.session_state.messages.append({"role": "user", "content": pormpt})
        with st.chat_message("user"):
            st.markdown(pormpt)

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,
                tools=tools,
                tool_choice="auto",
            )
        
        response_message = response.choices[0].message
        
        assistant_dict = {"role": "assistant", "content": response_message.content}
        if response_message.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response_message.tool_calls
            ]
        st.session_state.messages.append(assistant_dict)

        if response_message.content:
            with st.chat_message("assistant"):
                st.markdown(response_message.content)

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_arxiv":
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        st.error("The AI attempted to search but provided malformed arguments. Try asking again.")
                        
                    query = args.get("query", "")
                    try:
                        max_results = int(args.get("max_results", 1))
                    except (ValueError, TypeError):
                        max_results = 1
                    
                    if not query.strip():
                        tool_result = json.dumps({"error": "The AI failed to generate a search query. Tell the user you encountered an error."})
                        st.error("The AI failed to generate a search query. Please try asking again.")
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(f"*(Searching arXiv for up to {max_results} paper(s) for query: '{query}'...)*")
                            
                        with st.spinner("Searching and downloading papers..."):
                            tool_result = search_arxiv(query=query, max_results=max_results)
                    
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                elif tool_call.function.name == "search_local_documents":
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        st.error("The AI attempted to search local documents but provided malformed arguments.")
                        
                    query = args.get("query", "")
                    try:
                        max_results = int(args.get("max_results", 5))
                    except (ValueError, TypeError):
                        max_results = 5
                        
                    if not query.strip():
                        tool_result = json.dumps({"error": "Failed to generate a search query."})
                        st.error("The AI failed to generate a search query for local documents. Please try asking again.")
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(f"*(Searching your local documents for topic: '{query}'...)*")
                            
                        with st.spinner("Scanning downloaded PDFs..."):
                            tool_result = search_local_documents(query=query, max_results=max_results)
                            
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                else:
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": f"Unknown tool: {tool_call.function.name}"})
                    })
                    
            with st.spinner("Formulating final response..."):
                final_response = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.messages,
                )
            final_msg = final_response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": final_msg})
            with st.chat_message("assistant"):
                st.markdown(final_msg)

pages = [
    st.Page(chat_page, title="Chat", default=True),
    st.Page("Files.py", title="Files")
]

page = st.navigation(pages)
page.run()
