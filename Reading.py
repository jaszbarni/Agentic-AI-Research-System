import streamlit as st
import os
import pypdf
import re
from openai import OpenAI
import concurrent.futures
from pydantic import BaseModel, Field

class HighlightResult(BaseModel):
    phrases: list[str] = Field(description="A list of exact phrases or sentences to highlight from the text.")

DOWNLOADS_DIR = "downloads"

@st.cache_data(show_spinner="Generating summary...", persist="disk")
def get_summary(text):
    #client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        #model="llama3.2:1b",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert academic research assistant. Your task is to provide a concise, high-level summary of the main themes, objectives, and findings of the provided text. Keep the summary clear, brief, and informative. Do NOT include any conversational filler or introductions."},
            {"role": "user", "content": text[:5000]}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

@st.cache_data(show_spinner=False, persist="disk")
def get_highlighted_chunk(chunk, summary_text):
    #client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an expert reading assistant. Your task is to extract the most important key phrases or crucial sentences from the provided text. "
                                f"You MUST output your response in JSON format. The JSON must contain a single key 'phrases' containing a list of strings. "
                                f"The phrases MUST be exact substrings from the provided text. "
                                f"Use this summary of the document's theme to guide your highlights: {summary_text}"},
            {"role": "user", "content": chunk}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    highlighted_chunk = chunk
    try:
        result = HighlightResult.model_validate_json(response.choices[0].message.content)
        phrases_to_highlight = result.phrases
        
        for phrase in phrases_to_highlight:
            phrase = phrase.strip()
            if phrase and len(phrase) > 4 and phrase in highlighted_chunk:
                highlighted_chunk = highlighted_chunk.replace(phrase, f"<mark>{phrase}</mark>")
    except Exception as e:
        print(f"Failed to parse JSON for chunk: {e}")
            
    return highlighted_chunk

@st.cache_data(show_spinner="Highlighting page...", persist="disk")
def get_highlighted_page(cleaned_text, summary_text):
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len((current_chunk + sentence).split()) > 500 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_highlighted_chunk, chunk, summary_text) for chunk in chunks]
        highlighted_chunks = [future.result() for future in futures]
        
    return " ".join(highlighted_chunks)

def Reading_page(pdf_file):
    st.session_state.selected_pdf = pdf_file
    file_path = os.path.join(DOWNLOADS_DIR, st.session_state.selected_pdf)
    summary_text = ""

    reader = pypdf.PdfReader(file_path)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    col1, col2, col3 = st.columns([1, 2, 0.5])
    with col1:
        if st.button("Back to files list"):
            st.session_state.selected_pdf = None
            st.rerun()
        st.markdown("### Summary:")

        summary_text = get_summary(raw_text)
        st.markdown(summary_text)
        st.markdown("---")

    with col2:
        ai_highlight = st.toggle("AI Highlighting")
        
        if ai_highlight and st.button("Redo Highlighting"):
            get_highlighted_chunk.clear()
            
        if os.path.exists(file_path):
            st.subheader(f"Reading: {st.session_state.selected_pdf.replace('_', ' ').replace('.pdf', '')}")
            try:
                for i, page in enumerate(reader.pages):
                    raw_text = page.extract_text()
                    if raw_text:
                        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
                        
                        if cleaned_text:
                            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
                            chunks = []
                            current_chunk = ""
                            for sentence in sentences:
                                if len((current_chunk + sentence).split()) > 500 and current_chunk:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = sentence + " "
                                else:
                                    current_chunk += sentence + " "
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            
                            if ai_highlight:
                                with st.spinner("Highlighting page..."):
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        futures = [executor.submit(get_highlighted_chunk, chunk, summary_text) for chunk in chunks]
                                        highlighted_chunks = [future.result() for future in futures]
                            else:
                                highlighted_chunks = chunks
                            cleaned_text = " ".join(highlighted_chunks)
                            
                        st.markdown(cleaned_text, unsafe_allow_html=True)
                        if i < len(reader.pages) - 1:
                            st.markdown("---")
            except Exception as e:
                st.error(f"Error extracting text: {e}")
        else:
            st.error("File not found.")