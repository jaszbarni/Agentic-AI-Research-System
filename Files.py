import streamlit as st
import os
import dotenv
from Reading import Reading_page

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

st.title("Files")

DOWNLOADS_DIR = "downloads"
dotenv.load_dotenv()

if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

if st.session_state.selected_pdf:
    Reading_page(st.session_state.selected_pdf)
else:
    if not os.path.exists(DOWNLOADS_DIR):
        st.info("No downloads directory found. Search for some papers first!")
    else:
        pdf_files = [f for f in os.listdir(DOWNLOADS_DIR) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.info("No PDF files found.")
        else:
            for pdf in pdf_files:
                col1, col2, col3 = st.columns([3, 0.5, 0.5])
                with col1:
                    st.write(pdf.replace("_", " ").replace(".pdf", ""))
                with col2:
                    if st.button("Read", key=f"btn_{pdf}", use_container_width=True):
                        st.session_state.selected_pdf = pdf
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"del_{pdf}", use_container_width=True, type="primary"):
                        file_path = os.path.join(DOWNLOADS_DIR, pdf)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.rerun()