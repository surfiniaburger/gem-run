import streamlit as st
import os
import tempfile
import json
import logging
from google.cloud import secretmanager_v1
import vertexai
from vertexai import rag
from google import genai

# Page configuration
st.set_page_config(
    page_title="Document Q&A with Vertex AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Project constants
PROJECT_ID = "gem-rush-007"
SECRET_NAME = "cloud-run-invoker"
LOCATION = "us-central1"
DEFAULT_CORPUS_NAME = "user-documents-corpus"
MODEL_ID = "gemini-2.0-flash-001"

# Initialize session state variables if they don't exist
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'rag_corpus' not in st.session_state:
    st.session_state.rag_corpus = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to get secret from Secret Manager
@st.cache_resource
def get_secret(secret_name, project_id):
    """Retrieve secret from Secret Manager"""
    try:
        client = secretmanager_v1.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        service_account_json = response.payload.data.decode("UTF-8")
        
        # Parse and validate JSON
        credentials_dict = json.loads(service_account_json)
        required_fields = ['token_uri', 'client_email', 'private_key']
        
        for field in required_fields:
            if field not in credentials_dict:
                raise ValueError(f"Missing required service account field: {field}")
        
        return service_account_json
    
    except Exception as e:
        logging.error(f"Error retrieving secret {secret_name}: {e}")
        raise

# Function to initialize the RAG system
def initialize_rag_system():
    try:
        with st.spinner("Setting up the document search system..."):
            # Initialize Vertex AI with environment variables
            # This method uses the default credentials available in the environment
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            st.session_state.client = genai.Client(
                vertexai=True, 
                project=PROJECT_ID, 
                location=LOCATION
            )
            
            # Check if corpus already exists, otherwise create it
            corpora_list = rag.list_corpora()
            corpus_exists = False
            for corpus in corpora_list:
                if corpus.display_name == DEFAULT_CORPUS_NAME:
                    st.session_state.rag_corpus = corpus
                    corpus_exists = True
                    break
            
            if not corpus_exists:
                # Create new corpus
                st.session_state.rag_corpus = rag.create_corpus(
                    display_name=DEFAULT_CORPUS_NAME,
                    backend_config=rag.RagVectorDbConfig(
                        rag_embedding_model_config=rag.EmbeddingModelConfig(
                            publisher_model="publishers/google/models/text-embedding-004"
                        )
                    ),
                )
            
            st.session_state.is_initialized = True
            return True
    except Exception as e:
        st.error(f"Error initializing: {str(e)}")
        return False

# Sidebar for instructions and information
with st.sidebar:
    st.title("How to Use This Feature")
    st.markdown("""
        This feature allows you to upload documents and ask questions about their content.  It uses Google's Vertex AI Retrieval Augmented Generation (RAG) to provide accurate and context-aware answers.
        
        **1. Upload Documents:**
        
        * Go to the "Upload Documents" tab.
        * Drag and drop your files (PDF, TXT, MD, DOCX) or click to browse.
        * (Optional) Adjust "Chunk Size" and "Chunk Overlap" in "Advanced Settings".  Larger chunk sizes preserve more context. Chunk overlap helps maintain context between adjacent chunks.
        * Click "Process Documents". This uploads your files, splits them into smaller chunks, and creates embeddings for efficient searching.
        
        **2. Ask Questions:**
        
        * Go to the "Ask Questions" tab.
        * Type your question in the input box.
        * Click "Ask".  The AI will search your uploaded documents and provide an answer.
        
        **3. View Sources:**
        
        * After getting an answer, click "View Document Sources" to see the specific parts of your documents that were used to generate the response.  This helps you understand the AI's reasoning and verify the information.
        
        **4. Clear Conversation**
        * Click the "Clear Conversation" button to begin a fresh discussion.
    
        **Tips:**
        
        * Be specific with your questions for better results.
        * The AI understands natural language, so you can ask questions as you would to a person.
        * This app is powered by a large language model, so it might occasionally generate incorrect or misleading information.  Always double-check critical information.
        """)




# App title and description
st.title("📚 Document Q&A with Google AI")
st.markdown("""
Upload your documents and ask questions to get AI-powered answers based on your content.
""")

# Initialize the system if not already done
if not st.session_state.is_initialized:
    if not initialize_rag_system():
        st.stop()

# Main app interface
tab1, tab2 = st.tabs(["Upload Documents", "Ask Questions"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload Your Documents")
    st.markdown("Upload PDF, text, or markdown files to create your knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Text, Markdown, Word"
    )
    
    # Advanced options (collapsible)
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", min_value=128, max_value=2048, value=512, step=128,
                              help="Larger chunks preserve more context but may reduce relevance precision")
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=512, value=50, step=10,
                                 help="Overlap helps maintain context between chunks")
    
    # Upload button
    if uploaded_files and st.button("Process Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i / len(uploaded_files))
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                rag_file = rag.upload_file(
                    corpus_name=st.session_state.rag_corpus.name,
                    path=temp_path,
                    display_name=uploaded_file.name,
                    description="Uploaded from Q&A app"
                )
                # Clean up temp file
                os.unlink(temp_path)
            except Exception as e:
                st.error(f"Error uploading {uploaded_file.name}: {str(e)}")
                os.unlink(temp_path)
        
        progress_bar.progress(1.0)
        status_text.text("All documents processed successfully!")
        st.success("Your documents have been uploaded and processed. You can now ask questions in the 'Ask Questions' tab.")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions About Your Documents")
    
    # Show warning if no documents uploaded
    file_count = 0
    try:
        files = rag.list_files(corpus_name=st.session_state.rag_corpus.name)
        file_count = len(files)
    except:
        pass
    
    if file_count == 0:
        #st.warning("No documents found in your knowledge base. Please upload documents in the 'Upload Documents' tab first.")
        print("Let's go")
    else:
        st.markdown(f"Your knowledge base contains {file_count} document(s). Ask any question about their content.")
    
    # Question input
    user_question = st.text_input("Type your question here:", key="question_input")
    
    # Send button
    if st.button("Ask", type="primary") and user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        try:
            with st.spinner("Searching your documents..."):
                # Create a tool for the RAG Corpus
                from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
                
                rag_retrieval_tool = Tool(
                    retrieval=Retrieval(
                        vertex_rag_store=VertexRagStore(
                            rag_corpora=[st.session_state.rag_corpus.name],
                            similarity_top_k=5,
                            vector_distance_threshold=0.7,
                        )
                    )
                )
                
                response = st.session_state.client.models.generate_content(
                    model=MODEL_ID,
                    contents=user_question,
                    config=GenerateContentConfig(tools=[rag_retrieval_tool]),
                )
                
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    # Display chat history
    st.markdown("### Conversation")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
        st.markdown("---")
    
    # Clear conversation button
    if st.session_state.chat_history and st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Show sources expander (only if there's a response)
    if st.session_state.chat_history and any(msg["role"] == "assistant" for msg in st.session_state.chat_history):
        with st.expander("View Document Sources"):
            st.info("The AI generated its response based on information found in your uploaded documents.")
            try:
                # Get direct context retrieval for the last question
                if st.session_state.chat_history:
                    last_user_message = next((msg["content"] for msg in reversed(st.session_state.chat_history) 
                                             if msg["role"] == "user"), None)
                    if last_user_message:
                        contexts = rag.retrieval_query(
                            rag_resources=[rag.RagResource(rag_corpus=st.session_state.rag_corpus.name)],
                            rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
                            text=last_user_message,
                        )
                        
                        if hasattr(contexts, 'contexts') and hasattr(contexts.contexts, 'contexts'):
                            for i, context in enumerate(contexts.contexts.contexts):
                                with st.expander(f"Source {i+1}"):
                                    st.markdown(context.text)
                                    if hasattr(context, 'metadata') and context.metadata:
                                        st.caption(f"From: {context.metadata.get('source', 'Unknown')}")
            except Exception as e:
                st.warning(f"Couldn't retrieve source information: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Google Vertex AI RAG Engine")