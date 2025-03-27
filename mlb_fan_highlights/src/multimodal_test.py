# Initialize Vertex AI
import vertexai
from IPython.display import Markdown, display
from rich.markdown import Markdown as rich_Markdown
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image
from utils.intro_multimodal_rag_utils import get_document_metadata, get_similar_text_from_query

PROJECT_ID = "silver-455021"  
LOCATION = "us-central1" 
vertexai.init(project=PROJECT_ID, location=LOCATION)

text_model = GenerativeModel("gemini-2.5-pro-exp-03-25")
multimodal_model = text_model
multimodal_model_flash = text_model

query = "I need details for basic and diluted net income per share of Class A, Class B, and Class C share for google?"


# Matching user text query with "chunk_embedding" to find relevant chunks.
matching_results_text = get_similar_text_from_query(
    query,
    text_metadata_df,
    column_name="text_embedding_chunk",
    top_n=3,
    chunk_text=True,
)

# Print the matched text citations
print_text_to_text_citation(matching_results_text, print_top=False, chunk_text=True)