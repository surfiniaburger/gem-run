import streamlit as st
import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, MutableSequence
from collections.abc import MutableSequence

# Import necessary Google Cloud and Vertex AI libraries
from google.cloud import storage
from google.cloud.aiplatform_v1beta1.types import Context, RetrieveContextsResponse
import vertexai
from vertexai.preview import rag

# Import BEIR dataset libraries
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    st.warning("BEIR library not installed. Run `pip install beir` to use BEIR datasets.")

# Set page config
st.set_page_config(
    page_title="Vertex RAG Engine Evaluation",
    page_icon="🔍",
    layout="wide"
)

# Application title
st.title("Vertex RAG Engine Evaluation")
st.markdown("Evaluate retrieval quality and tune hyperparameters for Vertex AI RAG Engine")

# Helper functions
def convert_beir_to_rag_corpus(corpus: Dict[str, Dict[str, str]], output_dir: str) -> None:
    """
    Convert a BEIR corpus to Vertex RAG corpus format with a maximum of 10,000
    files per subdirectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_count, subdir_count = 0, 0
    current_subdir = os.path.join(output_dir, f"{subdir_count}")
    os.makedirs(current_subdir, exist_ok=True)

    for doc_id, doc_content in corpus.items():
        # Combine title and text (if title exists)
        full_text = doc_content.get("title", "")
        if full_text:
            full_text += "\n\n"
        full_text += doc_content["text"]

        # Create a new file for each document
        file_path = os.path.join(current_subdir, f"{doc_id}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(full_text)

        file_count += 1

        # Create a new subdirectory if the current one has reached the limit
        if file_count >= 10000:
            subdir_count += 1
            current_subdir = os.path.join(output_dir, f"{subdir_count}")
            os.makedirs(current_subdir, exist_ok=True)
            file_count = 0

    st.success(f"Conversion complete. {len(corpus)} files saved in {output_dir}")

def count_files_in_gcs_bucket(gcs_path: str) -> int:
    """
    Counts the number of files in a Google Cloud Storage path,
    excluding directories and hidden files.
    """
    bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    count = 0
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if not blob.name.endswith("/") and not any(
            part.startswith(".") for part in blob.name.split("/")
        ):
            count += 1

    return count

def count_directories_after_split(gcs_path: str) -> int:
    """
    Counts the number of directories needed based on files count.
    """
    num_files_in_path = count_files_in_gcs_bucket(gcs_path)
    num_directories = int(np.ceil(num_files_in_path / 10000))
    return num_directories

def import_rag_files_from_gcs(paths: List[str], chunk_size: int, chunk_overlap: int, corpus_name: str) -> int:
    """
    Imports files from Google Cloud Storage to a RAG corpus.
    
    Returns:
        Total number of imported files
    """
    total_imported, total_num_of_files = 0, 0
    progress_bar = st.progress(0, "Importing files...")

    for i, path in enumerate(paths):
        num_files_to_be_imported = count_files_in_gcs_bucket(path)
        total_num_of_files += num_files_to_be_imported
        max_retries, attempt, imported = 10, 0, 0
        
        while attempt < max_retries and imported < num_files_to_be_imported:
            response = rag.import_files(
                corpus_name,
                [path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                timeout=20000,
                max_embedding_requests_per_min=1400,
            )
            imported += response.imported_rag_files_count or 0
            attempt += 1
            
        total_imported += imported
        progress_bar.progress((i + 1) / len(paths), f"Imported {total_imported} files so far...")

    st.success(f"{total_imported} files out of {total_num_of_files} imported!")
    return total_imported

def extract_doc_id(file_path: str) -> str:
    """
    Extracts the document ID (filename without extension) from a file path.
    """
    try:
        parts = file_path.split("/")
        filename = parts[-1]
        filename = re.sub(r"\.\w+$", "", filename)  # Removes .txt, .pdf, .html, etc.
        return filename
    except:
        pass
    return None

def extract_retrieval_details(response: RetrieveContextsResponse) -> Tuple[str, str, float]:
    """
    Extracts the document ID, snippet, and score from a retrieval response.
    """
    doc_id = extract_doc_id(response.source_uri)
    retrieved_snippet = response.text
    distance = response.distance
    return (doc_id, retrieved_snippet, distance)

def rag_api_retrieve(query: str, corpus_name: str, top_k: int, vector_distance_threshold: float) -> MutableSequence[Context]:
    """
    Retrieves relevant contexts from a RAG corpus using the RAG API.
    """
    return rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query,
        similarity_top_k=top_k,
        vector_distance_threshold=vector_distance_threshold,
    ).contexts.contexts

def calculate_document_level_recall_precision(
    retrieved_response: MutableSequence[Context], cur_qrel: Dict[str, int]
) -> Tuple[float, float]:
    """
    Calculates the recall and precision for a list of retrieved contexts.
    """
    if not retrieved_response:
        return (0, 0)

    relevant_retrieved_unique = set()
    num_relevant_retrieved_snippet = 0
    for res in retrieved_response:
        doc_id, text, score = extract_retrieval_details(res)
        if doc_id in cur_qrel:
            relevant_retrieved_unique.add(doc_id)
            num_relevant_retrieved_snippet += 1
            
    recall = (
        len(relevant_retrieved_unique) / len(cur_qrel.keys())
        if len(cur_qrel.keys()) > 0
        else 0
    )
    precision = (
        num_relevant_retrieved_snippet / len(retrieved_response)
        if len(retrieved_response) > 0
        else 0
    )
    return (recall, precision)

def dcg_at_k_with_zero_padding_if_needed(r: List[int], k: int) -> float:
    """
    Calculates the Discounted Cumulative Gain (DCG) at a given rank k.
    """
    r = np.asarray(r)[:k]
    if r.size:
        # Pad with zeros if r is shorter than k
        if r.size < k:
            r = np.pad(r, (0, k - r.size))
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, k + 2)))
    return 0.0

def ndcg_at_k(
    retriever_results: MutableSequence[Context],
    ground_truth_relevances: Dict[str, int],
    k: int,
) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at a given rank k.
    """
    if not retriever_results:
        return 0

    # Prepare retriever results
    retrieved_relevances = []
    for res in retriever_results[:k]:
        doc_id, text, score = extract_retrieval_details(res)
        if doc_id in ground_truth_relevances:
            retrieved_relevances.append(ground_truth_relevances[doc_id])
        else:
            retrieved_relevances.append(0)  # Assume irrelevant if not in ground truth

    # Calculate DCG
    dcg = dcg_at_k_with_zero_padding_if_needed(retrieved_relevances, k)
    # Calculate IDCG
    ideal_relevances = sorted(ground_truth_relevances.values(), reverse=True)
    idcg = dcg_at_k_with_zero_padding_if_needed(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0

def calculate_document_level_metrics(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
    corpus_name: str,
    vector_distance_threshold: float,
    num_queries: int = None
) -> Dict[str, Dict[int, float]]:
    """
    Calculates and prints the average recall, precision, and NDCG for a set of queries at different top_k values.
    """
    results = {
        "recall": {},
        "precision": {},
        "ndcg": {},
    }
    
    # If num_queries is specified, randomly sample that many queries
    if num_queries and num_queries < len(queries):
        import random
        query_ids = random.sample(list(queries.keys()), num_queries)
        sampled_queries = {qid: queries[qid] for qid in query_ids}
        sampled_qrels = {qid: qrels[qid] for qid in query_ids}
    else:
        sampled_queries = queries
        sampled_qrels = qrels
        
    st.write(f"Evaluating with {len(sampled_queries)} queries")
    
    for top_k in k_values:
        start_time = time.time()
        total_recall, total_precision, total_ndcg = 0, 0, 0
        
        progress_text = f"Processing queries for top_k={top_k}"
        progress_bar = st.progress(0, progress_text)
        
        for i, (query_id, query) in enumerate(sampled_queries.items()):
            response = rag_api_retrieve(query, corpus_name, top_k, vector_distance_threshold)

            recall, precision = calculate_document_level_recall_precision(
                response, sampled_qrels[query_id]
            )
            ndcg = ndcg_at_k(response, sampled_qrels[query_id], top_k)

            total_recall += recall
            total_precision += precision
            total_ndcg += ndcg
            
            # Update progress
            progress_bar.progress((i + 1) / len(sampled_queries), 
                                 f"Processed {i+1}/{len(sampled_queries)} queries")

        end_time = time.time()
        execution_time = end_time - start_time
        num_queries = len(sampled_queries)
        
        average_recall = total_recall / num_queries
        average_precision = total_precision / num_queries
        average_ndcg = total_ndcg / num_queries
        
        results["recall"][top_k] = average_recall
        results["precision"][top_k] = average_precision
        results["ndcg"][top_k] = average_ndcg
        
        st.write(f"Top-k: {top_k}")
        st.write(f"- Average Recall@{top_k}: {average_recall:.4f}")
        st.write(f"- Average Precision@{top_k}: {average_precision:.4f}")
        st.write(f"- Average nDCG@{top_k}: {average_ndcg:.4f}")
        st.write(f"- Execution time: {execution_time:.2f} seconds")
        st.write("---")
        
    return results

# Initialize session state
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []
if "current_corpus" not in st.session_state:
    st.session_state.current_corpus = None
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "queries" not in st.session_state:
    st.session_state.queries = None
if "qrels" not in st.session_state:
    st.session_state.qrels = None

# Main app
with st.sidebar:
    st.header("Configuration")
    
    # Project and location
    project_id = st.text_input("Google Cloud Project ID", value=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    location = st.text_input("Google Cloud Region", value=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"))
    
    if st.button("Initialize Vertex AI"):
        with st.spinner("Initializing Vertex AI..."):
            try:
                vertexai.init(project=project_id, location=location)
                st.success("Vertex AI initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize Vertex AI: {str(e)}")

    # RAG Corpus Configuration
    st.subheader("RAG Corpus Configuration")
    corpus_option = st.radio(
        "Choose Corpus Option",
        ["Create New Corpus", "Use Existing Corpus"]
    )
    
    if corpus_option == "Create New Corpus":
        corpus_name = st.text_input("Corpus Name", "rag-evaluation-corpus")
        corpus_description = st.text_input("Corpus Description", "RAG evaluation corpus")
        
        embedding_models = [
            "publishers/google/models/text-embedding-004",
            "publishers/google/models/text-embedding-gecko",
            "publishers/google/models/text-multilingual-embedding-001",
            "publishers/google/models/textembedding-gecko@001",
            "publishers/google/models/textembedding-gecko@latest",
        ]
        embedding_model = st.selectbox("Embedding Model", embedding_models)
        
        if st.button("Create Corpus"):
            with st.spinner("Creating RAG Corpus..."):
                try:
                    embedding_model_config = rag.EmbeddingModelConfig(
                        publisher_model=embedding_model
                    )
                    
                    rag_corpus = rag.create_corpus(
                        display_name=corpus_name,
                        description=corpus_description,
                        embedding_model_config=embedding_model_config,
                    )
                    
                    st.session_state.current_corpus = rag_corpus
                    st.success(f"RAG Corpus created: {rag_corpus.name}")
                except Exception as e:
                    st.error(f"Failed to create RAG Corpus: {str(e)}")
    else:
        corpus_id = st.text_input("Existing Corpus ID")
        
        if st.button("Load Corpus"):
            with st.spinner("Loading RAG Corpus..."):
                try:
                    rag_corpus = rag.get_corpus(
                        name=f"projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}"
                    )
                    
                    st.session_state.current_corpus = rag_corpus
                    st.success(f"RAG Corpus loaded: {rag_corpus.name}")
                except Exception as e:
                    st.error(f"Failed to load RAG Corpus: {str(e)}")

    # Dataset Configuration
    st.subheader("Dataset Configuration")
    dataset_option = st.radio(
        "Choose Dataset Option",
        ["BEIR Dataset", "Custom GCS Path"]
    )
    
    if dataset_option == "BEIR Dataset":
        beir_datasets = [
            "fiqa", "arguana", "climate-fever", "fever", "hotpotqa", 
            "msmarco", "nfcorpus", "nq", "quora", "scidocs", "scifact", 
            "trec-covid", "webis-touche2020"
        ]
        selected_dataset = st.selectbox("Select BEIR Dataset", beir_datasets)
        dataset_split = st.selectbox("Dataset Split", ["test", "dev", "train"])
        max_queries = st.number_input("Max queries to evaluate (0 for all)", min_value=0, value=100)
        
    else:
        gcs_path = st.text_input("GCS Path to Data", "gs://your-bucket/path")
            
    # Hyperparameters
    st.subheader("Hyperparameters")
    chunk_size = st.slider("Chunk Size (tokens)", min_value=128, max_value=2048, value=512, step=64)
    chunk_overlap = st.slider("Chunk Overlap (tokens)", min_value=0, max_value=512, value=102, step=16)
    vector_distance_threshold = st.slider("Vector Distance Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    k_values = st.multiselect(
        "Top-k values to evaluate", 
        options=[1, 3, 5, 10, 20, 50, 100],
        default=[5, 10]
    )

# Main content area
if st.session_state.current_corpus:
    st.success(f"Current RAG Corpus: {st.session_state.current_corpus.name}")
    
    tabs = st.tabs(["Data Import", "Evaluation", "Results Analysis"])
    
    with tabs[0]:
        st.header("Data Import")
        
        if dataset_option == "BEIR Dataset" and not st.session_state.dataset_loaded:
            if st.button("Load BEIR Dataset"):
                if not BEIR_AVAILABLE:
                    st.error("BEIR library is not installed. Please run `pip install beir` to use BEIR datasets.")
                else:
                    with st.spinner(f"Loading {selected_dataset} dataset..."):
                        try:
                            # Download and load dataset
                            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{selected_dataset}.zip"
                            out_dir = "datasets"
                            data_path = util.download_and_unzip(url, out_dir)
                            
                            # Load the dataset
                            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=dataset_split)
                            
                            st.session_state.corpus = corpus
                            st.session_state.queries = queries
                            st.session_state.qrels = qrels
                            st.session_state.dataset_loaded = True
                            
                            st.success(f"Successfully loaded {selected_dataset} dataset with {len(corpus)} documents and {len(queries)} queries!")
                        except Exception as e:
                            st.error(f"Failed to load dataset: {str(e)}")
            
            if st.session_state.dataset_loaded:
                if st.button("Convert and Upload Dataset to RAG Corpus"):
                    with st.spinner("Converting BEIR corpus to RAG format..."):
                        converted_path = f"/tmp/converted_dataset_{selected_dataset}"
                        convert_beir_to_rag_corpus(st.session_state.corpus, converted_path)
                        
                        # Create a temporary bucket or use existing one
                        bucket_name = f"beir-{selected_dataset}-{int(time.time())}"
                        gcs_path = f"gs://{bucket_name}/{selected_dataset}"
                        
                        with st.spinner(f"Creating bucket and uploading files to {gcs_path}..."):
                            try:
                                os.system(f"gsutil mb gs://{bucket_name}")
                                os.system(f"gsutil -m rsync -r {converted_path} {gcs_path}")
                                st.success(f"Files uploaded to {gcs_path}")
                                
                                # Import files to RAG Corpus
                                with st.spinner("Importing files to RAG Corpus..."):
                                    num_subdirectories = count_directories_after_split(gcs_path)
                                    paths = [f"{gcs_path}/{i}/" for i in range(num_subdirectories)]
                                    
                                    total_imported = import_rag_files_from_gcs(
                                        paths=paths,
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        corpus_name=st.session_state.current_corpus.name,
                                    )
                                    
                                    if total_imported > 0:
                                        st.success("Dataset imported successfully to RAG Corpus!")
                            except Exception as e:
                                st.error(f"Failed to upload and import dataset: {str(e)}")
        
        elif dataset_option == "Custom GCS Path":
            if st.button("Import from GCS Path"):
                with st.spinner(f"Importing files from {gcs_path}..."):
                    try:
                        num_subdirectories = count_directories_after_split(gcs_path)
                        paths = [f"{gcs_path}/{i}/" for i in range(num_subdirectories)]
                        
                        total_imported = import_rag_files_from_gcs(
                            paths=paths,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            corpus_name=st.session_state.current_corpus.name,
                        )
                        
                        if total_imported > 0:
                            st.success("Files imported successfully to RAG Corpus!")
                            
                            # Load custom queries and qrels
                            if st.checkbox("I have custom queries and relevance judgments"):
                                st.info("Please provide a path to your queries and qrels files")
                    except Exception as e:
                        st.error(f"Failed to import files: {str(e)}")
    
    with tabs[1]:
        st.header("Evaluation")
        
        if st.session_state.dataset_loaded:
            st.write(f"Dataset: {selected_dataset}")
            st.write(f"Number of documents: {len(st.session_state.corpus)}")
            st.write(f"Number of queries: {len(st.session_state.queries)}")
            
            col1, col2 = st.columns(2)
            with col1:
                sample_query_id = st.selectbox("Sample Query ID", list(st.session_state.queries.keys())[:10])
                if sample_query_id:
                    st.write(f"Query: {st.session_state.queries[sample_query_id]}")
                    st.write(f"Relevant documents: {len(st.session_state.qrels[sample_query_id])}")
            
            with col2:
                if sample_query_id:
                    sample_results = rag_api_retrieve(
                        st.session_state.queries[sample_query_id], 
                        st.session_state.current_corpus.name, 
                        5,
                        vector_distance_threshold
                    )
                    
                    if sample_results:
                        retrieved_docs = []
                        for res in sample_results:
                            doc_id, snippet, score = extract_retrieval_details(res)
                            is_relevant = "✅" if doc_id in st.session_state.qrels[sample_query_id] else "❌"
                            retrieved_docs.append({
                                "Doc ID": doc_id,
                                "Relevant": is_relevant,
                                "Score": score
                            })
                        
                        st.write("Retrieved documents:")
                        st.dataframe(pd.DataFrame(retrieved_docs))
            
            # Run evaluation
            st.subheader("Run Evaluation")
            num_eval_queries = st.number_input(
                "Number of queries to evaluate (0 for all)", 
                min_value=0, 
                max_value=len(st.session_state.queries),
                value=min(100, len(st.session_state.queries))
            )
            
            run_params = {
                "corpus_name": st.session_state.current_corpus.name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vector_distance_threshold": vector_distance_threshold,
                "k_values": k_values,
                "num_queries": num_eval_queries if num_eval_queries > 0 else None
            }
            
            if st.button("Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    try:
                        results = calculate_document_level_metrics(
                            st.session_state.queries,
                            st.session_state.qrels,
                            k_values,
                            st.session_state.current_corpus.name,
                            vector_distance_threshold,
                            num_eval_queries if num_eval_queries > 0 else None
                        )
                        
                        # Save results
                        st.session_state.evaluation_results.append({
                            "params": run_params,
                            "results": results,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.success("Evaluation completed successfully!")
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")
        else:
            st.info("Please load a dataset first on the Data Import tab.")
    
    with tabs[2]:
        st.header("Results Analysis")
        
        if st.session_state.evaluation_results:
            # Select results to analyze
            result_options = [f"Run {i+1}: {r['timestamp']} (chunk_size={r['params']['chunk_size']}, overlap={r['params']['chunk_overlap']})" 
                             for i, r in enumerate(st.session_state.evaluation_results)]
            
            selected_result_idx = st.selectbox("Select results to analyze", range(len(result_options)), format_func=lambda x: result_options[x])
            
            if selected_result_idx is not None:
                result = st.session_state.evaluation_results[selected_result_idx]
                
                # Show parameters
                st.subheader("Parameters")
                params_df = pd.DataFrame({
                    "Parameter": list(result["params"].keys()),
                    "Value": list(result["params"].values())
                })
                st.dataframe(params_df)
                
                # Show metrics
                st.subheader("Metrics")
                
                # Plot metrics
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                metrics = ["recall", "precision", "ndcg"]
                titles = ["Recall@k", "Precision@k", "NDCG@k"]
                
                for i, (metric, title) in enumerate(zip(metrics, titles)):
                    k_values = list(result["results"][metric].keys())
                    values = list(result["results"][metric].values())
                    
                    axes[i].plot(k_values, values, 'o-', linewidth=2)
                    axes[i].set_title(title)
                    axes[i].set_xlabel("k")
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show metrics table
                metrics_data = {
                    "k": list(result["results"]["recall"].keys())
                }
                
                for metric in metrics:
                    metrics_data[metric.capitalize()] = [
                        f"{result['results'][metric][k]:.4f}" for k in metrics_data["k"]
                    ]
                
                st.dataframe(pd.DataFrame(metrics_data))
                
                # Export results
                if st.button("Export Results as CSV"):
                    csv_data = pd.DataFrame(metrics_data)
                    csv = csv_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"rag_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Recommendations based on metrics
                st.subheader("Recommendations")
                
                recall_5 = result["results"]["recall"].get(5, 0)
                precision_5 = result["results"]["precision"].get(5, 0)
                ndcg_5 = result["results"]["ndcg"].get(5, 0)
                
                recommendations = []
                
                if recall_5 < 0.5:
                    recommendations.append("- Low recall: Consider decreasing chunk size, increasing chunk overlap, or increasing top-k")
                
                if precision_5 < 0.3:
                    recommendations.append("- Low precision: Consider increasing chunk size, decreasing chunk overlap, or decreasing top-k")
                
                if ndcg_5 < 0.4:
                    recommendations.append("- Low nDCG: Consider trying a different embedding model that might better capture relevance")
                
                if not recommendations:
                    recommendations.append("- Your current settings are performing well!")
                    
                for rec in recommendations:
                    st.write(rec)
        else:
            st.info("No evaluation results available. Run an evaluation first.")
            
    # Cleanup
    if st.sidebar.button("Delete RAG Corpus"):
        if st.session_state.current_corpus:
            if st.sidebar.checkbox("I understand this will permanently delete the corpus"):
                with st.spinner("Deleting RAG Corpus..."):
                    try:
                        rag.delete_corpus(st.session_state.current_corpus.name)
                        st.session_state.current_corpus = None
                        st.success("RAG Corpus deleted successfully")
                    except Exception as e:
                        st.error(f"Failed to delete RAG Corpus: {str(e)}")
else:
    st.info("Please create or load a RAG Corpus to get started.")

# Add footer
st.markdown("---")
st.markdown("RAG Engine Evaluation - powered by Vertex AI")