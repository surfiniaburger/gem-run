import pandas as pd
import numpy as np
import os
import requests # To potentially download assets if needed
from moviepy import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
) # Requires moviepy
from google.cloud import storage # To interact with GCS if paths are gs://

# Assuming vertexai and models are initialized as per the example
# Make sure to replace 'your-project-id'
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- Configuration ---
PROJECT_ID = "your-project-id"  # <--- CHANGE THIS
LOCATION = "us-central1"
GCS_BUCKET_NAME = "your-gcs-bucket-name" # <--- CHANGE THIS (if using GCS paths)
STATIC_IMAGE_FOLDER = "path/to/your/local/static_images" # Or gs:// path prefix
HR_CSV_PATH = "path/to/your/2024-mlb-homeruns.csv" # Local path or gs:// path
SCRIPT_SEGMENTS_INPUT = "path/to/your/script_segments.json" # Or generate dynamically
AUDIO_NARRATION_FOLDER = "path/to/your/audio_narrations" # Local path or gs:// path prefix
OUTPUT_VIDEO_PATH = "mlb_highlight_output.mp4"

# Initialize Vertex AI (if not already done)
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"Vertex AI already initialized or error: {e}")

# Load Embedding Model
try:
    text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    print("Text Embedding Model loaded.")
except Exception as e:
    print(f"Error loading text embedding model: {e}")
    # Handle error appropriately, maybe exit
    exit()

# --- Utility Functions (Adapted/Simplified) ---

def get_text_embedding(text: str) -> list | None:
    """Generates text embedding using the Vertex AI model."""
    if not text:
        print("Warning: Empty text provided for embedding.")
        return None
    try:
        embeddings = text_embedding_model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        print(f"Error getting embedding for text '{text[:50]}...': {e}")
        return None

def calculate_cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculates cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Ensure vectors are not zero vectors before normalization
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Cosine similarity is undefined or zero for zero vectors
        
    # Normalize vectors
    vec1_normalized = vec1_np / norm1
    vec2_normalized = vec2_np / norm2
    
    # Calculate dot product
    similarity = np.dot(vec1_normalized, vec2_normalized)
    
    # Clip similarity to handle potential floating-point inaccuracies
    return np.clip(similarity, -1.0, 1.0)


# --- Data Loading and Preparation ---

print("Loading data...")

# 1. Load Home Run Data
try:
    # Adjust if reading from GCS using pandas (e.g., prepend 'gs://')
    hr_df = pd.read_csv(HR_CSV_PATH)
    # Ensure play_id is string for consistent matching
    hr_df['play_id'] = hr_df['play_id'].astype(str)
    # Create a quick lookup dictionary {play_id: video_url}
    hr_video_lookup = pd.Series(hr_df.video_url.values, index=hr_df.play_id).to_dict()
    print(f"Loaded {len(hr_df)} home run records.")
except Exception as e:
    print(f"Error loading Home Run CSV from {HR_CSV_PATH}: {e}")
    hr_video_lookup = {} # Continue without HR videos if loading fails

# 2. Simulate/Load Static Image Bank Metadata
# In a real scenario, this would come from a database or larger metadata file.
# IMPORTANT: Replace with your actual image paths and tagging logic.
image_bank_data = [
    {"image_id": "img001", "image_path": os.path.join(STATIC_IMAGE_FOLDER, "judge_hr_swing.jpg"), "tags": ["player:99", "team:147", "action:homerun", "pose:swing"]},
    {"image_id": "img002", "image_path": os.path.join(STATIC_IMAGE_FOLDER, "ohtani_pitching.jpg"), "tags": ["player:660271", "team:119", "action:pitching"]},
    {"image_id": "img003", "image_path": os.path.join(STATIC_IMAGE_FOLDER, "generic_catch.jpg"), "tags": ["action:catch", "position:outfield"]},
    {"image_id": "img004", "image_path": os.path.join(STATIC_IMAGE_FOLDER, "dodger_stadium.jpg"), "tags": ["stadium:22", "team:119", "location:los_angeles"]},
    {"image_id": "img005", "image_path": os.path.join(STATIC_IMAGE_FOLDER, "yankee_logo.png"), "tags": ["team:147", "logo"]},
    # Add many more images with relevant tags (player IDs, team IDs, actions)
]
image_bank_df = pd.DataFrame(image_bank_data)

# Create textual descriptions from tags for embedding
def create_description_from_tags(tags):
    # Simple example: join tags. Improve this based on your tagging schema.
    desc = "Image showing: " + ", ".join(tags).replace(":", " ").replace("_", " ")
    return desc

image_bank_df['description'] = image_bank_df['tags'].apply(create_description_from_tags)
print(f"Created image bank DataFrame with {len(image_bank_df)} images.")

# 3. Generate Embeddings for Image Bank (Simulating Offline Process)
print("Generating embeddings for image bank descriptions...")
image_bank_df['embedding'] = image_bank_df['description'].apply(get_text_embedding)
# Drop rows where embedding failed
image_bank_df.dropna(subset=['embedding'], inplace=True)
print(f"Generated embeddings for {len(image_bank_df)} image descriptions.")
# !! In Production: Store image_bank_df with embeddings in a more persistent/efficient way
# !! (e.g., Feather file, database, or load into a Vector Database)

# 4. Load/Simulate Script Segments
# Each segment needs: playId (can be None), text (narration script), audio_path
# Replace with loading your actual script segments (e.g., from JSON)
script_segments = [
    {"segment_id": 1, "playId": "560a2f9b-9589-4e4b-95f5-2ef796334a94", "text": "And here's the pitch to Freeman... a towering fly ball to deep right field! It's gone! A walk-off grand slam!", "audio_path": os.path.join(AUDIO_NARRATION_FOLDER, "segment1_audio.mp3")},
    {"segment_id": 2, "playId": "some-other-play-id", "text": "Ohtani winds up and delivers a blistering fastball, strike three called!", "audio_path": os.path.join(AUDIO_NARRATION_FOLDER, "segment2_audio.mp3"), "player_tags": ["player:660271"], "action_tags": ["action:strikeout", "action:pitching"]}, # Add relevant tags for RAG
    {"segment_id": 3, "playId": None, "text": "The Dodgers take the field under the California sun.", "audio_path": os.path.join(AUDIO_NARRATION_FOLDER, "segment3_audio.mp3"), "team_tags": ["team:119"]},
    # Add more segments... ensure playId matches HR CSV if applicable
    # Add specific tags (player_tags, team_tags, action_tags) derived from stats API for better matching
]
print(f"Loaded {len(script_segments)} script segments.")


# --- RAG Retrieval Function ---

def retrieve_visual_asset(script_segment, hr_lookup, image_df):
    """
    Retrieves the best visual asset (HR video or static image) for a script segment.
    """
    play_id = script_segment.get("playId")

    # 1. Check for Home Run Video
    if play_id and play_id in hr_lookup:
        video_url = hr_lookup[play_id]
        print(f"Found HR video for playId {play_id}")
        # Basic check if URL looks valid (very simplistic)
        if video_url and video_url.lower().endswith('.mp4'):
             return {"type": "video", "path": video_url}
        else:
            print(f"Warning: Invalid video URL found for playId {play_id}: {video_url}")


    # 2. Fallback to Static Image Search using Text Embedding Similarity
    print(f"No HR video found for {play_id}. Searching static image bank...")
    script_text = script_segment.get("text", "")
    # Consider adding player/team/action tags to the query text for better matching
    # query_text = f"{script_text} {' '.join(script_segment.get('player_tags', []))} {' '.join(script_segment.get('team_tags', []))} {' '.join(script_segment.get('action_tags', []))}"
    
    query_embedding = get_text_embedding(script_text) # Use refined query_text if available

    if query_embedding is None or image_df.empty:
        print("Warning: Could not get query embedding or image bank is empty. No image found.")
        return {"type": "image", "path": None} # Or a default image path

    # Calculate Cosine Similarities
    # Note: For large image banks, this linear scan is inefficient. Use a vector DB.
    similarities = image_df['embedding'].apply(
        lambda img_emb: calculate_cosine_similarity(query_embedding, img_emb)
    )

    # Find Best Match
    if not similarities.empty:
        best_match_index = similarities.idxmax()
        best_score = similarities.max()
        best_image_path = image_df.loc[best_match_index, 'image_path']
        print(f"Found best image match: {best_image_path} with score {best_score:.4f}")
        return {"type": "image", "path": best_image_path}
    else:
        print("Warning: No similarities calculated. No image found.")
        return {"type": "image", "path": None} # Or a default image path


# --- Video Assembly ---

print("\nStarting video assembly...")
video_clips_for_concat = []
default_duration = 5 # Default duration in seconds for images if audio fails

# Optional: Setup GCS client if reading/writing from GCS
# storage_client = storage.Client(project=PROJECT_ID)

for segment in script_segments:
    print(f"\nProcessing segment {segment.get('segment_id', 'N/A')}: {segment.get('text', '')[:50]}...")

    asset = retrieve_visual_asset(segment, hr_video_lookup, image_bank_df)
    audio_path = segment.get("audio_path")
    segment_duration = default_duration

    # Load Audio to determine duration
    audio_clip = None
    try:
        # Handle GCS paths if necessary
        # if audio_path.startswith("gs://"):
        #     blob = storage.Blob.from_string(audio_path, client=storage_client)
        #     local_audio_path = f"/tmp/audio_{segment.get('segment_id', 'temp')}.mp3"
        #     blob.download_to_filename(local_audio_path)
        #     audio_path = local_audio_path # Use local path for moviepy
            
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            segment_duration = audio_clip.duration
            print(f"Loaded audio: {audio_path}, Duration: {segment_duration:.2f}s")
        else:
             print(f"Warning: Audio path missing or invalid: {audio_path}. Using default duration {segment_duration}s.")

    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}. Using default duration {segment_duration}s.")
        audio_clip = None # Ensure audio_clip is None if loading failed

    # Load Visual Asset
    visual_clip = None
    asset_path = asset.get("path")

    if not asset_path:
        print("Warning: No visual asset path found for segment. Skipping visual.")
        # Option: Create a blank clip or skip segment entirely
        # For now, create a black screen
        from moviepy import ColorClip
        visual_clip = ColorClip(size=(640, 480), color=(0,0,0), duration=segment_duration)

    elif asset['type'] == 'video':
        try:
            # Handle GCS/HTTP URLs - moviepy might handle HTTP directly, GCS needs download
            print(f"Loading video: {asset_path}")
            # Simplification: Assuming URL is accessible or downloaded locally
            # if asset_path.startswith("gs://"): download first
            # elif asset_path.startswith("http"): use directly if moviepy supports it well
            
            temp_clip = VideoFileClip(asset_path, audio=False) # Load without its own audio
            # Resize video to a standard size if needed (e.g., 1280x720)
            # temp_clip = temp_clip.resize(height=720) 
            visual_clip = temp_clip.set_duration(segment_duration)
            print("Video loaded successfully.")
        except Exception as e:
            print(f"Error loading video {asset_path}: {e}. Using fallback image logic.")
            asset['type'] = 'image' # Fallback to treating as image path if video fails
            asset['path'] = image_bank_df['image_path'].iloc[0] # Use a default image path

    # This handles both the 'image' type and the video fallback
    if asset['type'] == 'image' and not visual_clip: # Check visual_clip hasn't been set by fallback
        try:
             # Handle GCS paths if necessary
             # if asset_path.startswith("gs://"): download first

            print(f"Loading image: {asset_path}")
            if asset_path and os.path.exists(asset_path):
                # Basic Ken Burns simulation (Simple Zoom In)
                img_clip = ImageClip(asset_path).set_duration(segment_duration)
                
                # Apply a simple zoom-in effect (optional)
                # Comment out if Ken Burns effect is not needed or causes issues
                # final_size_factor = 1.1 # Zoom in by 10%
                # img_clip_resized = img_clip.resize(lambda t: 1 + (final_size_factor - 1) * t / segment_duration)
                # visual_clip = CompositeVideoClip([img_clip_resized.set_position('center')], size=img_clip.size)
                
                visual_clip = img_clip # Use without Ken Burns effect for simplicity
                
                print("Image loaded successfully.")
            else:
                print(f"Warning: Image path missing or invalid: {asset_path}. Creating blank clip.")
                from moviepy import ColorClip
                visual_clip = ColorClip(size=(640, 480), color=(0,0,0), duration=segment_duration)


        except Exception as e:
            print(f"Error loading image {asset_path}: {e}. Creating blank clip.")
            from moviepy import ColorClip
            visual_clip = ColorClip(size=(640, 480), color=(0,0,0), duration=segment_duration)


    # Combine visual and audio
    if visual_clip:
        if audio_clip:
            final_segment_clip = visual_clip.set_audio(audio_clip)
        else:
            # Visual clip exists but audio failed, use the visual with no sound
            final_segment_clip = visual_clip.set_audio(None) 
            
        video_clips_for_concat.append(final_segment_clip)
        print(f"Added segment {segment.get('segment_id', 'N/A')} to concatenation list.")
    else:
        print(f"Warning: Could not create visual clip for segment {segment.get('segment_id', 'N/A')}. Skipping segment.")
        # Clean up downloaded temp files if any
        # if 'local_audio_path' in locals() and os.path.exists(local_audio_path): os.remove(local_audio_path)


# Concatenate all processed clips
if video_clips_for_concat:
    print("\nConcatenating final video...")
    try:
        final_video = concatenate_videoclips(video_clips_for_concat, method="compose")
        # Write the final video file
        # Handle GCS paths if necessary for output
        # if OUTPUT_VIDEO_PATH.startswith("gs://"): write locally then upload
        final_video.write_videofile(
            OUTPUT_VIDEO_PATH,
            codec="libx264",
            audio_codec="aac",
            threads=4, # Adjust based on your system
            ffmpeg_params=["-preset", "fast"] # Faster encoding, potentially larger file
            # fps=24 # Optional: set a standard frame rate
        )
        print(f"\nVideo successfully generated: {OUTPUT_VIDEO_PATH}")
    except Exception as e:
        print(f"Error during final video concatenation or writing: {e}")
else:
    print("\nNo video clips were generated to concatenate.")

# --- Cleanup ---
# Delete temporary downloaded files if created (e.g., from GCS)
print("Cleanup complete (if any temp files were used).")