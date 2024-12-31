from IPython.display import HTML, Markdown, display
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    VertexAISearch,
)
from google.cloud import bigquery
import os
import logging
from google.api_core import exceptions
from typing import List, Dict, Union
from datetime import datetime

import json

from IPython.display import Audio
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


PROJECT_ID = "gem-creation"  # Replace with your actual Google Cloud project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID # set this environment variable to your project ID
bq_client = bigquery.Client(project=PROJECT_ID)

client = genai.Client(vertexai=True, project="gem-creation", location="us-central1")
MODEL_ID = "gemini-2.0-flash-exp"  # @param {type: "string"}
TTS_LOCATION = "us"  # @param {type:"string"}


# Define constants
DEFAULT_LANGUAGE = "en-US"
SPEAKER_A_VOICE = "en-US-News-N"  # Choose a voice for speaker A
SPEAKER_B_VOICE = "en-US-Wavenet-H" # Choose a different voice for speaker B

SYSTEM_INSTRUCTION = """You are a podcast writer. Your task is to generate a short podcast-style dialogue between two speakers, Speaker A and Speaker B."""

response_schema = {
    "type": "object",
    "properties": {
        "dialogue": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["A", "B"]},
                    "line": {"type": "string"},
                },
                "required": ["speaker", "line"],
            },
        }
    },
    "required": ["dialogue"],
}


# Helper functions

def generate_podcast_script(file_uri: str) -> list:
    """Generates a podcast script using Gemini with controlled JSON output."""
    prompt = f"""{SYSTEM_INSTRUCTION}

    The dialogue should be engaging and natural, with each speaker contributing roughly equal amounts. Return the dialogue as a JSON array of objects, where each object has a 'speaker' (either 'A' or 'B') and a 'line' property.

    Use the following information to create the content for the podcast dialogue:
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                prompt,
                Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
            ],
            config=GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        generated_json = json.loads(response.text)
        return generated_json["dialogue"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error generating or parsing JSON script: {e}. Returning empty list.")
        return []

def synthesize_text(text: str, voice_name: str, output_filename: str):
    """Synthesizes speech from the given text using the specified voice."""
    tts_client = texttospeech.TextToSpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{TTS_LOCATION}-texttospeech.googleapis.com"
        )
    )
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=DEFAULT_LANGUAGE, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file `{output_filename}`")

def synthesize_podcast(dialogue: list[dict], output_filename: str):
    """Synthesizes speech for each speaker separately and combines the audio."""
    combined_audio = b""
    for turn_data in dialogue:
        speaker = turn_data["speaker"]
        line = turn_data["line"]
        temp_filename = f"temp_{speaker}.mp3"
        voice_name = SPEAKER_A_VOICE if speaker == "A" else SPEAKER_B_VOICE
        synthesize_text(line, voice_name, temp_filename)
        with open(temp_filename, "rb") as f:
            combined_audio += f.read()
        os.remove(temp_filename)  # Clean up temporary file

    with open(output_filename, "wb") as out:
        out.write(combined_audio)
    print(f"Combined audio content written to file `{output_filename}`")

# Call the Text-to-Speech API with script content
# Generate the podcast script from the content
# For this example, we will be using the Gemini 1.5 paper from arXiv.
# You can replace this with the URL with any publicly-accessible PDF.

PDF_URL = "gs://github-repo/2403_05530.pdf"  # @param {type: "string"}
dialogue = generate_podcast_script(PDF_URL)
print("Generated Dialogue:")
print(dialogue)

# Write the audio content into the output file
if dialogue:
    output_filename = "podcast_output.mp3"
    synthesize_podcast(dialogue, output_filename)
else:
    print("No dialogue generated. Skipping audio synthesis.")

# Listen to the audio file
if os.path.exists(output_filename):
    print("\nPlaying the generated podcast:")
    Audio(output_filename)
else:
    print("\nPodcast audio file not found.")