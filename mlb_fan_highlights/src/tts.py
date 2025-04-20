# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import necessary libraries
from google.cloud import texttospeech
from pydub import AudioSegment # For combining audio files
import os

# --- Configuration ---

# !!! IMPORTANT: Authentication !!!
# Assumes Application Default Credentials (ADC) are set up.
# Run `gcloud auth application-default login` in your terminal first.
# Alternatively, set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/keyfile.json"

# Sample conversational text, split into lines (representing alternating speakers)
conversation_lines = [
    "Hello and welcome to the AI Insights podcast!",
    "Thanks! It's great to be here. Today we're talking about text-to-speech, right?",
    "Exactly! We'll explore how easy it is to generate realistic voices.",
    "Like this one? Or perhaps, like *this* one?",
    "Precisely! Alternating voices makes conversations much more engaging.",
    "It definitely prevents monotony. So, how is this actually done?",
    "We use an API, feed it text line by line, and specify different voice profiles.",
    "Ah, so the system generates separate audio snippets for each line?",
    "Correct. Then, we stitch those snippets together into the final audio track.",
    "Clever! It sounds complex, but the results can be quite impressive.",
    "Indeed. And with modern AI, the voices sound increasingly natural.",
    "Well, thanks for explaining the process!",
    "You're welcome! Join us next time on AI Insights.",
]

# The desired final output filename for the combined audio
final_output_filename = "combined_dialogue.mp3"

# Voice selection parameters (the two voices to alternate between)
language_code = "en-US"
voice_name_1 = "en-US-Chirp3-HD-Aoede" # Voice for speaker 1 (even lines: 0, 2, 4...)
voice_name_2 = "en-US-Chirp3-HD-Puck"  # Voice for speaker 2 (odd lines: 1, 3, 5...)

# Audio encoding format for individual parts and final output
audio_encoding = texttospeech.AudioEncoding.MP3

# --- TTS Conversion and Combination ---

try:
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

    temp_audio_files = [] # To store the filenames of individual parts

    print("Starting TTS generation for each line...")
    # Loop through each line of the conversation
    for count, line in enumerate(conversation_lines):
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=line)

        # Choose the voice for the current line, alternating between hosts
        if count % 2 == 0:
            current_voice_name = voice_name_1
            speaker_num = 1
        else: # count % 2 == 1
            current_voice_name = voice_name_2
            speaker_num = 2

        print(f"  Generating line {count} (Speaker {speaker_num}, Voice: {current_voice_name}): '{line[:40]}...'")

        # Configure voice parameters: language and voice name
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=current_voice_name,
        )

        # Generate audio using the Text-to-Speech API
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save the generated audio to a temporary MP3 file
        temp_filename = f"temp_part-{str(count)}.mp3"
        temp_audio_files.append(temp_filename)
        with open(temp_filename, "wb") as out:
            out.write(response.audio_content)
            # print(f"    Audio content written to temporary file {temp_filename}")

    print("\nFinished generating individual audio parts.")
    print("Combining audio files...")

    # Combine the audio files using pydub
    # Initialize with a short silence for padding at the start
    full_audio = AudioSegment.silent(duration=200) # 200 milliseconds silence

    for file in temp_audio_files:
        try:
            segment = AudioSegment.from_mp3(file)
            # Add the segment and a short silence after it
            full_audio += segment + AudioSegment.silent(duration=300) # 300ms silence between lines
        except Exception as pydub_error:
            print(f"    Error processing file {file} with pydub: {pydub_error}")
            print(f"    Skipping this segment.")
        finally:
             # Clean up the temporary file immediately after processing
            try:
                os.remove(file)
                # print(f"    Removed temporary file: {file}")
            except OSError as e:
                print(f"    Error removing temporary file {file}: {e}")


    # Export the final combined audio
    full_audio.export(final_output_filename, format="mp3")
    print(f'\nFinal combined audio content written to file "{final_output_filename}"')


except Exception as e:
    print(f"\nAn error occurred during the process: {e}")
    print("Please ensure you have:")
    print("1. Installed libraries: pip install google-cloud-texttospeech pydub")
    print("2. Authenticated with Google Cloud (e.g., `gcloud auth application-default login`)")
    print("3. Enabled the Text-to-Speech API in your Google Cloud project.")
    print("4. Installed ffmpeg (required by pydub for MP3 handling). See pydub documentation.")

    # Attempt to clean up any temp files if an error occurred mid-process
    print("Attempting to clean up any remaining temporary files...")
    for temp_file in temp_audio_files:
         if os.path.exists(temp_file):
              try:
                  os.remove(temp_file)
                  print(f"  Removed temporary file: {temp_file}")
              except OSError as e_clean:
                  print(f"  Error removing temporary file {temp_file}: {e_clean}")

# --- End of Script ---