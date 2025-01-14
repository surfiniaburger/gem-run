# spanish_handler.py
from google.cloud import texttospeech
import os

def create_spanish_podcast(script_data: list, output_filename: str) -> str:
    """Handle Spanish podcast generation using the proven configuration."""
    speaker_configs = {
        "Play-by-play Announcer": {
            "voice": "es-ES-Neural2-B",
            "gender": "MALE",
            "speed": 1.1
        },
        "Color Commentator": {
            "voice": "es-ES-Neural2-C",
            "gender": "FEMALE",
            "speed": 1.0
        },
        "Player Quotes": {
            "voice": "es-ES-Neural2-D",
            "gender": "FEMALE",
            "speed": 0.95
        }
    }
    
    # Create TTS client
    client = texttospeech.TextToSpeechClient()
    combined_audio = b""
    
    for segment in script_data:
        speaker = segment['speaker']
        text = segment['text'].strip()
        
        if speaker in speaker_configs and text:
            config = speaker_configs[speaker]
            
            # Create synthesis input
            input_text = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=config["voice"],
                ssml_gender=texttospeech.SsmlVoiceGender[config["gender"]]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=config["speed"]
            )
            
            # Generate audio
            response = client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config
            )
            combined_audio += response.audio_content
            
            # Add pause between segments
            pause_response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=" "),
                voice=voice,
                audio_config=audio_config
            )
            combined_audio += pause_response.audio_content
    
    
    # Write final audio file
    with open(output_filename, "wb") as out:
        out.write(combined_audio)
    
    return output_filename