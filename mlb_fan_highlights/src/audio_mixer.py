from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from typing import List, Dict, Optional
import io
import logging
from google.cloud import logging as cloud_logging
from google.cloud import secretmanager_v1
from google.cloud import storage
from google.oauth2 import service_account
import json

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()


class MLBAudioMixer:
    def __init__(self, project_id, secret_name):
        service_account_json = self._get_secret(secret_name, project_id)
        if service_account_json:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
        else:
            # Fallback error handling
            raise Exception("Could not retrieve service account credentials")

        # Initialize storage client
        self.storage_client = storage.Client(
            project=project_id,
            credentials=credentials
        )
        self.bucket_name = "mlb-audio-assets"

        # Sound effects and music paths
        sound_effect_paths = {
            "crowd_cheer": "assets/sounds/crowd_cheer.mp3",
            "bat_hit": "assets/sounds/bat_hit.mp3",
            "crowd_tension": "assets/sounds/crowd_tension.mp3",
            "walkup_music": "assets/sounds/walkup_music.mp3",
            "stadium_ambience": "assets/sounds/stadium_ambience.mp3"
        }
        
        music_paths = {
            "intro": "assets/music/opener.mp3",
            "highlight": "assets/music/highlight.mp3",
            "outro": "assets/music/opener.mp3"
        }

        # Load sound effects and music
        self.sound_effects = {
            name: self._load_audio_from_gcs(path) 
            for name, path in sound_effect_paths.items()
        }

        self.background_music = {
            name: self._load_audio_from_gcs(path) 
            for name, path in music_paths.items()
        }

        
        # Niveles de volumen refinados
        self.VOICE_VOLUME = 0
        self.MUSIC_VOLUME = -25
        self.SFX_VOLUME = -18
        self.AMBIENCE_VOLUME = -30
        
        # Constantes de tiempo (en milisegundos)
        self.SPEAKER_PAUSE = 850
        self.CROSSFADE_LENGTH = 400
        self.INTRO_FADE = 2000
        self.EFFECT_FADE = 600

    def _get_secret(self, secret_name, project_id):
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

    def _load_audio_from_gcs(self, blob_path):
        """Load audio file from GCS bucket"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download audio content
            audio_content = blob.download_as_bytes()
            
            # Convert to AudioSegment
            audio_buffer = io.BytesIO(audio_content)
            return AudioSegment.from_mp3(audio_buffer)
        except Exception as e:
            logging.error(f"Error loading audio from {blob_path}: {e}")
            # Fallback to silent audio if loading fails
            return AudioSegment.silent(duration=1000)
      
    def _compress_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("compressing audio")
        """Apply compression to prevent audio peaks and crackling."""
        return compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=10, release=100)

    def _fade_effect(self, effect: AudioSegment) -> AudioSegment:
        logging.info("fading effect")
        """Apply smooth fading to sound effects."""
        return effect.fade_in(200).fade_out(self.EFFECT_FADE)

    def _process_voice_segment(self, voice_audio: AudioSegment) -> AudioSegment:
        logging.info("processing voice segments")
        """Process voice segments with compression and normalization."""
        # First normalize to ensure consistent volume
        voice_audio = self._normalize_audio(voice_audio)
        # Apply compression to prevent peaks
        voice_audio = self._compress_audio(voice_audio)
        # Final volume adjustment
        return voice_audio - abs(self.VOICE_VOLUME)

    def mix_podcast_audio(self, voice_segments: List[Dict[str, bytes]], 
                         include_background: bool = True) -> AudioSegment:
        """
        Mix podcast audio with improved handling of high-intensity moments.
        """
        final_mix = self.background_music["intro"].fade_in(self.INTRO_FADE)
        final_mix = self._normalize_audio(final_mix)
        
        if include_background:
            ambience = self.sound_effects["stadium_ambience"]
            ambience = ambience - abs(self.AMBIENCE_VOLUME)
            total_duration = sum(len(AudioSegment.from_mp3(io.BytesIO(segment["audio"]))) 
                               for segment in voice_segments)
            total_duration += len(voice_segments) * self.SPEAKER_PAUSE
            
            while len(ambience) < total_duration:
                ambience += ambience
            
            final_mix = final_mix.overlay(ambience[:len(final_mix)])

        previous_speaker = None
        for i, segment in enumerate(voice_segments):
            audio_bytes = io.BytesIO(segment["audio"])
            voice_audio = AudioSegment.from_mp3(audio_bytes)
            
            # Process voice audio
            voice_audio = self._process_voice_segment(voice_audio)
            
            # Handle sound effects with improved timing
            triggers = self._detect_event_triggers(segment["text"])
            if triggers:
                # Create a blank segment for effects
                effect_mix = AudioSegment.silent(duration=len(voice_audio))
                
                for trigger in triggers:
                    effect = self.sound_effects[trigger]
                    effect = self._fade_effect(effect)
                    effect = effect - abs(self.SFX_VOLUME)
                    
                    # Position effect slightly before the voice for home runs
                    if trigger == "crowd_cheer" and "home run" in segment["text"].lower():
                        # Start effect earlier and let it fade under the voice
                        voice_audio = AudioSegment.silent(duration=200) + voice_audio
                        effect_position = 0
                    else:
                        effect_position = 100
                    
                    effect_mix = effect_mix.overlay(effect, position=effect_position)
                
                # Overlay effects onto voice with careful volume control
                voice_audio = voice_audio.overlay(effect_mix)
                # Apply additional compression to prevent peaks
                voice_audio = self._compress_audio(voice_audio)
            
            # Handle highlight background music
            if "highlight" in segment["text"].lower() and include_background:
                highlight_music = self.background_music["highlight"]
                highlight_music = highlight_music - abs(self.MUSIC_VOLUME)
                highlight_music = self._fade_effect(highlight_music)
                voice_audio = voice_audio.overlay(highlight_music[:len(voice_audio)])
            
            # Add pause between different speakers
            current_speaker = segment.get("speaker", "")
            if previous_speaker and previous_speaker != current_speaker:
                final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
            
            # Smooth out transitions
            voice_audio = voice_audio.fade_in(150).fade_out(300)
            
            # Mix with appropriate crossfade
            if i == 0:
                final_mix = final_mix.append(voice_audio, crossfade=self.INTRO_FADE)
            else:
                final_mix = final_mix.append(voice_audio, crossfade=self.CROSSFADE_LENGTH)
            
            previous_speaker = current_speaker
        
        # Add outro
        outro = self.background_music["outro"].fade_in(self.INTRO_FADE)
        final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
        final_mix = final_mix.append(outro, crossfade=self.INTRO_FADE)
        
        # Final compression pass on the complete mix
        final_mix = self._compress_audio(final_mix)
        
        return self.to_bytes(final_mix)

    def _add_pause(self, audio: AudioSegment, duration: int) -> AudioSegment:
        logging.info("adding pause")
        return audio + AudioSegment.silent(duration=duration)

    def _detect_event_triggers(self, text: str) -> List[str]:
        triggers = []
        events = {
            "crowd_cheer": ["home run", "scores", "wins", "victory", "walk off"],
            "bat_hit": ["hits", "singles", "doubles", "triples", "batting"],
            "crowd_tension": ["full count", "bases loaded", "bottom of the ninth"],
        }
        
        for effect, keywords in events.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                triggers.append(effect)
                
        return triggers

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("normalizing audio")
        return normalize(audio)

    def to_bytes(self, mixed_audio: AudioSegment) -> bytes:
        logging.info("converting audio to bytes")
        """Converts the mixed AudioSegment to bytes."""
        
        # Export the AudioSegment to a byte array in mp3 format
        buffer = io.BytesIO()
        mixed_audio.export(buffer, format="mp3")
         # Get the bytes from the buffer
        audio_bytes = buffer.getvalue()
        
        # Close the buffer
        buffer.close()
        
        return audio_bytes