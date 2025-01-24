from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from typing import List, Dict, Optional
import io
import logging
from google.cloud import logging as cloud_logging

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()


class JapaneseMLBAudioMixer:
    def __init__(self):
        self.sound_effects = {
            "crowd_cheer": AudioSegment.from_mp3("assets/sounds/crowd_cheer.mp3"),
            "bat_hit": AudioSegment.from_mp3("assets/sounds/bat_hit.mp3"),
            "crowd_tension": AudioSegment.from_mp3("assets/sounds/crowd_tension.mp3"),
            "walkup_music": AudioSegment.from_mp3("assets/sounds/walkup_music.mp3"),
            "stadium_ambience": AudioSegment.from_mp3("assets/sounds/stadium_ambience.mp3")
        }
        
        self.background_music = {
            "intro": AudioSegment.from_mp3("assets/music/sakura.mp3"),
            "highlight": AudioSegment.from_mp3("assets/music/highlight.mp3"),
            "outro": AudioSegment.from_mp3("assets/music/sakura.mp3")
        }
        
        # Refined volume levels
        self.VOICE_VOLUME = 0
        self.MUSIC_VOLUME = -25
        self.SFX_VOLUME = -18
        self.AMBIENCE_VOLUME = -30
        
        # Timing constants (in milliseconds)
        self.SPEAKER_PAUSE = 850
        self.CROSSFADE_LENGTH = 400
        self.INTRO_FADE = 2000
        self.EFFECT_FADE = 600

    def _compress_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("Compressing audio...")
        return compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=10, release=100)

    def _fade_effect(self, effect: AudioSegment) -> AudioSegment:
        logging.info("fade effect")
        return effect.fade_in(200).fade_out(self.EFFECT_FADE)

    def _process_voice_segment(self, voice_audio: AudioSegment) -> AudioSegment:
        logging.info("Processing voice segment")
        voice_audio = self._normalize_audio(voice_audio)
        voice_audio = self._compress_audio(voice_audio)
        return voice_audio - abs(self.VOICE_VOLUME)

    def mix_podcast_audio(self, voice_segments: List[Dict[str, bytes]], 
                         include_background: bool = True) -> AudioSegment:
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
            voice_audio = self._process_voice_segment(voice_audio)
            
            triggers = self._detect_japanese_event_triggers(segment["text"])
            if triggers:
                effect_mix = AudioSegment.silent(duration=len(voice_audio))
                
                for trigger in triggers:
                    effect = self.sound_effects[trigger]
                    effect = self._fade_effect(effect)
                    effect = effect - abs(self.SFX_VOLUME)
                    
                    # Position effect for ホームラン (home run)
                    if trigger == "crowd_cheer" and "ホームラン" in segment["text"]:
                        voice_audio = AudioSegment.silent(duration=200) + voice_audio
                        effect_position = 0
                    else:
                        effect_position = 100
                    
                    effect_mix = effect_mix.overlay(effect, position=effect_position)
                
                voice_audio = voice_audio.overlay(effect_mix)
                voice_audio = self._compress_audio(voice_audio)
            
            # Handle ハイライト (highlight) background music
            if "ハイライト" in segment["text"] and include_background:
                highlight_music = self.background_music["highlight"]
                highlight_music = highlight_music - abs(self.MUSIC_VOLUME)
                highlight_music = self._fade_effect(highlight_music)
                voice_audio = voice_audio.overlay(highlight_music[:len(voice_audio)])
            
            current_speaker = segment.get("speaker", "")
            if previous_speaker and previous_speaker != current_speaker:
                final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
            
            voice_audio = voice_audio.fade_in(150).fade_out(300)
            
            if i == 0:
                final_mix = final_mix.append(voice_audio, crossfade=self.INTRO_FADE)
            else:
                final_mix = final_mix.append(voice_audio, crossfade=self.CROSSFADE_LENGTH)
            
            previous_speaker = current_speaker
        
        outro = self.background_music["outro"].fade_in(self.INTRO_FADE)
        final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
        final_mix = final_mix.append(outro, crossfade=self.INTRO_FADE)
        final_mix = self._compress_audio(final_mix)
        
        return self.to_bytes(final_mix)

    def _add_pause(self, audio: AudioSegment, duration: int) -> AudioSegment:
        logging.info("adding pause")
        return audio + AudioSegment.silent(duration=duration)

    def _detect_japanese_event_triggers(self, text: str) -> List[str]:
        logging.info("detecting japanese event triggers")
        triggers = []
        events = {
            "crowd_cheer": ["ホームラン", "得点", "勝利", "サヨナラ", "逆転"],
            "bat_hit": ["ヒット", "シングル", "ダブル", "トリプル", "打撃"],
            "crowd_tension": ["フルカウント", "満塁", "九回裏"],
        }
        
        for effect, keywords in events.items():
            if any(keyword in text for keyword in keywords):
                triggers.append(effect)
                
        return triggers

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        logging("normalizing audio")
        return normalize(audio)


    def to_bytes(self, mixed_audio: AudioSegment) -> bytes:
        logging.info("Exporting AudioSegment to bytes")
        """Converts the mixed AudioSegment to bytes."""
        
        # Export the AudioSegment to a byte array in mp3 format
        buffer = io.BytesIO()
        mixed_audio.export(buffer, format="mp3")
         # Get the bytes from the buffer
        audio_bytes = buffer.getvalue()
        
        # Close the buffer
        buffer.close()
        
        return audio_bytes