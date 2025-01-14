from google.cloud import texttospeech
import json
import os

def create_audio_for_speaker(text, speaker_config):
    """Creates audio for a single piece of dialogue."""
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-ES",  # Spanish
        name=speaker_config["voice"],
        ssml_gender=texttospeech.SsmlVoiceGender[speaker_config["gender"]]
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaker_config["speed"]
    )
    
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    
    return response.audio_content

def create_podcast(script_data, output_filename):
    """Process the entire podcast script with multiple speakers."""
    
    # Speaker configurations
    speaker_configs = {
        "Narrador de jugada por jugada": {
            "voice": "es-ES-Neural2-B",  # Male voice for play-by-play
            "gender": "MALE",
            "speed": 1.1  # Slightly faster for exciting moments
        },
        "Comentarista de color": {
            "voice": "es-ES-Neural2-C",  # Different male voice for color commentary
            "gender": "FEMALE",
            "speed": 1.0
        },
        "Citas de Jugadores": {
            "voice": "es-ES-Neural2-D",  # Different voice for player quotes
            "gender": "FEMALE",
            "speed": 0.95  # Slightly slower for quotes
        }
    }
    
    combined_audio = b""
    
    for i, segment in enumerate(script_data):
        speaker = segment['speaker']
        text = segment['text'].strip()
        
        if speaker in speaker_configs and text:
            print(f"Processing segment {i}: {speaker}")
            
            # Generate audio for the segment
            audio_content = create_audio_for_speaker(text, speaker_configs[speaker])
            combined_audio += audio_content
            
            # Add a short pause between segments (0.5 second of silence)
            pause = create_audio_for_speaker(" ", speaker_configs[speaker])
            combined_audio += pause
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # Write the final combined podcast
    with open(output_filename, "wb") as out:
        out.write(combined_audio)
        print(f'Full podcast written to file "{output_filename}"')
    
    return output_filename

def list_available_voices():
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices(language_code="es-ES")
    for voice in voices.voices:
        print(f"Name: {voice.name}")
        print(f"Gender: {voice.ssml_gender}")
        print(f"Language codes: {voice.language_codes}\n")

def main():
    # Load the script data
    script_data = [
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Bienvenidos, fanáticos del béisbol, a nuestro podcast de los Dodgers. Hoy, vamos a revivir el último partido de los Dodgers, un encuentro lleno de acción y momentos emocionantes.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Así es, un partido que mantuvo a los aficionados al borde de sus asientos. Analicemos los momentos clave y las jugadas que definieron el juego.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'El juego comenzó con los Dodgers al bate en la primera entrada. Mookie Betts se paró en el plato, buscando iniciar el juego con fuerza.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Betts es conocido por su habilidad para encender la chispa ofensiva. Veamos qué puede hacer hoy.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': '¡Y ahí está! Betts conecta un sencillo al jardín central. Un gran comienzo para los Dodgers.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Un contacto sólido, justo lo que los Dodgers necesitaban para empezar. Ahora, veremos cómo se desarrolla la entrada.'}, 
        {'speaker': 'Citas de Jugadores', 'text': 'Solo estaba tratando de hacer contacto y llegar a base para mis compañeros.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Pasamos a la tercera entrada, los Dodgers están abajo por una carrera. Freddie Freeman se prepara para batear.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Freeman es un bateador de poder, capaz de cambiar el rumbo del juego con un solo swing.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': '¡Y la manda lejos! ¡Un jonrón de Freeman! Los Dodgers toman la delantera.'}, 
        {'speaker': 'Comentarista de color', 'text': '¡Qué batazo! Freeman conectó esa pelota con toda su fuerza. Un jonrón que le da la ventaja a los Dodgers.'}, 
        {'speaker': 'Citas de Jugadores', 'text': 'Sabía que tenía que hacer algo para ayudar al equipo, y afortunadamente pude conectar esa pelota.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'En la sexta entrada, los Dodgers están defendiendo su ventaja. El lanzador estrella, Clayton Kershaw, está en el montículo.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Kershaw es un veterano, conocido por su control y habilidad para dominar a los bateadores.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Kershaw lanza un strikeout, ponchando al bateador rival. Una gran actuación del lanzador.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Un lanzamiento impecable, Kershaw sigue demostrando por qué es uno de los mejores lanzadores del juego.'}, 
        {'speaker': 'Citas de Jugadores', 'text': 'Solo estaba tratando de ejecutar mis lanzamientos y mantener a los bateadores fuera de base.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Avanzamos a la octava entrada, el juego está apretado. Los Dodgers necesitan asegurar la victoria.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Cada jugada cuenta en este momento del juego. La tensión es palpable.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': '¡Un doble de Will Smith! Los Dodgers aumentan su ventaja. Una jugada clave en la octava entrada.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Smith conectó un batazo fuerte, justo lo que los Dodgers necesitaban para ampliar su ventaja.'}, 
        {'speaker': 'Citas de Jugadores', 'text': 'Solo estaba tratando de hacer un buen contacto y ayudar al equipo a anotar más carreras.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Llegamos a la novena entrada, los Dodgers están a tres outs de la victoria. El cerrador entra al juego.'}, 
        {'speaker': 'Comentarista de color', 'text': 'El cerrador tiene la tarea de asegurar la victoria. La presión está sobre sus hombros.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': '¡Y ahí está! El último out. ¡Los Dodgers ganan el juego! Una gran victoria para el equipo.'}, 
        {'speaker': 'Comentarista de color', 'text': 'Un juego emocionante de principio a fin. Los Dodgers demostraron su capacidad para superar los desafíos y asegurar la victoria.'}, 
        {'speaker': 'Narrador de jugada por jugada', 'text': 'Eso es todo por hoy en nuestro podcast de los Dodgers. Gracias por acompañarnos y nos vemos en el próximo juego.'}
    ]
    
    output_filename = "podcast_output/full_podcast.mp3"
    podcast_file = create_podcast(script_data, output_filename)
    print(f"Created podcast: {podcast_file}")
    #list_available_voices()
    
    # Note: You would need additional code to combine these segments
    # into a single audio file, which could be done using a library like pydub

if __name__ == "__main__":
    main()