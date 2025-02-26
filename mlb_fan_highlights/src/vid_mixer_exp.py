import os
from google.cloud import secretmanager_v1
from audio_mixer import MLBAudioMixer
from vid_mixer import MLBVideoPodcastGenerator

def main():
    # Initialize the audio mixer
    project_id = "gem-rush-007"
    secret_name = "cloud-run-invoker"
    audio_mixer = MLBAudioMixer(project_id, secret_name)
    
    # Initialize the video podcast generator
    video_generator = MLBVideoPodcastGenerator(audio_mixer)
    
    # Example voice segments (in a real scenario, these would come from your podcast audio)
    # For demonstration, we'll use the example from the audio mixer
    voice_segments = [
        {
            "audio": b"...",  # Your actual audio bytes would go here
            "text": "Welcome to today's MLB game recap between the Yankees and the Red Sox.",
            "speaker": "Host"
        },
        {
            "audio": b"...",  # Your actual audio bytes would go here
            "text": "The Yankees took an early lead with a two-run homer from Judge in the first inning.",
            "speaker": "Host"
        },
        {
            "audio": b"...",  # Your actual audio bytes would go here
            "text": "What really impressed me was Judge's batting average this season, hitting an incredible .297.",
            "speaker": "Analyst"
        }
    ]
    
    # Example stat segments to include in the video
    stat_segments = [
        {
            "type": "title",
            "content": {
                "text": "Yankees vs Red Sox",
                "subtitle": "Game Recap"
            },
            "duration": 4.0
        },
        {
            "type": "stat",
            "content": {
                "name": "OPS",
                "data": {
                    "Judge": 0.988,
                    "Devers": 0.876,
                    "Stanton": 0.822,
                    "Bogaerts": 0.833
                },
                "teams": ["NYY", "BOS", "NYY", "BOS"]
            },
            "duration": 5.0
        },
        {
            "type": "player",
            "content": {
                "name": "Aaron Judge",
                "team": "NYY",
                "stats": {
                    "HR": 2,
                    "RBI": 5,
                    "AVG": ".342"
                }
            },
            "duration": 4.0
        },
        {
            "type": "stat",
            "content": {
                "name": "WHIP",
                "data": {
                    "Cole": 0.92,
                    "Eovaldi": 1.23,
                    "Taillon": 1.15,
                    "Wacha": 1.35
                },
                "teams": ["NYY", "BOS", "NYY", "BOS"]
            },
            "duration": 5.0
        },
        {
            "type": "stat",
            "content": {
                "name": "SO",
                "data": {
                    "Cole": 248,
                    "Eovaldi": 176,
                    "Taillon": 151,
                    "Wacha": 104
                },
                "teams": ["NYY", "BOS", "NYY", "BOS"]
            },
            "duration": 4.5
        }
    ]
    
    # Generate the video podcast
    output_file = "yankees_redsox_recap.mp4"
    video_generator.create_video_podcast(
        voice_segments=voice_segments,
        stat_segments=stat_segments,
        output_path=output_file,
        title="Yankees vs Red Sox Game Recap",
        include_background=True
    )
    
    print(f"Video podcast created: {output_file}")
    
    # Alternatively, create a sample video for testing
    sample_file = "sample_video.mp4"
    video_generator.create_sample_video(sample_file)
    print(f"Sample video created: {sample_file}")

if __name__ == "__main__":
    main()