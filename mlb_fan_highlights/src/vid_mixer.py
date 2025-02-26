from pydub import AudioSegment
from moviepy import (
    ImageClip, AudioFileClip, concatenate_videoclips, 
    CompositeVideoClip, TextClip, ColorClip
)
from typing import List, Dict, Optional, Tuple
import io
import os
import tempfile
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

class MLBVideoPodcastGenerator:
    def __init__(self, audio_mixer):
        """
        Initialize the MLB Video Podcast Generator.
        
        Args:
            audio_mixer: An instance of MLBAudioMixer to handle audio processing
        """
        self.audio_mixer = audio_mixer
        self.storage_client = audio_mixer.storage_client
        self.bucket_name = "mlb-audio-assets"
        self.image_bucket_name = "mlb-image-assets"  # Could be the same bucket or a different one
        
        # Video settings
        self.resolution = (1280, 720)  # 720p
        self.fps = 24
        self.transition_duration = 1.0  # seconds
        
        # Stats visualization settings
        self.stat_chart_duration = 5.0  # seconds to show each stat chart
        self.team_colors = {
            "NYY": {"primary": "#0C2340", "secondary": "#C4CED4"},  # Yankees
            "BOS": {"primary": "#BD3039", "secondary": "#0C2340"},  # Red Sox
            "LAD": {"primary": "#005A9C", "secondary": "#A5ACAF"},  # Dodgers
            # Add more team colors as needed
        }
        
        # Default team color if not found
        self.default_team_color = {"primary": "#1E88E5", "secondary": "#FFFFFF"}
        
        # Load or create background templates
        self.backgrounds = self._load_backgrounds()

    def _load_backgrounds(self) -> Dict[str, ImageClip]:
        """Load background templates from storage or create them dynamically"""
        backgrounds = {}
        
        try:
            # Try to load from GCS
            bucket = self.storage_client.bucket(self.image_bucket_name)
            background_paths = {
                "default": "templates/default_bg.png",
                "stats": "templates/stats_bg.png",
                "player": "templates/player_bg.png",
                "highlight": "templates/highlight_bg.png"
            }
            
            for name, path in background_paths.items():
                try:
                    blob = bucket.blob(path)
                    img_bytes = blob.download_as_bytes()
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                        temp.write(img_bytes)
                        temp_path = temp.name
                    backgrounds[name] = ImageClip(temp_path)
                    # Clean up temp file after loading
                    os.remove(temp_path)
                except Exception as e:
                    logging.warning(f"Could not load background {name}: {e}")
                    # Create a default background
                    backgrounds[name] = self._create_default_background(name)
        except Exception as e:
            logging.error(f"Error loading backgrounds: {e}")
            # Create default backgrounds
            for name in ["default", "stats", "player", "highlight"]:
                backgrounds[name] = self._create_default_background(name)
                
        return backgrounds

    def _create_default_background(self, bg_type: str) -> ImageClip:
        """Create a default background if none is available from storage"""
        width, height = self.resolution
        
        # Create a PIL Image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if bg_type == "stats":
            # Stats background - blue gradient with overlaid grid
            for y in range(height):
                # Create gradient from dark blue to lighter blue
                color_val = int(40 + (y / height) * 40)
                draw.line([(0, y), (width, y)], fill=(15, 25, color_val, 255))
            
            # Add grid lines
            grid_color = (255, 255, 255, 30)  # Transparent white
            for x in range(0, width, 50):
                draw.line([(x, 0), (x, height)], fill=grid_color)
            for y in range(0, height, 50):
                draw.line([(0, y), (width, y)], fill=grid_color)
                
        elif bg_type == "player":
            # Player background - dark with spotlight effect
            for y in range(height):
                for x in range(width):
                    # Create radial gradient
                    dx, dy = x - width//2, y - height//2
                    distance = (dx**2 + dy**2)**0.5
                    max_distance = ((width//2)**2 + (height//2)**2)**0.5
                    ratio = distance / max_distance
                    color_val = max(10, int(40 * (1 - ratio)))
                    img.putpixel((x, y), (color_val, color_val, color_val + 10, 255))
        
        elif bg_type == "highlight":
            # Highlight background - dynamic with light streaks
            for y in range(height):
                for x in range(width):
                    # Base dark color
                    color = (20, 20, 30, 255)
                    
                    # Add light streaks
                    if (x + y) % 100 < 5:
                        streak_intensity = 100 - ((x + y) % 100) * 20
                        color = (min(255, color[0] + streak_intensity), 
                                min(255, color[1] + streak_intensity), 
                                min(255, color[2] + streak_intensity), 255)
                    
                    img.putpixel((x, y), color)
        
        else:  # default background
            # Simple dark gradient
            for y in range(height):
                # Create gradient from dark to a bit lighter
                color_val = int(20 + (y / height) * 30)
                draw.line([(0, y), (width, y)], fill=(color_val, color_val, color_val + 5, 255))
        
        # Add MLB logo watermark
        logo_size = (200, 100)
        logo_pos = (width - logo_size[0] - 20, height - logo_size[1] - 20)
        draw.rectangle([logo_pos, (logo_pos[0] + logo_size[0], logo_pos[1] + logo_size[1])], 
                      fill=(255, 255, 255, 50), outline=(255, 255, 255, 100))
        draw.text((logo_pos[0] + 50, logo_pos[1] + 40), "MLB", fill=(255, 255, 255, 150))
        
        # Convert PIL Image to numpy array for MoviePy
        img_array = np.array(img)
        
        # Create ImageClip
        return ImageClip(img_array)

    def _generate_stat_visualization(self, 
                                    stat_name: str, 
                                    stat_data: Dict[str, float],
                                    teams: List[str] = None) -> np.ndarray:
        """
        Generate visualization for a specific stat.
        
        Args:
            stat_name: Name of the statistic (e.g., "WHIP", "OPS")
            stat_data: Dictionary mapping player/team names to stat values
            teams: Optional list of team abbreviations for coloring
            
        Returns:
            numpy.ndarray: Image as a numpy array
        """
        plt.figure(figsize=(12, 7))
        
        if stat_name.lower() in ["ops", "avg", "obp", "slg", "whip", "era"]:
            # Bar chart for these stats
            names = list(stat_data.keys())
            values = list(stat_data.values())
            
            # Determine colors based on teams if provided
            colors = []
            if teams and len(teams) == len(names):
                for team in teams:
                    team_color = self.team_colors.get(team, self.default_team_color)
                    colors.append(team_color["primary"])
            else:
                # Default color scheme
                colors = ['#1E88E5', '#FFC107', '#D81B60', '#8BC34A', '#5E35B1']
                colors = colors * (len(names) // len(colors) + 1)
                colors = colors[:len(names)]
            
            plt.bar(names, values, color=colors)
            plt.title(f"{stat_name.upper()} Comparison", fontsize=24, pad=20)
            plt.ylabel(stat_name.upper(), fontsize=18)
            
            # Set y-axis limits based on the stat type
            if stat_name.lower() == "ops":
                plt.ylim(0, max(1.1, max(values) * 1.1))
            elif stat_name.lower() == "era":
                plt.ylim(0, max(10, max(values) * 1.2))
            elif stat_name.lower() == "whip":
                plt.ylim(0, max(2, max(values) * 1.2))
            
            # Add value labels on top of each bar
            for i, v in enumerate(values):
                if stat_name.lower() in ["ops", "avg", "obp", "slg"]:
                    # Format as 3 decimal places for these stats
                    plt.text(i, v + 0.02, f"{v:.3f}", 
                            ha='center', fontsize=16, fontweight='bold')
                else:
                    # Format as 2 decimal places for other stats
                    plt.text(i, v + 0.02, f"{v:.2f}", 
                            ha='center', fontsize=16, fontweight='bold')
            
        elif stat_name.lower() in ["hr", "rbi", "so", "bb", "runs", "hits"]:
            # For counting stats, use horizontal bar chart
            names = list(stat_data.keys())
            values = list(stat_data.values())
            
            # Sort by value for better visualization
            sorted_indices = np.argsort(values)
            sorted_names = [names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Determine colors
            colors = ['#1E88E5'] * len(sorted_names)
            if teams and len(teams) == len(names):
                sorted_teams = [teams[i] for i in sorted_indices]
                colors = [self.team_colors.get(team, self.default_team_color)["primary"] 
                         for team in sorted_teams]
            
            plt.barh(sorted_names, sorted_values, color=colors)
            plt.title(f"{stat_name.upper()} Leaders", fontsize=24, pad=20)
            plt.xlabel(stat_name.upper(), fontsize=18)
            
            # Add value labels
            for i, v in enumerate(sorted_values):
                plt.text(v + 0.5, i, str(int(v)), 
                        va='center', fontsize=16, fontweight='bold')
                
        # Style the chart with a transparent background
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(pad=3.0)
        
        # Convert matplotlib figure to numpy array
        fig = plt.gcf()
        fig.patch.set_alpha(0.0)
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return image_array

    def _create_stat_clip(self, 
                         stat_name: str, 
                         stat_data: Dict[str, float],
                         teams: List[str] = None,
                         duration: float = None) -> ImageClip:
        """
        Create a video clip for a specific statistic.
        
        Args:
            stat_name: Name of the statistic
            stat_data: Dictionary mapping player/team names to stat values
            teams: Optional list of team abbreviations for coloring
            duration: Duration to show this stat (defaults to self.stat_chart_duration)
            
        Returns:
            ImageClip: A video clip showing the statistic
        """
        if duration is None:
            duration = self.stat_chart_duration
            
        # Generate the stat visualization
        stat_img = self._generate_stat_visualization(stat_name, stat_data, teams)
        
        # Create the clip
        stat_clip = ImageClip(stat_img).set_duration(duration)
        
        # Apply background
        bg = self.backgrounds["stats"].copy().set_duration(duration)
        
        # Composite the stat over background
        return CompositeVideoClip([bg, stat_clip.set_position("center")])

    def _create_overlay_text(self, 
                           text: str, 
                           position: Tuple[int, int] = ("center", "bottom"),
                           duration: float = 3.0,
                           fontsize: int = 40,
                           color: str = "white") -> TextClip:
        """Create text overlay for video clips"""
        return (TextClip(text, fontsize=fontsize, color=color, font="Arial-Bold")
                .set_position(position)
                .set_duration(duration))

    def create_video_podcast(self, 
                           voice_segments: List[Dict[str, bytes]],
                           stat_segments: List[Dict],
                           output_path: str,
                           title: str = "MLB Game Recap",
                           include_background: bool = True) -> str:
        """
        Create a complete video podcast with audio and visual elements.
        
        Args:
            voice_segments: List of dictionaries with audio bytes and text
            stat_segments: List of dictionaries with stat data
                Each dict should contain:
                    - 'type': type of segment ('stat', 'player', 'title', etc.)
                    - 'content': content data specific to the type
                    - 'duration': optional duration to show (seconds)
            output_path: Path to save the final video
            title: Title of the podcast
            include_background: Whether to include background music/ambience
            
        Returns:
            str: Path to the created video file
        """
        # First, create the audio mix
        audio_bytes = self.audio_mixer.mix_podcast_audio(voice_segments, include_background)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            audio_path = temp_audio.name
        
        # Load the audio file with MoviePy
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        # Create intro clip
        intro_bg = self.backgrounds["default"].copy().set_duration(3.0)
        title_text = TextClip(title, fontsize=60, color='white', font='Arial-Bold')
        title_text = title_text.set_position('center').set_duration(3.0)
        intro_clip = CompositeVideoClip([intro_bg, title_text])
        
        video_clips = [intro_clip]
        current_time = 3.0  # Start after intro
        
        # Process each stat segment
        for segment in stat_segments:
            segment_type = segment.get('type', 'stat')
            content = segment.get('content', {})
            duration = segment.get('duration', self.stat_chart_duration)
            
            if segment_type == 'stat':
                # Create stat visualization
                stat_name = content.get('name', 'Stat')
                stat_data = content.get('data', {})
                teams = content.get('teams', None)
                
                stat_clip = self._create_stat_clip(stat_name, stat_data, teams, duration)
                video_clips.append(stat_clip)
                current_time += duration
                
            elif segment_type == 'player':
                # Player highlight
                player_name = content.get('name', '')
                player_stats = content.get('stats', {})
                player_team = content.get('team', '')
                
                # Try to load player image
                player_img = None
                try:
                    if player_name:
                        bucket = self.storage_client.bucket(self.image_bucket_name)
                        # Normalize player name for file path (lowercase, spaces to underscores)
                        player_file = f"players/{player_name.lower().replace(' ', '_')}.png"
                        blob = bucket.blob(player_file)
                        
                        if blob.exists():
                            img_bytes = blob.download_as_bytes()
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                                temp.write(img_bytes)
                                player_img = ImageClip(temp.name)
                                os.remove(temp.name)
                except Exception as e:
                    logging.warning(f"Could not load player image for {player_name}: {e}")
                
                # Create player highlight clip
                bg = self.backgrounds["player"].copy().set_duration(duration)
                player_clip = CompositeVideoClip([bg])
                
                # Add player image if available
                if player_img:
                    # Resize maintaining aspect ratio
                    img_ar = player_img.w / player_img.h
                    target_height = min(self.resolution[1] * 0.7, 500)
                    target_width = target_height * img_ar
                    
                    player_img = (player_img.resize(height=int(target_height))
                                 .set_position(("center", "center"))
                                 .set_duration(duration))
                    
                    player_clip = CompositeVideoClip([player_clip, player_img])
                
                # Add player name and stats
                name_text = self._create_overlay_text(
                    player_name, 
                    position=("center", 50), 
                    duration=duration,
                    fontsize=50
                )
                
                player_clip = CompositeVideoClip([player_clip, name_text])
                
                # Add stats text
                if player_stats:
                    stats_str = " | ".join([f"{k.upper()}: {v}" for k, v in player_stats.items()])
                    stats_text = self._create_overlay_text(
                        stats_str,
                        position=("center", self.resolution[1] - 100),
                        duration=duration,
                        fontsize=36
                    )
                    player_clip = CompositeVideoClip([player_clip, stats_text])
                
                # Add team name if available
                if player_team:
                    team_text = self._create_overlay_text(
                        player_team,
                        position=("center", 120),
                        duration=duration,
                        fontsize=40,
                        color=self.team_colors.get(player_team, self.default_team_color)["secondary"]
                    )
                    player_clip = CompositeVideoClip([player_clip, team_text])
                
                video_clips.append(player_clip)
                current_time += duration
                
            elif segment_type == 'title':
                # Section title slide
                title_text = content.get('text', '')
                subtitle = content.get('subtitle', '')
                
                bg = self.backgrounds["highlight"].copy().set_duration(duration)
                title_clip = CompositeVideoClip([bg])
                
                # Add title text
                if title_text:
                    main_text = self._create_overlay_text(
                        title_text,
                        position=("center", "center"),
                        duration=duration,
                        fontsize=60
                    )
                    title_clip = CompositeVideoClip([title_clip, main_text])
                
                # Add subtitle if available
                if subtitle:
                    sub_text = self._create_overlay_text(
                        subtitle,
                        position=("center", self.resolution[1]//2 + 80),
                        duration=duration,
                        fontsize=40
                    )
                    title_clip = CompositeVideoClip([title_clip, sub_text])
                
                video_clips.append(title_clip)
                current_time += duration
        
        # Add a final clip if needed to match audio duration
        remaining_duration = max(0, total_duration - current_time)
        if remaining_duration > 0:
            outro_bg = self.backgrounds["default"].copy().set_duration(remaining_duration)
            outro_text = TextClip("Thanks for watching!", fontsize=50, color='white', font='Arial-Bold')
            outro_text = outro_text.set_position('center').set_duration(remaining_duration)
            outro_clip = CompositeVideoClip([outro_bg, outro_text])
            video_clips.append(outro_clip)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips)
        
        # Set the audio
        final_video = final_video.set_audio(audio)
        
        # Write the final video file
        final_video.write_videofile(
            output_path, 
            fps=self.fps, 
            codec='libx264', 
            audio_codec='aac',
            threads=4
        )
        
        # Clean up temporary files
        os.remove(audio_path)
        
        return output_path

    def create_sample_video(self, output_path: str) -> str:
        """
        Create a sample video with dummy data for testing.
        
        Args:
            output_path: Path to save the sample video
            
        Returns:
            str: Path to the created video file
        """
        # Create sample voice segments with dummy audio
        voice_segments = []
        
        # Generate a silent audio segment as placeholder
        silent_audio = AudioSegment.silent(duration=3000)  # 3 seconds
        buffer = io.BytesIO()
        silent_audio.export(buffer, format="mp3")
        audio_bytes = buffer.getvalue()
        
        # Sample voice segments
        voice_segments = [
            {
                "audio": audio_bytes,
                "text": "Welcome to today's MLB game recap between the Yankees and the Red Sox.",
                "speaker": "Host"
            },
            {
                "audio": audio_bytes,
                "text": "The Yankees pulled off an amazing victory with a walk off home run in the 9th inning.",
                "speaker": "Host"
            },
            {
                "audio": audio_bytes,
                "text": "Let's look at some key stats from today's matchup.",
                "speaker": "Analyst"
            },
            {
                "audio": audio_bytes,
                "text": "Aaron Judge had an incredible day with 2 home runs and 5 RBIs.",
                "speaker": "Analyst"
            }
        ]
        
        # Sample stat segments
        stat_segments = [
            {
                "type": "title",
                "content": {
                    "text": "Yankees vs Red Sox",
                    "subtitle": "Game Recap"
                },
                "duration": 3.0
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
                "duration": 4.0
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
                "duration": 3.5
            },
            {
                "type": "stat",
                "content": {
                    "name": "HR",
                    "data": {
                        "Judge": 42,
                        "Stanton": 31,
                        "Devers": 28,
                        "Rizzo": 22
                    },
                    "teams": ["NYY", "NYY", "BOS", "NYY"]
                },
                "duration": 4.0
            }
        ]
        
        return self.create_video_podcast(
            voice_segments=voice_segments,
            stat_segments=stat_segments,
            output_path=output_path,
            title="Yankees vs Red Sox Recap"
        )