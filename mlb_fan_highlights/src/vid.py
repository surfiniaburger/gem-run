# vid.py - Complete Video Generation Module
from google.cloud.video import transcoder_v1
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient
from vertexai.preview.vision_models import ImageGenerationModel
from google.genai.types import Tool, GoogleSearch, SafetySetting, GenerateContentConfig
import json
import uuid
import time
import logging
from typing import List, Dict, Any
from google import genai
from google.cloud import logging as cloud_logging
import enum

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

class CloudVideoGenerator:
    def __init__(self, gcs_handler):
        self.gcs_handler = gcs_handler
        self.client = TranscoderServiceClient()
        self.parent = f"projects/{gcs_handler.project_id}/locations/us-central1"
        
        # Initialize AI models with latest configurations
        logging.info("Initializing Vertex AI client and Imagen model")
        self.genai_client = genai.Client(vertexai=True, project="gem-rush-007", location="us-east4")
        self.imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
        
        # Configure safety settings
        self.safety_config = [
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
        ]
        logging.info("CloudVideoGenerator initialized successfully.")

    def _analyze_script(self, script_data: list) -> Dict[str, Any]:
        """Analyze podcast script using Gemini 2.0 with enhanced configurations."""
        logging.info("Starting script analysis.")
        full_text = " ".join([segment['text'] for segment in script_data])
        logging.debug(f"Full script text: {full_text[:100]}...")  # Log first 100 characters

        analysis_prompt = """Analyze this baseball podcast script and generate:
        {
            "key_moments": [{
                "timestamp": "HH:MM:SS",
                "description": "text",
                "visual_prompt": "Imagen prompt",
                "duration": 5,
                "transition": "fade/cut/zoom"
            }],
            "theme": "modern/retro/dramatic",
            "color_palette": {
                "primary": "#hex",
                "secondary": "#hex",
                "accent": "#hex"
            },
            "graphics_style": "dynamic/animated/static",
            "audio_intensity": 0-100
        }"""
        try:
            logging.info("Sending analysis request to Gemini 2.0 model.")
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"role": "user", "parts": [{"text": analysis_prompt + full_text}]}],
                config=GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                    tools=[Tool(google_search=GoogleSearch())],
                    safety_settings=self.safety_config
                ),
            )
            logging.info("Received response from Gemini 2.0.")
            parsed_response = self._parse_gemini_response(response.text)
            logging.info("Script analysis completed successfully.")
            return parsed_response
        
        except Exception as e:
            logging.error(f"Script analysis failed: {str(e)}")
            raise

    def _parse_gemini_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse and validate Gemini response with error handling."""
        logging.info("Parsing Gemini response.")
        try:
            clean_text = raw_response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean_text)
            logging.info("Successfully parsed Gemini response.")
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            raise ValueError("Failed to parse AI response")
        except Exception as e:
            logging.error(f"Unexpected parsing error: {str(e)}")
            raise

    def _generate_images(self, analysis: Dict[str, Any]) -> List[bytes]:
        logging.info("Starting image generation for key moments.")
        images = []
        for idx, moment in enumerate(analysis['key_moments']):
            logging.info(f"Generating image for key moment {idx + 1}: {moment.get('description', 'No description')}")
            try:
                # Add prompt engineering/enhancement
                enhanced_prompt = self._enhance_prompt(moment['visual_prompt'])
                logging.debug(f"Enhanced prompt: {enhanced_prompt}")
                
                # Add better retry strategy
                response = self._retry_with_backoff(
                    lambda: self.imagen_model.generate_images(
                        prompt=enhanced_prompt,
                        aspect_ratio="16:9",
                    )
                )
                
                if not response.images:
                    raise ValueError("Empty image response")
                    
                logging.info(f"Image generation successful for moment {idx + 1}.")
                images.append(response.images[0]._image_bytes)
                
            except Exception as e:
                logging.warning(f"Image generation failed for moment {idx + 1}: {str(e)}. Using default image.")
                default_image = self._create_default_image()
                images.append(default_image)
        logging.info("Completed image generation for all key moments.")
        return images

    def _retry_with_backoff(self, operation, max_retries=3, initial_delay=1):
        """Execute operation with exponential backoff retry logic.
    
        Args:
            operation: Callable to execute
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
        """
        logging.info("Starting retry with exponential backoff.")
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                result = operation()
                logging.info(f"Operation succeeded on attempt {attempt + 1}.")
                return result
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Waiting for {delay} seconds before retrying.")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        logging.error("Maximum retry attempts reached. Operation failed.")
        raise last_exception

    def _create_default_image(self) -> bytes:
        """Creates a white default image with proper RGB format"""
        logging.info("Creating a default image as fallback.")
        from PIL import Image
        from io import BytesIO

        # Create RGB image instead of RGBA to avoid alpha channel issues
        image = Image.new('RGB', (1920, 1080), (255, 255, 255))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        logging.info("Default image created successfully.")
        return img_byte_arr.getvalue()

    def _create_job_config(self, image_uris: List[str], audio_uri: str) -> transcoder_v1.types.JobConfig:
        """Create a proper job configuration following API specs"""
        logging.info("Creating transcoder job configuration.")
        from google.protobuf.duration_pb2 import Duration
        if not audio_uri.startswith("gs://"):
            logging.error(f"Invalid audio URI format: {audio_uri}")
            raise ValueError(f"Invalid audio URI format: {audio_uri}")
        
        overlays = []
        for idx, uri in enumerate(image_uris):
            logging.info(f"Configuring overlay for image {idx + 1}: {uri}")
            # Create Duration objects for timing
            start_time = Duration()
            start_time.seconds = 5 * idx
        
            end_time = Duration()
            end_time.seconds = 5 * (idx + 1)
            overlays.append(
                transcoder_v1.types.Overlay(
                    image=transcoder_v1.types.Overlay.Image(
                        uri=uri,
                        alpha=1,
                        resolution=transcoder_v1.types.Overlay.NormalizedCoordinate(x=0, y=0)
                    ),
                    animations=[
                        transcoder_v1.types.Overlay.Animation(
                            animation_fade=transcoder_v1.types.Overlay.AnimationFade(
                                fade_type=transcoder_v1.types.Overlay.FadeType.FADE_IN,
                                start_time_offset=start_time,
                                end_time_offset=end_time,
                                xy=transcoder_v1.types.Overlay.NormalizedCoordinate(x=0.5, y=0.5)
                            )
                        ),
                        transcoder_v1.types.Overlay.Animation(
                            animation_fade=transcoder_v1.types.Overlay.AnimationFade(
                                fade_type=transcoder_v1.types.Overlay.FadeType.FADE_OUT,
                                start_time_offset=end_time,
                                end_time_offset=Duration(seconds=end_time.seconds + 1),
                                xy=transcoder_v1.types.Overlay.NormalizedCoordinate(x=0.5, y=0.5)
                            )
                        )
                    ]
                )
            )

        logging.info("Job configuration created successfully.")
        return transcoder_v1.types.JobConfig(
            elementary_streams=[
                transcoder_v1.types.ElementaryStream(
                    key="video-stream0",
                    video_stream=transcoder_v1.types.VideoStream(
                        h264=transcoder_v1.types.VideoStream.H264CodecSettings(
                            height_pixels=1080,
                            width_pixels=1920,
                            bitrate_bps=8000000,
                            frame_rate=30,
                            pixel_format="yuv420p"
                        )
                    )
                ),
                transcoder_v1.types.ElementaryStream(
                    key="audio-stream0",
                    audio_stream=transcoder_v1.types.AudioStream(
                        codec="aac",
                        bitrate_bps=256000,
                        channel_count=2,
                        sample_rate_hertz=48000
                    )
                )
            ],
            mux_streams=[
                transcoder_v1.types.MuxStream(
                    key="hd-mp4",
                    container="mp4",
                    elementary_streams=["video-stream0", "audio-stream0"]
                )
            ],
            inputs=[
                transcoder_v1.types.Input(
                    key="audio0",
                    uri=audio_uri
                )
            ],
            overlays=overlays
        )

    def _create_transcoder_job(self, image_uris: List[str], audio_uri: str) -> str:
        """Create and execute a transcoder job with proper API usage"""
        logging.info("Submitting transcoder job.")
        job_id = f"job-{uuid.uuid4()}"
        job_config = self._create_job_config(image_uris, audio_uri)
        
        job = transcoder_v1.types.Job(
            config=job_config,
            output_uri=f"gs://{self.gcs_handler.bucket_name}/videos/"
        )
        
        response = self.client.create_job(
            parent=self.parent,
            job=job
        )
        job_name = response.name
        logging.info(f"Transcoder job submitted: {job_name}")
        
        # Poll while the job state is either PENDING or RUNNING.
        pending_states = [
            transcoder_v1.types.Job.ProcessingState.PENDING,
            transcoder_v1.types.Job.ProcessingState.RUNNING,
        ]
        max_wait_seconds = 600  # e.g., 10 minutes
        start_time = time.time()
        
        logging.info("Polling transcoder job status...")
        while response.state in pending_states:
            if time.time() - start_time > max_wait_seconds:
                logging.error("Transcoding timed out.")
                raise RuntimeError("Transcoding timed out.")
            logging.info(f"Job state is {response.state.name}; waiting...")
            time.sleep(10)
            response = self.client.get_job(name=job_name)
        
        if response.state != transcoder_v1.types.Job.ProcessingState.SUCCEEDED:
            logging.error(f"Transcoding failed with state: {response.state}")
            raise RuntimeError(f"Transcoding failed: {response.state}")
        
        logging.info("Transcoding completed successfully.")
        return f"gs://{self.gcs_handler.bucket_name}/videos/{job_name.split('/')[-1]}.mp4"

    def create_video(self, audio_uri: str, script_data: list) -> str:
        """End-to-end video generation pipeline."""
        logging.info("Starting video creation pipeline.")
        try:
            # 1. AI-powered script analysis
            logging.info("Analyzing script...")
            analysis = self._analyze_script(script_data)
            
            # 2. Generate visual assets
            logging.info("Generating visual assets from analysis.")
            images = self._generate_images(analysis)
            image_uris = []
            for idx, img in enumerate(images):
                logging.info(f"Uploading image {idx + 1}.")
                uploaded_image = self.gcs_handler.upload_image(img, f"images/{uuid.uuid4()}.png")
                signed_uri = self.gcs_handler.signed_url_to_gcs_uri(uploaded_image)
                image_uris.append(signed_uri)
            
            # 3. Create transcoder job
            logging.info("Creating transcoder job with generated assets.")
            video_uri = self._create_transcoder_job(image_uris, audio_uri)
            
            final_url = self.gcs_handler.get_signed_url(video_uri.replace("gs://", ""))
            logging.info(f"Video creation completed successfully. Video URL: {final_url}")
            return final_url

        except Exception as e:
            logging.error(f"Video generation pipeline failed: {str(e)}")
            raise RuntimeError(f"Video creation error: {str(e)}")
