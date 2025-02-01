from google.cloud import videointelligence_v1 as videointelligence
from google.cloud import storage
import tempfile
import os
import json
from datetime import timedelta
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import time
import uuid

class CloudVideoGenerator:
    def __init__(self, gcs_handler, location="us-central1"):
        """
        Initialize Video Generator with Google Cloud services.
        
        Args:
            gcs_handler: Instance of GCSHandler for secure storage operations
            location: Location for Vertex AI services
        """
        self.gcs_handler = gcs_handler
        self.project_id = "gem-rush-007"
        self.location = location
        
        # Initialize Video Intelligence client
        self.video_client = videointelligence.VideoIntelligenceServiceClient()
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self.imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
        
    def create_video_composition(self, images, audio_path, output_filename):
        """
        Create a video composition using Google Cloud services.
        
        Args:
            images: List of image paths
            audio_path: Path to audio file
            output_filename: Desired output filename
        """
        try:
            # Create a temporary manifest file for video composition
            manifest = {
                "compositions": [{
                    "id": "mlb_podcast",
                    "video": {
                        "tracks": [{
                            "clips": self._create_image_clips(images)
                        }],
                        "format": "mp4"
                    },
                    "audio": {
                        "tracks": [{
                            "clips": [{
                                "asset_id": "audio_track",
                                "start_time": "0s",
                                "end_time": "auto"
                            }]
                        }]
                    }
                }],
                "assets": self._prepare_assets(images, audio_path)
            }
            
            # Save manifest to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_manifest:
                json.dump(manifest, temp_manifest)
                manifest_path = temp_manifest.name
            
            # Upload manifest to GCS
            manifest_gcs_path = f"manifests/{uuid.uuid4()}.json"
            with open(manifest_path, 'rb') as f:
                self.gcs_handler.upload_audio(f.read(), manifest_gcs_path)
            
            # Create video composition request
            request = videointelligence.CreateVideoCompositionRequest(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                video_composition={
                    "manifest_path": f"gs://{self.gcs_handler.bucket_name}/{manifest_gcs_path}",
                    "output_path": f"gs://{self.gcs_handler.bucket_name}/videos/{output_filename}"
                }
            )
            
            # Start composition operation
            operation = self.video_client.create_video_composition(request)
            result = operation.result(timeout=300)  # Wait up to 5 minutes
            
            # Generate signed URL for the output video
            video_url = self.gcs_handler.refresh_signed_url(
                f"gs://{self.gcs_handler.bucket_name}/videos/{output_filename}"
            )
            
            return video_url
            
        except Exception as e:
            raise Exception(f"Error creating video composition: {str(e)}")
        finally:
            # Cleanup temporary files
            if 'manifest_path' in locals():
                os.remove(manifest_path)
    
    def _create_image_clips(self, images):
        """Create clip configurations for images."""
        clips = []
        duration = 5  # 5 seconds per image
        
        for i, image_path in enumerate(images):
            clips.append({
                "asset_id": f"image_{i}",
                "start_time": f"{i * duration}s",
                "end_time": f"{(i + 1) * duration}s",
                "transition": {
                    "type": "FADE",
                    "duration": "1s"
                }
            })
        
        return clips
    
    def _prepare_assets(self, images, audio_path):
        """Prepare asset configurations for manifest."""
        assets = {}
        
        # Add image assets
        for i, image_path in enumerate(images):
            assets[f"image_{i}"] = {
                "sources": [{
                    "uri": image_path,
                    "type": "IMAGE"
                }]
            }
        
        # Add audio asset
        assets["audio_track"] = {
            "sources": [{
                "uri": audio_path,
                "type": "AUDIO"
            }]
        }
        
        return assets
    
    def generate_video(self, audio_path, script):
        """
        Generate a complete video from audio and script.
        
        Args:
            audio_path: Path to the audio file
            script: Script content for generating images
        """
        try:
            # Generate images from script
            image_prompts = self.generate_images_from_script(script)
            image_paths = []
            
            # Generate and upload images
            for prompt in image_prompts:
                image_path = self._generate_and_upload_image(prompt)
                image_paths.append(image_path)
            
            # Create unique output filename
            output_filename = f"highlight-{uuid.uuid4()}.mp4"
            
            # Create video composition
            video_url = self.create_video_composition(
                image_paths,
                audio_path,
                output_filename
            )
            
            return video_url
            
        except Exception as e:
            raise Exception(f"Error generating video: {str(e)}")
    
    def _generate_and_upload_image(self, prompt):
        """Generate image using Vertex AI and upload to GCS."""
        try:
            # Generate image
            response = self.imagen_model.generate_images(
                prompt=prompt,
                number_of_images=1
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                image = response.images[0]
                image_path = temp_file.name
                image.save(image_path)
                
                # Upload to GCS
                filename = f"images/image_{uuid.uuid4()}.jpg"
                with open(image_path, 'rb') as f:
                    gcs_path = self.gcs_handler.upload_audio(f.read(), filename)
                
                os.remove(image_path)
                return gcs_path
                
        except Exception as e:
            raise Exception(f"Error generating and uploading image: {str(e)}")