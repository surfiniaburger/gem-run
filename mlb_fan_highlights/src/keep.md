def fetch_player_plays(player_name: str, limit: int = 100) -> dict:
    LOOKER_STUDIO_BASE_URL = "https://lookerstudio.google.com/embed/reporting/f60f900b-9d43-46b8-b46a-4fba57e7637e/page/p_jsdpfv6qod"  # Replace with your actual base report URL
    """
    Fetches play-by-play data for a specific player from Dodgers games and generates a Looker Studio iframe URL.

    Args:
        player_name (str): Full name of the player
        limit (int, optional): Maximum number of plays to return. Defaults to 100.

    Returns:
        dict: A dictionary containing:
            - iframe_url (str): Looker Studio iframe URL with a filter for the player name
            - plays (list): List of dictionaries containing play-by-play data
    """
    try:
        query = """
        SELECT
            p.play_id,
            p.inning,
            p.half_inning,
            p.event,
            p.event_type,
            p.description,
            p.start_time,
            g.official_date as game_date,
         
        FROM
            `gem-rush-007.dodgers_mlb_data_2024.plays` AS p
        INNER JOIN 
            `gem-rush-007.dodgers_mlb_data_2024.games` AS g 
            ON p.game_id = g.game_id
        INNER JOIN
            `gem-rush-007.dodgers_mlb_data_2024.roster` AS r
            ON (p.batter_id = r.player_id OR p.pitcher_id = r.player_id)
        
        WHERE
            r.full_name = @player_name
            AND (g.home_team_id = 119 OR g.away_team_id = 119)
        ORDER BY 
            g.official_date DESC,
            p.start_time ASC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("player_name", "STRING", player_name),
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())

        # Construct the Looker Studio iframe URL with a filter for the player name
        params = {
            "ds6": f"player_name_{player_name}",  # Replace 'df1' with the actual filter ID in your Looker Studio report, this is an example.
        }
        encoded_params = urllib.parse.urlencode(params)
        iframe_url = f"{LOOKER_STUDIO_BASE_URL}?{encoded_params}"
        

        # Convert the results to dictionaries and format datetime objects
        formatted_results = []
        for row in results:
            row_dict = dict(row)
            # Convert datetime objects to ISO format strings
            if 'start_time' in row_dict and row_dict['start_time']:
                row_dict['start_time'] = row_dict['start_time'].isoformat()
            if 'game_date' in row_dict and row_dict['game_date']:
                row_dict['game_date'] = row_dict['game_date'].isoformat()
            formatted_results.append(row_dict)
        return {
            "iframe_url": iframe_url,
            "plays": formatted_results
        }

    except Exception as e:
        logging.error(f"Error in fetch_player_plays: {e}")
        return []





class CloudVideoGenerator:
    def __init__(self, gcs_handler):
        self.gcs_handler = gcs_handler
        self.parent = f"projects/{gcs_handler.project_id}/locations/us-central1"
        
        # Initialize AI models with latest configurations
        logging.info("Initializing Vertex AI client and Imagen model")
        self.genai_client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
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


    def _parse_gemini_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse and validate Gemini response with enhanced error handling and cleanup."""
        logging.info("Parsing Gemini response.")
        
        def clean_json_string(text: str) -> str:
            """Clean and prepare JSON string for parsing."""
            # Remove markdown code blocks
            text = re.sub(r'^[^{]*', '', text)  # Remove anything before first {
            text = re.sub(r'[^}]*$', '', text)  # Remove anything after last }
            
            # Fix common JSON formatting issues
           # text = re.sub(r'(?<=\d),(?=\d)', '.', text)  # Fix decimal numbers
               # Replace single quotes with double quotes
            text = text.replace("'", '"')
            # Add quotes to bare keys (only for keys that do not start with a quote)
            text = re.sub(r'(?<=\{|,)\s*(\w+)\s*:', r'"\1":', text)
            # Remove trailing commas before closing braces or brackets
            text = re.sub(r',\s*([\]}])', r'\1', text)
            
            return text.strip()

        def extract_json(text: str) -> str:
            """Extract JSON object from text using a more reliable method."""
            # Look for content between curly braces, handling nested structures
            stack = []
            start = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    if not stack:
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            return text[start:i+1]
            
            raise ValueError("No valid JSON object found")
        
        def validate_response_structure(parsed_data: Dict) -> None:
            """Validate the required structure of the parsed response."""
            required_fields = {
                'key_moments': list,
                'theme': str,
                'color_palette': dict,
                'graphics_style': str,
                'audio_intensity': (int, float, str)
            }
            
            for field, expected_type in required_fields.items():
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(parsed_data[field], expected_type):
                    raise ValueError(f"Invalid type for {field}: expected {expected_type}, got {type(parsed_data[field])}")
            
            # Validate key_moments structure
            for idx, moment in enumerate(parsed_data['key_moments']):
                required_moment_fields = {
                    'timestamp': str,
                    'description': str,
                    'visual_prompt': str,
                    'duration': (int, float),
                    'transition': str
                }
                
                for field, expected_type in required_moment_fields.items():
                    if field not in moment:
                        raise ValueError(f"Missing required field '{field}' in moment {idx}")
                    if not isinstance(moment[field], expected_type):
                        raise ValueError(f"Invalid type for '{field}' in moment {idx}")
        
        try:
            # First attempt: direct JSON parsing
            try:
                cleaned_text = clean_json_string(raw_response)
                parsed_data = json.loads(cleaned_text)
               
            # Validate structure
                required_fields = {
                'key_moments': list,
                'theme': str,
                'color_palette': dict,
                'graphics_style': str,
                'audio_intensity': (int, float)
            }
            
                for field, expected_type in required_fields.items():
                   if field not in parsed_data:
                     raise ValueError(f"Missing required field: {field}")
                   if not isinstance(parsed_data[field], expected_type):
                     raise ValueError(f"Invalid type for {field}")
            
                logging.info("Successfully parsed and validated Gemini response")
                return parsed_data
                        
            except json.JSONDecodeError as e:
                logging.warning(f"Initial JSON parsing failed: {str(e)}")
                
                # Second attempt: Try to extract JSON using improved method
                json_text = extract_json(raw_response)
                cleaned_text = clean_json_string(json_text)
                parsed_data = json.loads(cleaned_text)
            
            # Validate the structure
            validate_response_structure(parsed_data)
            
            logging.info("Successfully parsed and validated Gemini response")
            return parsed_data
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            logging.error(error_msg)
            logging.debug(f"Raw response: {raw_response[:500]}...")  # Log first 500 chars
            raise ValueError(error_msg)
            
        except ValueError as e:
            error_msg = f"Response validation error: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            logging.error(error_msg)
            logging.debug(f"Raw response: {raw_response[:500]}...")
            raise ValueError(error_msg)
        
    def _analyze_script(self, script_data: list) -> Dict[str, Any]:
        """Analyze podcast script using Gemini with enhanced prompt and error handling."""
        logging.info("Starting script analysis.")
        full_text = " ".join([segment['text'] for segment in script_data])
        
        # Improved prompt with explicit JSON structure
        analysis_prompt = """
        You are a JSON generator that must analyze this baseball podcast script and return ONLY a valid JSON object.
        
        Rules:
        1. Return ONLY the JSON object, no explanation or wrapper text
        2. Ensure ALL properties and values are properly quoted
        3. Include ALL required fields exactly as shown
        4. Use ONLY the specified data types
        5. Include ALL necessary commas between elements
        6. Use proper JSON number format (no trailing decimal)
        
        Required JSON structure:
        {
            "key_moments": [
                {
                    "timestamp": "MM:SS",
                    "description": "string",
                    "visual_prompt": "string",
                    "duration": 0.0,
                    "transition": "string"
                }
            ],
            "theme": "string",
            "color_palette": {
                "primary": "string",
                "secondary": "string",
                "accent": "string"
            },
            "graphics_style": "string",
            "audio_intensity": 0.0
        }
        
        Analyze this script and fill in appropriate values while maintaining the exact structure above:

   
        """
        
        try:
            logging.info("Sending analysis request to Gemini model.")
            response = self.genai_client.models.generate_content(
                model="gemini-1.5-flash-002",
                contents=[{
                    "role": "user", 
                    "parts": [{
                        "text": analysis_prompt + "\n\nScript to analyze:\n" + full_text
                    }]
                }],
                config=GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more consistent JSON
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                    safety_settings=self.safety_config
                ),
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
                
          
            # Pre-process response to ensure it only contains JSON
            response_text = response.text.strip()
            if '```json' in response_text:
                # Extract content between JSON code blocks if present
                response_text = response_text.split('```json')[-1].split('```')[0].strip()
            
            # Validate JSON structure before parsing
            try:
                json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON structure received: {str(e)}")
                logging.debug(f"Raw response: {response_text[:500]}...")
                print(response_text)
                
                # Attempt basic JSON structure fixes
                response_text = response_text.replace('\\n', '\n')
                response_text = response_text.replace('\\"', '"')
                response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)  # Remove trailing commas
                
            return self._parse_gemini_response(response_text)
            
        except Exception as e:
            error_msg = f"Script analysis failed: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance the input prompt for an MLB podcast by adding vivid commentary flair."""
        # Remove extra whitespace from the original prompt
        base_prompt = prompt.strip()
        
        # Ensure the prompt explicitly references MLB context
        if "MLB" not in base_prompt.upper():
            base_prompt = f"MLB: {base_prompt}"
        
        # Append additional dramatic and descriptive commentary
        enhanced_prompt = (
            f"{base_prompt} – In a game that defies expectations, witness heart-stopping plays, "
            f"thunderous home runs, and strategic brilliance unfolding on the diamond. "
            f"Feel the roar of the crowd, the crack of the bat, and the adrenaline-pumping tension "
            f"of every inning. Get ready for an immersive, play-by-play narrative that brings America's pastime to life!"
        )
        
        return enhanced_prompt

    def _create_default_image(self) -> bytes:
        """Creates a white default image with proper RGB format."""
        logging.info("Creating a default image as fallback.")
        from PIL import Image
        from io import BytesIO

        # Create RGB image instead of RGBA to avoid alpha channel issues
        image = Image.new('RGB', (1920, 1080), (255, 255, 255))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        logging.info("Default image created successfully.")
        return img_byte_arr.getvalue()
    
    def _retry_with_backoff(self, operation, max_retries=6, initial_delay=5, max_delay=120):
        """Execute operation with improved exponential backoff retry logic."""
        logging.info("Starting retry with enhanced exponential backoff.")
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                # Add initial wait before first attempt
                if attempt > 0:
                    jitter = random.uniform(0.8, 1.2)  # Add randomness to avoid thundering herd
                    wait_time = min(delay * jitter, max_delay)
                    logging.info(f"Waiting for {wait_time:.2f} seconds before attempt {attempt + 1}")
                    time.sleep(wait_time)
                
                result = operation()
                
                # Add validation check for empty results
                if not result or (hasattr(result, 'images') and not result.images):
                    raise ValueError("Empty result received from image generation")
                    
                # Add completion check
                if hasattr(result, 'images'):
                    for img in result.images:
                        if not img or not img._image_bytes:
                            raise ValueError("Incomplete image generation result")
                
                logging.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                delay = min(delay * 2, max_delay)  # Exponential backoff with max delay cap
        
        logging.error(f"Maximum retry attempts reached. Operation failed: {str(last_exception)}")
        raise last_exception

    def _generate_images(self, analysis: Dict[str, Any]) -> List[bytes]:
        """Generate images with improved handling and validation."""
        logging.info("Starting image generation for key moments.")
        images = []
        
        for idx, moment in enumerate(analysis['key_moments']):
            logging.info(f"Generating image for key moment {idx + 1}: {moment.get('description', 'No description')}")
            
            try:
                enhanced_prompt = self._enhance_prompt(moment['visual_prompt'])
                logging.debug(f"Enhanced prompt: {enhanced_prompt}")
                
                def generate_image():
                    # Wrap image generation in a function that includes validation
                    response = self.imagen_model.generate_images(
                        prompt=enhanced_prompt,
                        aspect_ratio="16:9",
                        number_of_images=1,
                        safety_filter_level="block_some",
                    )
                    
                    # Add immediate validation of the response
                    if not response or not response.images:
                        raise ValueError("No images in response")
                    
                    image_bytes = response.images[0]._image_bytes
                    if not image_bytes:
                        raise ValueError("Empty image bytes received")
                    
                    return response
                
                # Use enhanced retry mechanism
                response = self._retry_with_backoff(
                    generate_image,
                    max_retries=6,
                    initial_delay=10  # Increased initial delay
                )
                
                images.append(response.images[0]._image_bytes)
                logging.info(f"Image generation successful for moment {idx + 1}")
                
                # Add cooldown period between generations to avoid rate limiting
                if idx < len(analysis['key_moments']) - 1:
                    cooldown = random.uniform(2, 5)
                    logging.info(f"Cooling down for {cooldown:.2f} seconds before next generation")
                    time.sleep(cooldown)
                    
            except Exception as e:
                error_msg = f"Image generation failed for moment {idx + 1} after all retries: {str(e)}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
                
        logging.info("Completed image generation for all key moments successfully")
        return images





class CloudVideoGenerator:
    def __init__(self, gcs_handler):
        self.gcs_handler = gcs_handler
        self.parent = f"projects/{gcs_handler.project_id}/locations/us-central1"
        
        # Initialize AI models with latest configurations
        logging.info("Initializing Vertex AI client and Imagen model")
        self.genai_client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
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


    def _parse_gemini_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse and validate Gemini response with enhanced error handling and cleanup."""
        logging.info("Parsing Gemini response.")
        
        def clean_json_string(text: str) -> str:
            """Clean and prepare JSON string for parsing."""
            # Remove markdown code blocks
            text = re.sub(r'^[^{]*', '', text)  # Remove anything before first {
            text = re.sub(r'[^}]*$', '', text)  # Remove anything after last }
            
            # Fix common JSON formatting issues
           # text = re.sub(r'(?<=\d),(?=\d)', '.', text)  # Fix decimal numbers
               # Replace single quotes with double quotes
            text = text.replace("'", '"')
            # Add quotes to bare keys (only for keys that do not start with a quote)
            text = re.sub(r'(?<=\{|,)\s*(\w+)\s*:', r'"\1":', text)
            
            
            return text.strip()

        def extract_json(text: str) -> str:
            """Extract JSON object from text using a more reliable method."""
            # Look for content between curly braces, handling nested structures
            stack = []
            start = -1
            
            for i, char in enumerate(text):
                if char == '{':
                    if not stack:
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            return text[start:i+1]
            
            raise ValueError("No valid JSON object found")
        
        def validate_response_structure(parsed_data: Dict) -> None:
            """Validate the required structure of the parsed response."""
            required_fields = {
                'key_moments': list,
                'theme': str,
                'color_palette': dict,
                'graphics_style': str,
                'audio_intensity': (int, float, str)
            }
            
            for field, expected_type in required_fields.items():
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(parsed_data[field], expected_type):
                    raise ValueError(f"Invalid type for {field}: expected {expected_type}, got {type(parsed_data[field])}")
            
            # Validate key_moments structure
            for idx, moment in enumerate(parsed_data['key_moments']):
                required_moment_fields = {
                    'timestamp': str,
                    'description': str,
                    'visual_prompt': str,
                    'duration': (int, float),
                    'transition': str
                }
                
                for field, expected_type in required_moment_fields.items():
                    if field not in moment:
                        raise ValueError(f"Missing required field '{field}' in moment {idx}")
                    if not isinstance(moment[field], expected_type):
                        raise ValueError(f"Invalid type for '{field}' in moment {idx}")
        
        try:
            # First attempt: direct JSON parsing
            try:
                cleaned_text = clean_json_string(raw_response)
                parsed_data = json.loads(cleaned_text)
               
            # Validate structure
                required_fields = {
                'key_moments': list,
                'theme': str,
                'color_palette': dict,
                'graphics_style': str,
                'audio_intensity': (int, float)
            }
            
                for field, expected_type in required_fields.items():
                   if field not in parsed_data:
                     raise ValueError(f"Missing required field: {field}")
                   if not isinstance(parsed_data[field], expected_type):
                     raise ValueError(f"Invalid type for {field}")
            
                logging.info("Successfully parsed and validated Gemini response")
                return parsed_data
                        
            except json.JSONDecodeError as e:
                logging.warning(f"Initial JSON parsing failed: {str(e)}")
                
                # Second attempt: Try to extract JSON using improved method
                json_text = extract_json(raw_response)
                cleaned_text = clean_json_string(json_text)
                parsed_data = json.loads(cleaned_text)
            
            # Validate the structure
            validate_response_structure(parsed_data)
            
            logging.info("Successfully parsed and validated Gemini response")
            return parsed_data
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            logging.error(error_msg)
            logging.debug(f"Raw response: {raw_response[:500]}...")  # Log first 500 chars
            raise ValueError(error_msg)
            
        except ValueError as e:
            error_msg = f"Response validation error: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            logging.error(error_msg)
            logging.debug(f"Raw response: {raw_response[:500]}...")
            raise ValueError(error_msg)
        
    def _analyze_script(self, script_data: list) -> Dict[str, Any]:
        """Analyze podcast script using Gemini with enhanced prompt and error handling."""
        logging.info("Starting script analysis.")
        full_text = " ".join([segment['text'] for segment in script_data])
        
        # Improved prompt with explicit JSON structure
        analysis_prompt = """
        You are a JSON generator that must analyze this baseball podcast script and return ONLY a valid JSON object.
        
        Rules:
        1. Return ONLY the JSON object, no explanation or wrapper text
        2. Ensure ALL properties and values are properly quoted
        3. Include ALL required fields exactly as shown
        4. Use ONLY the specified data types
        5. Include ALL necessary commas between elements
        6. Use proper JSON number format (no trailing decimal)
        
        Required JSON structure:
        {
            "key_moments": [
                {
                    "timestamp": "MM:SS",
                    "description": "string",
                    "visual_prompt": "string",
                    "duration": 0.0,
                    "transition": "string"
                }
            ],
            "theme": "string",
            "color_palette": {
                "primary": "string",
                "secondary": "string",
                "accent": "string"
            },
            "graphics_style": "string",
            "audio_intensity": 0.0
        }
        
        Analyze this script and fill in appropriate values while maintaining the exact structure above:

   
        """
        
        try:
            logging.info("Sending analysis request to Gemini model.")
            response = self.genai_client.models.generate_content(
                model="gemini-1.5-flash-002",
                contents=[{
                    "role": "user", 
                    "parts": [{
                        "text": analysis_prompt + "\n\nScript to analyze:\n" + full_text
                    }]
                }],
                config=GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more consistent JSON
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                    safety_settings=self.safety_config
                ),
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
                
          
            # Pre-process response to ensure it only contains JSON
            response_text = response.text.strip()
            if '```json' in response_text:
                # Extract content between JSON code blocks if present
                response_text = response_text.split('```json')[-1].split('```')[0].strip()
            
            # Validate JSON structure before parsing
            try:
                json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON structure received: {str(e)}")
                logging.debug(f"Raw response: {response_text[:500]}...")
                print(response_text)
                
                # Attempt basic JSON structure fixes
                response_text = response_text.replace('\\n', '\n')
                response_text = response_text.replace('\\"', '"')
                response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)  # Remove trailing commas
                
            return self._parse_gemini_response(response_text)
            
        except Exception as e:
            error_msg = f"Script analysis failed: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance the input prompt for an MLB podcast by adding vivid commentary flair."""
        # Remove extra whitespace from the original prompt
        base_prompt = prompt.strip()
        
        # Ensure the prompt explicitly references MLB context
        if "MLB" not in base_prompt.upper():
            base_prompt = f"MLB: {base_prompt}"
        
        # Append additional dramatic and descriptive commentary
        enhanced_prompt = (
            f"{base_prompt} – In a game that defies expectations, witness heart-stopping plays, "
            f"thunderous home runs, and strategic brilliance unfolding on the diamond. "
            f"Feel the roar of the crowd, the crack of the bat, and the adrenaline-pumping tension "
            f"of every inning. Get ready for an immersive, play-by-play narrative that brings America's pastime to life!"
        )
        
        return enhanced_prompt

    def _create_default_image(self) -> bytes:
        """Creates a white default image with proper RGB format."""
        logging.info("Creating a default image as fallback.")
        from PIL import Image
        from io import BytesIO

        # Create RGB image instead of RGBA to avoid alpha channel issues
        image = Image.new('RGB', (1920, 1080), (255, 255, 255))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        logging.info("Default image created successfully.")
        return img_byte_arr.getvalue()
    
    def _retry_with_backoff(self, operation, max_retries=6, initial_delay=5, max_delay=120):
        """Execute operation with improved exponential backoff retry logic."""
        logging.info("Starting retry with enhanced exponential backoff.")
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                # Add initial wait before first attempt
                if attempt > 0:
                    jitter = random.uniform(0.8, 1.2)  # Add randomness to avoid thundering herd
                    wait_time = min(delay * jitter, max_delay)
                    logging.info(f"Waiting for {wait_time:.2f} seconds before attempt {attempt + 1}")
                    time.sleep(wait_time)
                
                result = operation()
                
                # Add validation check for empty results
                if not result or (hasattr(result, 'images') and not result.images):
                    raise ValueError("Empty result received from image generation")
                    
                # Add completion check
                if hasattr(result, 'images'):
                    for img in result.images:
                        if not img or not img._image_bytes:
                            raise ValueError("Incomplete image generation result")
                
                logging.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                delay = min(delay * 2, max_delay)  # Exponential backoff with max delay cap
        
        logging.error(f"Maximum retry attempts reached. Operation failed: {str(last_exception)}")
        raise last_exception

    def _generate_images(self, analysis: Dict[str, Any]) -> List[bytes]:
        """Generate images with improved handling and validation."""
        logging.info("Starting image generation for key moments.")
        images = []
        
        for idx, moment in enumerate(analysis['key_moments']):
            logging.info(f"Generating image for key moment {idx + 1}: {moment.get('description', 'No description')}")
            
            try:
                enhanced_prompt = self._enhance_prompt(moment['visual_prompt'])
                logging.debug(f"Enhanced prompt: {enhanced_prompt}")
                
                def generate_image():
                    # Wrap image generation in a function that includes validation
                    response = self.imagen_model.generate_images(
                        prompt=enhanced_prompt,
                        aspect_ratio="16:9",
                        number_of_images=1,
                        safety_filter_level="block_some",
                    )
                    
                    # Add immediate validation of the response
                    if not response or not response.images:
                        raise ValueError("No images in response")
                    
                    image_bytes = response.images[0]._image_bytes
                    if not image_bytes:
                        raise ValueError("Empty image bytes received")
                    
                    return response
                
                # Use enhanced retry mechanism
                response = self._retry_with_backoff(
                    generate_image,
                    max_retries=6,
                    initial_delay=10  # Increased initial delay
                )
                
                images.append(response.images[0]._image_bytes)
                logging.info(f"Image generation successful for moment {idx + 1}")
                
                # Add cooldown period between generations to avoid rate limiting
                if idx < len(analysis['key_moments']) - 1:
                    cooldown = random.uniform(2, 5)
                    logging.info(f"Cooling down for {cooldown:.2f} seconds before next generation")
                    time.sleep(cooldown)
                    
            except Exception as e:
                error_msg = f"Image generation failed for moment {idx + 1} after all retries: {str(e)}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
                
        logging.info("Completed image generation for all key moments successfully")
        return images
