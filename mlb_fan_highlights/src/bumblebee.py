from vertexai.preview import reasoning_engines

# resource name = projects/1011675918473/locations/us-central1/reasoningEngines/2601796355038904320
REASONING_ENGINE_RESOURCE_NAME = "projects/1011675918473/locations/us-central1/reasoningEngines/2601796355038904320"  # Use the resource name

remote_agent = reasoning_engines.ReasoningEngine(REASONING_ENGINE_RESOURCE_NAME)
response = remote_agent.query(input="Give me a summary of the last Yankees plays.")
print(response)