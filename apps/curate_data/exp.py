from utils.llm_client import LLMClient  
from utils.tts_client import TTSClient  

# Initialize LLM and TTS clients
llm_client = LLMClient(prompt_version="v1")
tts_client = TTSClient(tts_model="tts_models/en/ljspeech/tacotron2-DDC")

# Generate conversations using LLM client
user_inputs = {
    "Scenario": "A customer enters a small bookstore on a rainy day.",
    "Character": "A friendly, middle-aged bookstore owner who loves to chat with customers.",
    "Request": "Greet the customer and make them feel welcome.",
    "nSample": 2,
    "Tone": "Warm and inviting"
}

# Get the generated output from the LLM client
try:
    llm_output = llm_client.generate_conversations(user_inputs)
    llm_output_json = json.loads(llm_output)
    utterances = llm_output_json["outputs"]

    # Process utterances with TTS client
    lookup_table = tts_client.process_utterances(utterances, output_dir="audio_outputs")

    # Print the lookup table
    print("Generated audio lookup table:")
    print(json.dumps(lookup_table, indent=4))
except ValueError as e:
    print(f"Error in LLM client: {e}")
except KeyError as e:
    print(f"Missing expected key in LLM output: {e}")
