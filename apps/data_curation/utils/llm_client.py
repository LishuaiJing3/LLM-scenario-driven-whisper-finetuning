import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
import os
import json
from jinja2 import Template
import sqlite3
from apps.data_curation.utils.db import hash_scenario_text
from apps.data_curation.utils.language_mapping import get_language_code, is_supported_language

load_dotenv()


class LLMClient:
    def __init__(self, model="gemini-2.0-flash-exp", prompt_version="v1", db_path="data/assets/scenarios.db"):
        """
        Initializes the LLM client.
        :param model: The Gemini 2.0 model to use.
        :param prompt_version: The version of the prompts to load.
        :param db_path: Path to the SQLite database.
        """
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model)
        self.prompt_version = prompt_version
        self.db_path = db_path

    def _load_prompts(self, prompts_path):
        """
        Load system and user prompts from the specified versioned directory.
        :return: Tuple of (system_prompt, user_prompt).
        """
        #prompts_path = f"apps/data_curation/prompts/{self.prompt_version}/"
        print(f"Loading prompts from: {os.path.abspath(prompts_path)}")
        with open(f"{prompts_path}/system_prompt.txt", "r") as system_file:
            system_prompt = system_file.read()
        with open(f"{prompts_path}/user_prompt.txt", "r") as user_file:
            user_prompt = user_file.read()
        return system_prompt, user_prompt

    def generate_conversations(self, user_inputs, prompt_path, output_dir="data/assets"):
        """
        Generate conversations based on the provided user inputs and save to SQLite and file system.
        :param user_inputs: Dictionary containing the user-provided parameters.
        :return: List of unique hashes generated for the utterances.
        """
        # Load prompts
        system_prompt, user_prompt_template = self._load_prompts(prompt_path)

        # Render the user prompt using Jinja2 template
        template = Template(user_prompt_template)
        rendered_user_prompt = template.render(**user_inputs)

        # Combine system and user prompts
        final_prompt = f"{system_prompt}\n\nUser Prompt:\n{rendered_user_prompt}"

        # Define generation configuration
        generation_config = GenerationConfig(
            max_output_tokens=10000,
            temperature=0.7,
            top_p=0.9,
            response_mime_type="application/json"
        )

        # Call the Gemini 2.0 API
        response = self.model.generate_content(
            contents=final_prompt,
            generation_config=generation_config,
            safety_settings=[]
        )

        # Parse the output
        try:
            output = json.loads(response.text)
            print(json.dumps(output, indent=4))
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON response from the LLM.")

        # Save data and generate unique hashes for each utterance
        utterance_hashes = self._save_data(output, user_inputs, output_dir)
        return utterance_hashes

    def _save_data(self, llm_output, user_inputs, output_dir):
        """
        Save generated data to SQLite and the file system with unique hashes for each utterance.
        :param llm_output: Output from the LLM client.
        :param language: Full language name (e.g., "Danish").
        :return: List of unique hashes for the utterances.
        """
        scenario_dir = os.path.join(output_dir, user_inputs["language"])
        os.makedirs(scenario_dir, exist_ok=True)

        utterance_hashes = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for idx, output in enumerate(llm_output["outputs"]):
                utterance = output["utterance"]
                scenario_hash = hash_scenario_text(f"{user_inputs['language']}_{user_inputs['scenario']}_{utterance}_{idx}")
                audio_path = os.path.join(scenario_dir, f"{scenario_hash}_utterance.wav")
                # Normalize language input to code
                language_code = get_language_code(user_inputs['language'])
                if not language_code or not is_supported_language(language_code):
                    raise ValueError(f"Unsupported language: {user_inputs['language']}")

                try:
                    cursor.execute("""
                    INSERT INTO scenarios (scenario, scenario_id, language, language_code, utterance, audio_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_inputs["scenario"], scenario_hash, user_inputs["language"], language_code, utterance, audio_path))
                except sqlite3.IntegrityError:
                    print(f"Duplicate entry skipped for utterance: {utterance}")
                    continue

                # Save utterance JSON
                script_path = os.path.join(scenario_dir, f"{scenario_hash}_utterance.json")
                with open(script_path, "w") as script_file:
                    json.dump(output, script_file, indent=4)

                utterance_hashes.append(scenario_hash)

            conn.commit()
        print(f"Data saved to SQLite and assets folder. Generated hashes: {utterance_hashes}")
        return utterance_hashes
        
        
if __name__ == "__main__":
    llm_client = LLMClient(prompt_version="v1")

    user_inputs = {
        "language": "English",
        "scenario": "A customer enters a small bookstore on a rainy day.",
        "character": "A friendly, middle-aged bookstore owner who loves to chat with customers.",
        "request": "Greet the customer and make them feel welcome.",
        "nSample": 2,
        "tone": "Warm and inviting"
    }

    try:
        result = llm_client.generate_conversations(user_inputs, prompts_path="apps/data_curation/prompts/v1/", output_dir="data/assets")
        print("Generated Conversations:")
        print(json.dumps(result, indent=4))
    except ValueError as e:
        print(f"Error: {e}")
'''
{
"language": "English",
"scenario": "A customer enters a small bookstore on a rainy day.",
"character": "A friendly, middle-aged bookstore owner who loves to chat with customers.",
"request": "Greet the customer and make them feel welcome.",
"nSample": 2,
"tone": "Warm and inviting"
}

{
"language": "English",
"scenario": "A customer enters a small bookstore on a sunny day.",
"character": "A friendly, middle-aged bookstore owner who does not like to chat with customers.",
"request": "Greet the customer and make them feel uncomfortable.",
"nSample": 2,
"tone": "neutral"
}
'''