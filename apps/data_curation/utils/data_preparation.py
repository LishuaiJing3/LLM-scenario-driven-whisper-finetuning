import sqlite3
import json
import os

def prepare_whisper_data(db_path, output_dir="data/training_data_whisper"):
    """
    Prepare Whisper-compatible data from the database.
    :param db_path: Path to the SQLite database.
    :param output_dir: Directory to save the prepared dataset.
    :return: Path to the prepared JSON dataset.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to fetch audio paths and transcriptions
    query = "SELECT audio_path, utterance, language_code FROM scenarios"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Prepare data in Whisper-compatible format
    prepared_data = []
    for row in rows:
        audio_path, text, language = row
        prepared_data.append({
            "audio_filepath": audio_path,
            "text": text,
            "language_code": language
        })

    # Save the prepared data as a JSON file
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "prepared_data.json")
    with open(data_path, "w") as f:
        json.dump(prepared_data, f, indent=4)

    print(f"Prepared data saved to: {data_path}")
    return data_path


if __name__ == "__main__":
    db_path = "data/assets/scenarios.db"
    prepared_data_path = prepare_whisper_data(db_path)
    print(f"Prepared dataset is available at: {prepared_data_path}")
