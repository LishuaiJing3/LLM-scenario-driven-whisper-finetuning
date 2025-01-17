import sqlite3
import os
import hashlib

import sqlite3

def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_sqlite_database(db_filename="assets/scenarios.db"):
    """
    Creates a SQLite database with the necessary schema for storing data.

    Args:
        db_filename (str): The name or path of the database file. Defaults to "assets/scenarios.db".
    """
    try:
        # Ensure the directory exists for the database file
        os.makedirs(os.path.dirname(db_filename), exist_ok=True)

        # Check if the database file already exists
        if os.path.exists(db_filename):
            print(f"Database file '{db_filename}' already exists.")
            return

        # Create a connection to the database (this creates the file if it doesn't exist)
        conn = sqlite3.connect(db_filename)

        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Create the single table 'dataset' with the required schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                language TEXT NOT NULL,
                language_code TEXT NOT NULL,
                utterance TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scenario_id, utterance)
            )
        ''')

        # Commit the changes to save the table creation
        conn.commit()

        print(f"Database file '{db_filename}' created successfully with the required schema.")

    except sqlite3.Error as e:
        print(f"An error occurred while creating the database: {e}")
    finally:
        # Ensure the database connection is always closed
        if conn:
            conn.close()


def hash_scenario_text(scenario_text):
    """
    Generates a unique hash for a given scenario text.

    Args:
        scenario_text (str): The scenario text to hash.

    Returns:
        str: A unique hash string.
    """
    return hashlib.sha256(scenario_text.encode()).hexdigest()[:16]  # 16-character hash


if __name__ == "__main__":
    # Define the default database file path
    db_file = "../../data/assets/scenarios.db"

    # Create the database with the required schema
    create_sqlite_database(db_file)

    # Example: Hash a scenario text
    sample_scenario = "A customer enters a small bookstore on a rainy day."
    print(f"Hash for scenario: {hash_scenario_text(sample_scenario)}")
