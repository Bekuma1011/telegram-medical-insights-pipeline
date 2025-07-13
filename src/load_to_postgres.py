# scripts/load_raw_data.py
import os
import json
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL connection details
conn_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Data lake path
DATA_LAKE_PATH = Path("../data/raw/telegram_messages")

# Connect to PostgreSQL and create raw schema/table
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()
cursor.execute("""
    CREATE SCHEMA IF NOT EXISTS raw;
    CREATE TABLE IF NOT EXISTS raw.telegram_messages (
        channel_name VARCHAR(255),
        message_date DATE,
        message_id BIGINT,
        date TIMESTAMP,
        text TEXT,
        sender_id BIGINT,
        media BOOLEAN,
        views INTEGER,
        forwards INTEGER
    );
""")
conn.commit()

# Load JSON files
for channel_dir in DATA_LAKE_PATH.iterdir():
    if not channel_dir.is_dir():
        continue
    channel_name = channel_dir.name
    for json_file in channel_dir.glob("*.json"):
        message_date = json_file.stem  # YYYY-MM-DD
        with open(json_file, 'r', encoding='utf-8') as f:
            messages = [json.loads(line) for line in f]
        
        # Prepare data for batch insert
        records = [
            (
                channel_name,
                message_date,
                msg["message_id"],
                msg["date"],
                msg["text"],
                msg["sender_id"],
                msg["media"],
                msg["views"],
                msg["forwards"]
            )
            for msg in messages
        ]
        
        # Batch insert
        query = """
            INSERT INTO raw.telegram_messages (
                channel_name, message_date, message_id, date, text, 
                sender_id, media, views, forwards
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cursor, query, records)
        conn.commit()

cursor.close()
conn.close()