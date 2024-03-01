import psycopg2
from psycopg2 import OperationalError

from config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

def connect_to_db():
    # Connect to PostgreSQL database
    conn = psycopg2.connect(
        dbname="digit_classifier",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    # print(DB_PASSWORD)
    return conn

def create_table(conn, table_name):
    # Create a Postgresql table with the given name
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            image_name TEXT,
            label INTEGER,
            processed_image_path TEXT
        );
    """)
    conn.commit()
    cursor.close()