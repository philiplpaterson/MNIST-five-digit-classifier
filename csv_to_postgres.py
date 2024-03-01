import pandas as pd
import psycopg2
from db_utils import connect_to_db, create_table

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

def import_csv_to_db(csv_file, conn, target_table_name):
    df = pd.read_csv(csv_file)

    # Insert the CSV data into the postgresql table
    cursor = conn.cursor()
    for index, row in df.iterrows():
        cursor.execute(
            f"INSERT INTO {target_table_name} (image_name, label) VALUES (%s, %s)",
            (row['Image Name'], row['Label'])
        )
    conn.commit()
    cursor.close()

if __name__ == '__main__':
    train_anno_csv = 'data_progS24/labels/train_anno.csv'
    test_anno_csv = 'data_progS24/labels/test_anno.csv'
    
    conn = connect_to_db()
    create_table(conn, "train_annotations")
    create_table(conn, "test_annotations")
    import_csv_to_db(train_anno_csv, conn, 'train_annotations')
    import_csv_to_db(test_anno_csv, conn, 'test_annotations')
    conn.close()
