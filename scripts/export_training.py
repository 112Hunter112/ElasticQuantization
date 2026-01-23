import psycopg2
import csv
import os

# Connect to Postgres
conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/consistency_db")
cur = conn.cursor()

print("Fetching vectors from database...")
# Fetch vectors as string "[0.123, 0.456, ...]"
cur.execute("SELECT embedding::text FROM articles LIMIT 1000;")
rows = cur.fetchall()

# Ensure config directory exists
output_path = "config/training_data.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Exporting {len(rows)} vectors to {output_path}...")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in rows:
        # Postgres returns "[0.1, 0.2]", we strip brackets to get "0.1, 0.2"
        clean_vector = row[0].strip("[]")
        # Write as a row of numbers
        writer.writerow(clean_vector.split(","))

print("✅ Export complete!")
conn.close()