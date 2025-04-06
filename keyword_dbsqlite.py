import json
import os
import tqdm
import sqlite3
import os


def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []


def make_db():
    """Creates a SQLite database and adds documents to it."""
    folder_path = "hackathon_data"
    files_in_folder = os.listdir(folder_path)

    # Delete old DB for a clean slate (optional)
    if os.path.exists("pages.db"):
        os.remove("pages.db")

    # Connect to SQLite and enable FTS5
    conn = sqlite3.connect("pages.db")
    cursor = conn.cursor()

    # Create FTS5 virtual table
    cursor.execute(
        """
        CREATE VIRTUAL TABLE pages USING fts5(doc_id, content);
    """
    )

    for file in tqdm.tqdm(files_in_folder, desc="Processing files"):
        if file.endswith(".json"):
            json_file = load_documents(os.path.join(folder_path, file))
            if json_file:
                cursor.executemany(
                    "INSERT INTO pages(doc_id, content) VALUES (?, ?);",
                    list(json_file["text_by_page_url"].items()),
                )
            else:
                print(f"Skipping {file}, no valid JSON data found.")

    conn.commit()


# SEARCHING
def search_documents_bm25(keywords, top_k=5):
    """Searches the database using BM25 ranking."""
    conn = sqlite3.connect("pages.db")
    cursor = conn.cursor()
    query = " AND ".join(keywords)
    cursor.execute(
        """
        SELECT doc_id, content, bm25(pages) AS relevance 
        FROM pages 
        WHERE content MATCH ? 
        ORDER BY relevance DESC
        LIMIT ?;
    """,
        (query, top_k),
    )
    results = cursor.fetchall()
    return results


if __name__ == "__main__":
    make_db()

    # Example queries
    queries = ["Germany", "Aluminium", "Internship"]

    print(f"\n🔎 Query: '{queries}'")
    for doc_id, content, score in search_documents_bm25(queries):
        print(f"📄 Doc {doc_id} / {score}: {content[:500]}")


