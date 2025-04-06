import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
      try:
          data = json.load(f)
          return data
      except json.JSONDecodeError:
          print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []

def is_css(sub_url, text):
    return any(map(lambda x: x.lower().startswith('css'), sub_url.split(".")))

if __name__ == "__main__":
    FOLDER_PATH = "hackathon_data/"
    OUTPUT_PATH = "filtered_data/"

    files_in_folder = os.listdir(FOLDER_PATH)

    total_docs_pre_filter = 0
    total_docs_post_filter = 0
    
    for filename in tqdm(files_in_folder):
        if filename.endswith('.json'):
            docs = load_documents(os.path.join(FOLDER_PATH, filename))
            if not docs:
                continue
            try:
                url = docs['url']
            except KeyError:
                url = docs['website_url']

            doc_id = docs['doc_id']
            out = {'doc_id': doc_id, 'url': url, 'text_by_page_url': {}}

            min_len = min(map(len, docs['text_by_page_url'].keys()))

            for sub_url in docs['text_by_page_url'].keys():
                total_docs_pre_filter += 1
                text = docs['text_by_page_url'][sub_url]
                
                # FILTER LOGIC
                # Exclude CSS Files
                if is_css(sub_url, text):
                    continue
                
                # Only Include Landing Pages and Product Related Pages
                contains_product = "product" in sub_url.lower()
            
                is_base_url = len(sub_url) == min_len

                if not (contains_product or is_base_url):
                    continue
                
                out['text_by_page_url'][sub_url] = text
                total_docs_post_filter += 1

            if out:
                with open(os.path.join(OUTPUT_PATH, filename), 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=4)


    print(f"Total documents before filtering: {total_docs_pre_filter}")
    print(f"Total documents after filtering: {total_docs_post_filter}")
    print(f"Filtered {total_docs_pre_filter - total_docs_post_filter} documents. {total_docs_post_filter / total_docs_pre_filter * 100:.2f}% of documents were kept.")
    print(f"Filtered data saved to {OUTPUT_PATH}")
