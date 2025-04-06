import os
import json
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core import Document
from llama_index.core.retrievers import AutoMergingRetriever
from tqdm import tqdm


# CONGIG
api_key = ...

def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
      try:
          data = json.load(f)
          return data
      except json.JSONDecodeError:
          print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []

def json_to_nodes(json_file):
    """Converts the JSON file into chunks."""

    # Step 1: Create a document for each sub-url on the website
    docs = []
    i = 0

    ## Find the base URL
    all_sub_urls = list(json_file['text_by_page_url'].keys())
    base_url = min(all_sub_urls, key=len)

    for sub_url, content in json_file['text_by_page_url'].items():
        # Create a document for each URL on the website

        docs.append(Document(
            text=content,
            metadata={
                "website_id": json_file['doc_id'],
                "website_url": base_url,
                "pageID": 'page_' + str(i),
                "url": sub_url,
                "total_content_length": len(content),
            }
        ))
        i += 1

    # Step 2: Convert each document into hierarchical chunks

    ## Option 1: HierarchicalNodeParser --> Has to be combined with AutoMergingRetriever !!
    # node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[10000, 1024])

    ## Option 2: SentenceSplitter
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    chunks = node_parser.get_nodes_from_documents(docs)

    return chunks

def json_to_nodes_base_only(json_file):
    """Converts the JSON file into chunks."""

    # Step 1: Create a document for each sub-url on the website
    docs = []
    i = 0

    ## Find the base URL
    all_sub_urls = list(json_file['text_by_page_url'].keys())
    base_url = min(all_sub_urls, key=len)

    base_url_doc = json_file['text_by_page_url'][base_url]

    docs.append(Document(
        text=base_url_doc,
        metadata={
            "website_id": json_file['doc_id'],
            "website_url": base_url,
            "pageID": 'page_' + str(i),
            "url": base_url,
            "total_content_length": len(base_url_doc),
        }
    ))

    if (len(docs) > 1):
        print("Warning: More than one document found. Only the first one will be used.")

    # Step 2: Convert each document into hierarchical chunks
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    chunks = node_parser.get_nodes_from_documents(docs)

    return chunks

def add_nodes(folder_path):
    """Add nodes in folder_path to the index."""
    files_in_folder = os.listdir(folder_path)
    print(f"Embedding {len(files_in_folder)} files...")

    nodes = []
    for filename in tqdm(files_in_folder, desc="Processing files"):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            json_file = load_documents(file_path)
            chunks = json_to_nodes_base_only(json_file)
            nodes.extend(chunks)
    return nodes


if __name__ == "__main__":

    # Define models
    embed_model = OpenAIEmbedding(
        embed_batch_size=1000,
        api_key=api_key,
        model="text-embedding-3-small"
    )

    Settings.embed_model = embed_model

    # Create nodes
    folder_path = "filtered_data/" # path of the dataset
    nodes = add_nodes(folder_path)
    print("Number of nodes:", len(nodes))

    # Create index
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    index.storage_context.persist("DBCleaned")

