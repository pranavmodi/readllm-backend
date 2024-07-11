import os
from ebooklib import epub
from unstructured.partition.epub import partition_epub
from unstructured.documents.elements import Text, Title, NarrativeText
from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes
from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from openai import OpenAI




global_client = None
api_key = None

def create_client():
    print('creating client in book pipeline')
    global global_client
    if global_client is None:
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        api_key = openai_api_key
        global_client = OpenAI(api_key=openai_api_key)
    return global_client

create_client()

def clean_text(text):
    text = clean_extra_whitespace(text)
    text = replace_unicode_quotes(text)
    return text

def chunk_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def vectorize_chunks(chunks, encoder):
    # Convert chunks to Text elements
    text_elements = [Text(text=chunk) for chunk in chunks]
    
    # Embed the text elements
    embedded_elements = encoder.embed_documents(text_elements)
    
    # Extract embeddings from the embedded elements
    embeddings = [elem.embeddings for elem in embedded_elements]
    
    return np.array(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def init_book_vectorize(file_path, book_name, output_dir, socketio=None, force_recreate=False):
    index_path = os.path.join(output_dir, f"{book_name}_faiss.index")
    chunks_path = os.path.join(output_dir, f"{book_name}_chunks.pkl")

    # Check if embeddings already exist
    if os.path.exists(index_path) and os.path.exists(chunks_path) and not force_recreate:
        print(f"Embeddings for {book_name} already exist. Skipping processing.")
        if socketio:
            socketio.emit('processing_complete', {'book_name': book_name, 'status': 'skipped'})
        return
    

    print("going to create the embeddings for book", book_name)
    print("going to create the embeddings at path", index_path)
    print("going to create the chunks at path", chunks_path)


    

    # Partition EPUB
    elements = partition_epub(file_path)
    
    # Clean and chunk text
    # text_content = " ".join([str(elem) for elem in elements if elem.text_as_html])

    text_content = " ".join([
    str(elem) for elem in elements 
    if isinstance(elem, (Text, Title, NarrativeText))
    ])

    cleaned_text = clean_text(text_content)
    chunks = chunk_text(cleaned_text)
    
    # Initialize OpenAI encoder
    config = OpenAIEmbeddingConfig(api_key=api_key)
    encoder = OpenAIEmbeddingEncoder(config=config)
    
    # Vectorize chunks
    embeddings = vectorize_chunks(chunks, encoder)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    
    # Save chunks for later reference
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    
    if socketio:
        socketio.emit('processing_complete', {'book_name': book_name, 'status': 'created'})
    
    print(f"Completed processing {book_name}")

# Usage
# file_path = "path/to/your/book.epub"
# book_name = "Example Book"
# api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key
# output_dir = "path/to/output/directory"

# init_book_vectorize(file_path, book_name, output_dir)