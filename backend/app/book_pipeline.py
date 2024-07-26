import os
from ebooklib import epub
from unstructured.partition.epub import partition_epub
from unstructured.documents.elements import Text, Title, NarrativeText
from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes
from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
from backend.app.process_book import lookup_book_summary, lookup_summary
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
import openai
from openai import OpenAI
import re

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

def explain_the_page(book_name: str, chapter_name: str, page_text: str):
    # Initialize OpenAI API (make sure to set your API key in the environment variables)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Fetch book summary
    book_summary = lookup_book_summary(book_name)
    if not book_summary:
        return {"error": "Book summary not found"}

    print("motherfucking here")
    # Fetch chapter summary
    chapter_id = f"{book_name}_Chapter_{chapter_name}"
    chapter_summary = lookup_summary(chapter_id)
    # print('the chapter summary', chapter_summary)
    if not chapter_summary:
        return {"error": "Chapter summary not found"}

    # Prepare the prompt for OpenAI
    prompt = f"""
    Book: {book_name}
    Book Summary: {book_summary}
    
    Chapter: {chapter_name}
    Chapter Summary: {chapter_summary['summary']}
    
    Current Page Text:
    {page_text}
    
    Please provide a simplified explanation of the page text, considering the context of the book and chapter summaries. 
    Specifically explain what is written on this page, by quoting sentences, and not the summaries:
    Explain like to a bright teenager, without any complex jargon
    """

    try:
        # Call OpenAI API for explanation
        print('just before the request')
        response = global_client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains complex text in simpler terms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract the explanation from the API response
        # print('the response', response.choices[0])
        # explanation = response.choices[0].message['content'].strip()
        explanation = response.choices[0].message.content

        print('going to return the explanation', explanation)
        return {
            "book_name": book_name,
            "chapter_name": chapter_name,
            "explanation": explanation
        }

    except Exception as e:
        print(e)
        return {"error": f"Failed to generate explanation: {str(e)}"}

def chat_response(user_query, book_index, text_chunks, top_k=5):

    # Set up the embedding encoder
    config = OpenAIEmbeddingConfig(api_key=api_key)
    encoder = OpenAIEmbeddingEncoder(config=config)

    # Vectorize the user query
    query_element = Text(text=user_query)
    query_embedding = encoder.embed_documents([query_element])[0].embeddings

    # Perform the similarity search
    D, I = book_index.search(np.array([query_embedding]), top_k)

    # Retrieve the most relevant text chunks
    relevant_chunks = [text_chunks[i] for i in I[0]]
    print('the relevant chunks are ', relevant_chunks)

    # Prepare the context for the ChatGPT query
    context = "\n".join(relevant_chunks)

    # Prepare the messages for the ChatGPT API
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context from a book. If the answer cannot be found in the context, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]

    # Make the API call to ChatGPT
    try:
        response = global_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract and return the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()        
        return assistant_reply

    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return "I'm sorry, but I encountered an error while processing your request."


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