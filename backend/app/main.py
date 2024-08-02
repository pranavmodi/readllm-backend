from collections import defaultdict
from flask import Flask, jsonify, request, url_for, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from backend.app.process_book import book_main, lookup_book_summary, lookup_summary, all_summaries, books_collection, generate_file_hash, connect_to_mongodb
from backend.app.book_pipeline_copy import init_book_vectorize, chat_response, explain_the_page
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from PIL import Image
import json
import zipfile
from lxml import etree
import os
import threading
import logging
from flask_socketio import SocketIO, emit
from bson.binary import Binary
from bson import ObjectId
import io
import datetime


app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
mongo_collection, books_collection = connect_to_mongodb()

BOOKS_DIR = 'static/epubs'
THUMBNAILS_DIR = 'static/thumbnails'
JSON_DIR = 'static/jsons'
EMB_DIR = 'static/embeddings'

if not os.path.exists(BOOKS_DIR):
    os.makedirs(BOOKS_DIR)

if not os.path.exists(THUMBNAILS_DIR):
    os.makedirs(THUMBNAILS_DIR)

if not os.path.exists(EMB_DIR):
    os.makedirs(EMB_DIR)

# class BookChat:
#     def __init__(self):
#         self.index = None

#     def reset_index(self):
#         self.index = None

#     def get_index(self, embedding_path):
#         if self.index is None:
#             self.index = create_book_index(embedding_path)
#         return self.index

# book_chat = BookChat()

namespaces = {
    "calibre": "http://calibre.kovidgoyal.net/2009/metadata",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "opf": "http://www.idpf.org/2007/opf",
    "u": "urn:oasis:names:tc:opendocument:xmlns:container",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@app.route('/debug/paths', methods=['GET'])
def debug_paths():
    return jsonify(
        current_working_directory=os.getcwd(),
        static_folder=app.static_folder,
        template_folder=app.template_folder
    )


@app.route('/')
def hello():
    return 'Hello, new beautiful World!'

def clean_book_name(name):
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').replace('-', ' ').split())

def get_epub_cover(epub_file):
    namespaces = {'opf': 'http://www.idpf.org/2007/opf', 'u': 'urn:oasis:names:tc:opendocument:xmlns:container'}
    with zipfile.ZipFile(epub_file, 'r') as z:
        t = etree.fromstring(z.read("META-INF/container.xml"))
        rootfile_elements = t.xpath("/u:container/u:rootfiles/u:rootfile", namespaces=namespaces)
        if not rootfile_elements:
            return None
        rootfile_path = rootfile_elements[0].get("full-path")

        t = etree.fromstring(z.read(rootfile_path))
        cover_meta_elements = t.xpath("//opf:metadata/opf:meta[@name='cover']", namespaces=namespaces)
        if not cover_meta_elements:
            return None
        cover_id = cover_meta_elements[0].get("content")

        cover_item_elements = t.xpath("//opf:manifest/opf:item[@id='" + cover_id + "']", namespaces=namespaces)
        if not cover_item_elements:
            return None
        cover_href = cover_item_elements[0].get("href")

        cover_path = os.path.join(os.path.dirname(rootfile_path), cover_href)
        return z.open(cover_path)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/get-books')
def get_books():
    books = []
    for book in books_collection.find():
        book_data = {
            "name": book['book_name'],
            "filename": book['filename'],
            "epub": url_for('serve_epub', book_id=str(book['_id'])),
            "thumbnail": url_for('serve_thumbnail', book_id=str(book['_id'])) if book.get('cover_image') else None,
            "upload_date": book['upload_date']
        }
        books.append(book_data)

    return jsonify(books)


@app.route('/serve-epub/<book_id>')
def serve_epub(book_id):
    book = books_collection.find_one({'_id': ObjectId(book_id)})
    if book and 'epub_content' in book:
        return send_file(
            io.BytesIO(book['epub_content']),
            mimetype='application/epub+zip',
            as_attachment=True,
            download_name=book['filename']
        )
    return 'Book not found', 404

@app.route('/serve-thumbnail/<book_id>')
def serve_thumbnail(book_id):
    book = books_collection.find_one({'_id': ObjectId(book_id)})
    if book and 'cover_image' in book:
        return send_file(
            io.BytesIO(book['cover_image']),
            mimetype='image/jpeg'
        )
    return 'Thumbnail not found', 404


@app.route('/upload-epub', methods=['POST'])
def upload_epub():
    if 'file' not in request.files:
        return 'No epub file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    book_name = os.path.splitext(filename)[0]

    # Read the file content
    epub_content = file.read()

    # Generate hash
    file_hash = generate_file_hash(epub_content)

    # Check if the book already exists
    existing_book = books_collection.find_one({'file_hash': file_hash})
    if existing_book:
        return jsonify({
            "message": "File already exists", 
            "filename": existing_book['filename'], 
            "id": str(existing_book['_id'])
        })

    # Get the cover image
    cover_image = get_epub_cover(io.BytesIO(epub_content))
    cover_image_binary = None
    if cover_image:
        cover_image_binary = Binary(cover_image.read())

    # Create a document to store in MongoDB
    book_document = {
        'filename': filename,
        'book_name': book_name,
        'epub_content': Binary(epub_content),
        'cover_image': cover_image_binary,
        'upload_date': datetime.datetime.utcnow(),
        'file_hash': file_hash  # Add the hash to the document
    }

    # Insert the document into MongoDB
    result = books_collection.insert_one(book_document)

    return jsonify({
        "message": "File upload successful", 
        "filename": filename, 
        "id": str(result.inserted_id)
    })

# @app.route('/process-epub', methods=['POST'])
# def process_epub():
#     logging.info("Inside process_epub")
#     data = request.get_json()
#     filename = data.get('filename')
#     book_name = data.get('name')

#     if not filename:
#         return 'No filename provided', 400

#     file_path = os.path.join(BOOKS_DIR, filename)
#     logging.info("The books_dir is %s and the filename is %s", BOOKS_DIR, filename)
#     logging.info("The file path is: %s", file_path)

#     if not os.path.exists(file_path):
#         logging.info("The file not found")
#         return 'File not found', 404

#     bname = os.path.splitext(filename)[0]
#     json_path = os.path.join(JSON_DIR, bname + '.json')
#     embeddings_path = os.path.join(EMB_DIR, bname + '.npy')

#     logging.info("Starting a new thread for processing the ePub file and json path is %s", json_path)
#     thread = threading.Thread(target=book_main, args=(file_path, book_name, socketio, json_path, embeddings_path))
#     thread.start()

#     # logging.info("Starting a new thread for embedding the ePub file and embedding path is %s", EMB_DIR)
#     # thread = threading.Thread(target=init_book_vectorize, args=(file_path, book_name, EMB_DIR))
#     # thread.start()
#     # init_book_vectorize(file_path, book_name, output_dir)

#     return jsonify({"message": "Book processing initiated", "filename": filename})

@app.route('/process-epub', methods=['POST'])
def process_epub():
    logging.info("Inside process_epub")
    data = request.get_json()
    book_id = data.get('book_id')
    book_name = data.get('name')

    if not book_id:
        return 'No book ID provided', 400

    book = books_collection.find_one({'_id': ObjectId(book_id)})
    if not book:
        logging.info("The book not found")
        return 'Book not found', 404

    logging.info("Starting a new thread for processing the ePub file")
    thread = threading.Thread(target=book_main, args=(book['epub_content'], book_name, socketio, book_id))
    thread.start()

    return jsonify({"message": "Book processing initiated", "book_id": str(book_id)})

@app.route('/book-summary/<path:book_title>', methods=['GET'])
def book_summary(book_title):
    summary_document = lookup_book_summary(book_title)

    if summary_document:
        return jsonify({
            "status": "success",
            "book_summary": summary_document
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Summary not found for book: " + book_title
        }), 404

@app.route('/chapter-summary/<path:chapter_id>', methods=['GET'])
def get_summary(chapter_id):
    summary_document = lookup_summary(chapter_id)

    if summary_document:
        return jsonify({
            "status": "success",
            "chapter_summary": summary_document
        })
    else:
        return jsonify({
            "status": "pending",
            "message": "Summary is pending for chapter ID: " + chapter_id
        })
    

@app.route('/all-summaries', methods=['POST'])
def get_all_summaries():
    data = request.json
    book_name = data.get('bookName')
    chapter_ids = data.get('chapterIds')

    if not book_name or not chapter_ids:
        return jsonify({
            "status": "error",
            "message": "Missing book name or chapter IDs"
        }), 400

    summaries = all_summaries(chapter_ids, book_name, socketio)

    response_data = {
        "status": "success",
        "summaries": {}
    }

    print('the response data is', response_data)

    for chapter_id, summary in summaries.items():
        if summary:
            response_data["summaries"][chapter_id] = {
                "status": "success",
                "chapter_summary": summary
            }
        else:
            response_data["summaries"][chapter_id] = {
                "status": "pending",
                "message": f"Summary is pending for chapter ID: {chapter_id}"
            }

    return jsonify(response_data)


@app.route('/initialize_book', methods=['POST'])
def initialize_book():
    print('going to initialize book')
    data = request.json
    if not data or 'book_id' not in data:
        return jsonify({"error": "Book ID is required"}), 400

    book_id = data['book_id']
    force_recreate = data.get('force_recreate', False)

    book = books_collection.find_one({'_id': ObjectId(book_id)})
    if not book:
        print('book not found')
        return jsonify({"error": "Book not found"}), 404

    # Start the vectorization process in a separate thread
    thread = threading.Thread(target=run_vectorization, args=(book['epub_content'], book['book_name'], force_recreate, book_id))
    thread.start()

    return jsonify({"message": f"Vectorization process started for {book['book_name']}"}), 202

def run_vectorization(file_path, book_name, force_recreate):
    try:
        init_book_vectorize(book, book_name, EMB_DIR, force_recreate=force_recreate)
        # You could emit a socket event here if you're using SocketIO
        # socketio.emit('vectorization_complete', {'book_name': book_name, 'status': 'success'})
    except Exception as e:
        print(f"Error vectorizing {book_name}: {str(e)}")
        # socketio.emit('vectorization_error', {'book_name': book_name, 'error': str(e)})


def manage_conversation_history(book_name, question, answer, max_history=2):
    history_file = f"{book_name}_chat_history.json"
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append({"question": question, "answer": answer})
    
    # Keep only the last 'max_history' interactions
    history = history[-max_history:]
    
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    return history

@app.route('/chat_with_book', methods=['POST'])
def chat_with_book():
    print('Processing chat request')
    data = request.json
    query = data.get('query')
    book_name = data.get('book_name')
    
    if not query or not book_name:
        return jsonify({"error": "Query and book name must be provided"}), 400

    # Construct file path
    index_path = os.path.join(EMB_DIR, f"{book_name}_faiss.pkl")

    # Check if required file exists
    if not os.path.exists(index_path):
        return jsonify({"error": "Book embeddings not found"}), 404

    try:
        # Load the FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Call chat_response function
        history_file = f"{book_name}_chat_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []

        response = chat_response(query, vectorstore, book_name, history)
        print('the response is', response)
        manage_conversation_history(book_name, query, response)

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error in chat_with_book: {str(e)}")
        print("Full stacktrace:")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request"}), 500


@app.route('/explain-page', methods=['POST'])
def explain_page():
    data = request.json
    book_name = data.get('book_name')
    chapter_name = data.get('chapter_name')
    page_text = data.get('page_text')
    highlighted_text = data.get('highlighted_text')

    print("in explain page")
    print("the chapter name", chapter_name)

    if not book_name or not chapter_name or not page_text:
        return jsonify({
            "status": "error",
            "message": "Missing book name, chapter name, or page text"
        }), 400

    try:
        explanation = explain_the_page(book_name, chapter_name, page_text, highlighted_text)
        
        if isinstance(explanation, dict) and 'error' in explanation:
            return jsonify({
                "status": "error",
                "message": explanation['error']
            }), 500

        return jsonify({
            "status": "success",
            "explanation": explanation
        })

    except Exception as e:
        logging.exception(f"Error in explain_page: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }), 500

if __name__ == '__main__':
    # app.run(debug=True)
    print("Current Working Directory:", os.getcwd())
    print("Static Folder:", app.static_folder)
    print("Template Folder:", app.template_folder)
    socketio.run(app, host='0.0.0.0', port=8000,  debug=True)
