from flask import Flask, jsonify, request, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
# from process_book import book_main, lookup_summary, lookup_book_summary
# from readai import chat_response, create_book_index
from backend.app.process_book import book_main, lookup_book_summary, lookup_summary, all_summaries
# from .process_book import book_main
from PIL import Image
from io import BytesIO
import zipfile
from lxml import etree
import os
import threading
import logging
from flask_socketio import SocketIO, emit
# import json

# app = Flask(__name__, static_folder='static')
app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

BOOKS_DIR = 'static/epubs'
THUMBNAILS_DIR = 'static/thumbnails'
JSON_DIR = 'static/jsons'
EMB_DIR = 'static/embeddings'

if not os.path.exists(BOOKS_DIR):
    os.makedirs(BOOKS_DIR)

if not os.path.exists(THUMBNAILS_DIR):
    os.makedirs(THUMBNAILS_DIR)

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

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

def get_epub_cover(epub_path):
    namespaces = {'opf': 'http://www.idpf.org/2007/opf', 'u': 'urn:oasis:names:tc:opendocument:xmlns:container'}
    with zipfile.ZipFile(epub_path, 'r') as z:
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
    book_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith('.epub')]

    books = []
    for book_file in book_files:
        book_name = os.path.splitext(book_file)[0]
        epub_path = os.path.join(BOOKS_DIR, book_file)
        thumbnail_path = os.path.join(THUMBNAILS_DIR, book_name + '.jpg')

        if os.path.exists(thumbnail_path):
            thumbnail_url = url_for('static', filename=os.path.join('thumbnails', book_name + '.jpg'))
        else:
            thumbnail_url = None

        books.append({
            "name": clean_book_name(book_name),
            "filename": book_file,
            "epub": url_for('static', filename=os.path.join('epubs', book_file)),
            "thumbnail": thumbnail_url
        })

    return jsonify(books)

@app.route('/upload-epub', methods=['POST'])
def upload_epub():
    logging.info("Inside new upload_epub")
    if 'file' not in request.files:
        return 'No epub file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(BOOKS_DIR, filename)
    file.save(file_path)

    try:
        cover_image = get_epub_cover(file_path)
        book_name = os.path.splitext(os.path.basename(file_path))[0]
        if cover_image is None:
            raise Exception("Cover image not found")
        cover_image_path = os.path.join(THUMBNAILS_DIR, book_name + '.jpg')
        image = Image.open(cover_image)
        image.save(cover_image_path, 'JPEG')
        logging.info("Cover image saved for book: %s", book_name)
    except Exception as e:
        logging.warning("No cover image found or error in processing for book: %s. Error: %s", book_name, str(e))

    return jsonify({"message": "File upload successful", "filename": filename})

@app.route('/process-epub', methods=['POST'])
def process_epub():
    logging.info("Inside process_epub")
    data = request.get_json()
    filename = data.get('filename')
    book_name = data.get('name')

    if not filename:
        return 'No filename provided', 400

    file_path = os.path.join(BOOKS_DIR, filename)
    logging.info("The books_dir is %s and the filename is %s", BOOKS_DIR, filename)
    logging.info("The file path is: %s", file_path)

    if not os.path.exists(file_path):
        logging.info("The file not found")
        return 'File not found', 404

    bname = os.path.splitext(filename)[0]
    json_path = os.path.join(JSON_DIR, bname + '.json')
    embeddings_path = os.path.join(EMB_DIR, bname + '.npy')

    logging.info("Starting a new thread for processing the ePub file and json path is %s", json_path)
    thread = threading.Thread(target=book_main, args=(file_path, book_name, socketio, json_path, embeddings_path))
    thread.start()

    return jsonify({"message": "Book processing initiated", "filename": filename})

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

# @app.route('/chat_with_book', methods=['POST'])
# def chat_with_book():
#     data = request.json
#     query = data.get('query')
#     embedding_name = data.get('npy_path')
#     embedding_path = os.path.join(EMB_DIR, embedding_name)
#     logging.info("the embedding file path is %s", embedding_path)
#     json_path = os.path.join(JSON_DIR, data.get('json_name'))
#     logging.info("the json file path is %s", json_path)

#     with open(json_path, 'r', encoding='utf-8') as f:
#         text_chunks = json.load(f)

#     if not query:
#         return jsonify({"error": "No query provided"}), 400

#     index = book_chat.get_index(embedding_path)
#     response = chat_response(query, index, text_chunks, top_k=10)

#     return jsonify({"response": response}), 200

if __name__ == '__main__':
    # app.run(debug=True)
    print("Current Working Directory:", os.getcwd())
    print("Static Folder:", app.static_folder)
    print("Template Folder:", app.template_folder)
    socketio.run(app, host='0.0.0.0', port=8000,  debug=True)
