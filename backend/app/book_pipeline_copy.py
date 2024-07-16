import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from backend.app.process_book import lookup_book_summary, lookup_summary


load_dotenv()

def create_client():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def init_book_vectorize(file_path, book_name, output_dir, socketio=None, force_recreate=False):
    index_path = os.path.join(output_dir, f"{book_name}_faiss.pkl")
    
    # Check if embeddings already exist
    if os.path.exists(index_path) and not force_recreate:
        print(f"Embeddings for {book_name} already exist. Skipping processing.")
        if socketio:
            socketio.emit('processing_complete', {'book_name': book_name, 'status': 'skipped'})
        return

    print(f"Creating embeddings for book: {book_name}")
    print(f"Embeddings will be saved at: {index_path}")

    # Load EPUB
    loader = UnstructuredEPubLoader(file_path)
    documents = loader.load()

    # Clean and chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Create and save FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    # if socketio:
    #     socketio.emit('processing_complete', {'book_name': book_name, 'status': 'created'})

    print(f"Completed processing {book_name}")

import logging
import traceback
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def chat_response(query, vectorstore, book_name):
    try:
        # Create a memory object with a custom input key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",  # Specify the input key
            return_messages=True
        )

        # Create a prompt template
        prompt_template = """You are an AI assistant helping with questions about the book "{book_name}".
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        AI Assistant:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "book_name"]
        )

        # Create the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=False,
            get_chat_history=lambda h: h,  # Return full chat history
            output_key='answer',  # Add this line
            verbose=True
        )

        # Get the response
        result = chain({"question": f"For the book '{book_name}': {query}", "book_name": book_name})
        # result = chain({"question": f"For the book '{book_name}': {query}", "book_name": '{book_name}'})
        
        return result['answer']

    except Exception as e:
        # Log the full stack trace
        logger.error(f"Error in chat_response: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Returning an error message:
        return f"An error occurred while processing your request: {str(e)}"
 

def explain_the_page(book_name: str, chapter_name: str, page_text: str):
    # Fetch book and chapter summaries (assuming these functions exist)
    book_summary = lookup_book_summary(book_name)
    chapter_id = f"{book_name}_Chapter_{chapter_name}"
    chapter_summary = lookup_summary(chapter_id)

    if not book_summary:
        return {"error": "Book summary not found"}
    if not chapter_summary:
        return {"error": "Chapter summary not found"}

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["book_name", "book_summary", "chapter_name", "chapter_summary", "page_text"],
        template="""
        Book: {book_name}
        Book Summary: {book_summary}
        
        Chapter: {chapter_name}
        Chapter Summary: {chapter_summary}
        
        Current Page Text:
        {page_text}
        
        Please provide a simplified explanation of the page text, considering the context of the book and chapter summaries. 
        Specifically explain what is written on this page, by quoting sentences, and not the summaries:
        Explain like to a bright teenager, without any complex jargon
        """
    )

    # Create the chain
    llm = OpenAI(temperature=0.7)
    chain = prompt_template | llm

    # Run the chain
    result = chain.invoke({
        "book_name": book_name,
        "book_summary": book_summary,
        "chapter_name": chapter_name,
        "chapter_summary": chapter_summary['summary'],
        "page_text": page_text
    })

    return {
        "book_name": book_name,
        "chapter_name": chapter_name,
        "explanation": result
    }