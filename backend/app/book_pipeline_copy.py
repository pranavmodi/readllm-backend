import os
from dotenv import load_dotenv
import json
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from backend.app.process_book import lookup_book_summary, lookup_summary


load_dotenv()

def create_client():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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


def chat_response(query, vectorstore, book_name, history):
    try:

        # Create a memory object with the existing chat history
        print('history is ', history)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )

        for interaction in history:
            memory.save_context({"question": interaction["question"]}, {"output": interaction["answer"]})

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
            # llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo"),
            llm=ChatOpenAI(temperature=0.7, model_name="gpt-4o"),
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=False,
            get_chat_history=lambda h: h,
            output_key='answer',
        )

        # Get the response
        result = chain({"question": f"For the book '{book_name}': {query}", "book_name": book_name})
        
        return result['answer']
    
    except Exception as e:
        
        # Returning an error message:
        return f"An error occurred while processing your request: {str(e)}"
 

def explain_the_page(book_name: str, chapter_name: str, page_text: str, highlighted_text: str = ''):
    # Fetch book and chapter summaries (assuming these functions exist)
    book_summary = lookup_book_summary(book_name)
    chapter_id = f"{book_name}_Chapter_{chapter_name}"
    chapter_summary = lookup_summary(chapter_id)

    if not book_summary:
        return {"error": "Book summary not found"}
    if not chapter_summary:
        return {"error": "Chapter summary not found"}

    # Create a prompt template based on whether highlighted text is available
    print('highlighted text is', highlighted_text)

    if highlighted_text:
        print('going to use highlighted text')
        prompt_template = PromptTemplate(
            input_variables=["book_name", "book_summary", "chapter_name", "chapter_summary", "page_text", "highlighted_text"],
            template="""
            Book: {book_name}
            Book Summary: {book_summary}
            
            Chapter: {chapter_name}
            Chapter Summary: {chapter_summary}
            
            Current Page Text:
            {page_text}
            
            Highlighted Text:
            {highlighted_text}
            
            Please provide a simplified explanation of the highlighted text, considering the context of the book, chapter summaries, and the current page text.
            Focus specifically on explaining the highlighted portion:
            "{highlighted_text}"
            Explain it like you would to a bright teenager, without using complex jargon. 
            Relate the explanation to the broader context of the page and chapter if relevant.
            """
        )
    else:
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

    # Prepare the input dictionary
    input_dict = {
        "book_name": book_name,
        "book_summary": book_summary,
        "chapter_name": chapter_name,
        "chapter_summary": chapter_summary['summary'],
        "page_text": page_text,
    }
    
    # Add highlighted_text to input if available
    if highlighted_text:
        input_dict["highlighted_text"] = highlighted_text

    # Run the chain
    result = chain.invoke(input_dict)

    return {
        "book_name": book_name,
        "chapter_name": chapter_name,
        "explanation": result
    }