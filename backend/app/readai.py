import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import tiktoken
import numpy as np
# import faiss
import logging
from transformers import AutoTokenizer, AutoModel
import subprocess
import re
import json
import logging

global_client = None

def create_client():
    global global_client
    if global_client is None:
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        print('the key', openai_api_key)
        global_client = OpenAI(api_key=openai_api_key)
    return global_client


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_into_chunks(chapter_text, max_token_length=3000, overlap_length=500):
    """
    Splits a text into chunks, each having a maximum of max_token_length tokens.
    Each chunk starts and ends at the end of a sentence and there is overlap between chunks.
    This uses a rough approximation of 4 characters per token.
    """
    avg_chars_per_token = 4
    max_chunk_length = max_token_length * avg_chars_per_token
    overlap_chars = overlap_length * avg_chars_per_token
    chunks = []

    while chapter_text:
        if len(chapter_text) <= max_chunk_length:
            chunks.append(chapter_text)
            break

        # Find the end of a sentence near max_chunk_length
        end = chapter_text.rfind('.', 0, max_chunk_length) + 1

        # If a sentence end wasn't found, take the whole chunk
        if end == 0:
            end = max_chunk_length

        chunk = chapter_text[:end]
        chunks.append(chunk)

        # Find the start of the next sentence for the next chunk
        next_sentence_start = chapter_text.find('. ', end - 1) + 2
        if next_sentence_start < 2:
            next_sentence_start = end

        # Start next chunk with some overlap, but from the start of a sentence
        overlap_start = max(end, next_sentence_start - overlap_chars)
        chapter_text = chapter_text[overlap_start:]

    return chunks

def summarize_chunk(chunk, client):
    system_prompt = ("You are an AI assistant skilled in summarizing book chapters. "
                     "Please provide a concise summary of this text chunk, focusing on key points and main ideas.")

    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk}
      ]
    )

    return completion.choices[0].message.content



def consolidate_summaries(summaries, client):
    system_prompt = ("You are an AI assistant. You have received summaries of a book chapter. "
                     "Combine these into a single coherent summary. Return the result as a JSON object. "
                     "The JSON object must have exactly these three keys: 'title', 'summary', and 'is_main_content'. "
                     "Ensure that 'is_main_content' is a boolean value (true or false, not 'Yes' or 'No'). "
                     "Use double quotes for all keys and string values. "
                     "Do not include any text before or after the JSON object. "
                     "Here's an example of the expected format: "
                     '{"title": "Chapter 1", "summary": "This chapter introduces...", "is_main_content": true}')

    combined_summaries = " ".join(summaries) if isinstance(summaries, list) else summaries

    for attempt in range(1, 4):  # Try up to 3 times
        logging.debug(f"Attempt {attempt} to get valid JSON")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_summaries}
            ]
        )

        response = completion.choices[0].message.content
        logging.debug(f"Raw response from model:\n{response}")

        try:
            unified_summary = json.loads(response)
            if all(key in unified_summary for key in ["title", "summary", "is_main_content"]):
                logging.info("Successfully parsed valid JSON")
                return unified_summary
            else:
                logging.warning("JSON parsed but missing required keys")
                print('failed to get all the keys')
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {str(e)}")
            print('failted to parse json')

        # If we're here, the response wasn't valid. Ask the model to try again.
        combined_summaries = (f"The previous response was not valid JSON. Please try again, ensuring that you return "
                              f"a valid JSON object with the keys 'title', 'summary', and 'is_main_content'. "
                              f"Here was your previous attempt:\n\n{response}")

    logging.error("Failed to get a valid JSON response after 3 attempts.")
    return None


def summarize_book_chapter(chapter_text):
    client = create_client()
    chunks = split_into_chunks(chapter_text)
    chunk_summaries = [summarize_chunk(chunk, client) for chunk in chunks]
    consolidated_summary = consolidate_summaries(chunk_summaries, client)
    return consolidated_summary


def summarize_summaries(chapter_summaries, client=None):
    print("Inside summarize_summaries")
    client = create_client()  # Initialize client

    # Check if the token count exceeds the limit
    if num_tokens_from_string(chapter_summaries, 'r50k_base') > 4096:
        chunks = split_into_chunks(chapter_summaries)
        chunk_summaries = [summarize_chunk(chunk, client) for chunk in chunks]

        combined_summaries = " ".join(chunk_summaries)
        if num_tokens_from_string(combined_summaries, 'r50k_base') > 4096:
            return summarize_summaries(combined_summaries, client)  # Recursive call with return
        else:
            return summarize_chunk(combined_summaries, client)  # Summarize the combined summaries
    else:
        system_prompt = ("You are a book reader, skilled in reading chapters and summarizing them. "
                         "You have read several summaries of different parts of a book. "
                         "Please provide a concise, unified summary that captures the overall summary of the entire book.")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chapter_summaries}
            ]
        )
        return completion.choices[0].message.content

# Example usage (assuming necessary functions and client are defined)
# summary = summarize_summaries(your_chapter_summaries)




# Function to summarize a book chapter
# def summarize_book_chapter(chapter_text):
#     def create_client():
#       load_dotenv()  # Load environment variables from .env file
#       openai_api_key = os.getenv('OPENAI_API_KEY1')  # Retrieve the OpenAI API key
#       client = OpenAI(api_key=openai_api_key)  # Initialize OpenAI client
#       return client

#     client = create_client()
#     system_prompt = ("You are a book reader, skilled in reading chapters and summarizing them. "
#              "Create a response with 2 sections: 1. Summary of the chapter. 2. Key Takeaways from the Chapter "
#              "If the chapter text is empty, return empty, just wait for the first chapter text.")

#     # Generate the summary for the current chapter
#     completion = client.chat.completions.create(
#       model="gpt-4o-mini",
#       messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": chapter_text}
#       ]
#     )

#     # Extract the generated summary from the response
#     cc_message = completion.choices[0].message
#     cumulative_book_summary = cc_message.content

#     return cumulative_book_summary

# def clean_json_string(json_string):
#     # Find the first { and the last }
#     start = json_string.find('{')
#     end = json_string.rfind('}') + 1
#     if start == -1 or end == 0:
#         return None
#     json_string = json_string[start:end]
    
#     # Replace 'Yes' with true and 'No' with false
#     json_string = re.sub(r'"Yes"', 'true', json_string)
#     json_string = re.sub(r'"No"', 'false', json_string)
    
#     # Replace single quotes with double quotes
#     json_string = json_string.replace("'", '"')
    
#     return json_string


# logging.basicConfig(level=logging.DEBUG)

