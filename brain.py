import os
from typing import List
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# Load environment variables
load_dotenv()

def text_to_docs(text: str, filename: str) -> List[Document]:
    doc = Document(page_content=text)
    doc.metadata["filename"] = filename  # Add filename to metadata
    return [doc]

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index

def get_index_for_text(text_inputs: List[str], openai_api_key: str, filename: str = "input.txt"):
    documents = []
    for text_input in text_inputs:
        documents.extend(text_to_docs(text_input, filename))
    index = docs_to_index(documents, openai_api_key)
    return index

