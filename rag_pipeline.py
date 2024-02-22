from PyPDF2 import PdfReader
import openai
from openai import OpenAI
import pandas as pd
import numpy as np
from getpass import getpass
from pinecone import Pinecone
import pinecone
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st


def extract_text_from_pdf(file_path):
    """
    Extracts the text from a PDF file.

    Args:
    file_path: The file path to the PDF from which text is to be extracted.

    Return:
    str: A string containing all extracted text from the PDF file.
    """
    # Initialize an empty string to hold the extracted text
    text = ''
    # Open the PDF file in binary read mode
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        # Iterate through each page in the PDF
        for page in pdf_reader.pages:       
            text += page.extract_text()
    return text

data = extract_text_from_pdf('/Users/shuai/Desktop/astr.pdf')



def get_chunks(data, chunk_size, overlap_size):
    """
    Splits the data into chunks 

    Args:
    data: Extracted text
    chunk_size: The size of each chunk.
    overlap_size: The number of elements to be overlapped between chunks.
    
    Return:
    A list of chunks.
    """
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be smaller than chunk size")
    
    chunks = []
    for i in range(0, len(data), chunk_size - overlap_size):
        chunks.append(data[i:i + chunk_size])
        # Ensure that we don't go past the end of the list
        if i + chunk_size >= len(data):
            break
    return chunks

chunk_data = get_chunks(data, 1000, 200)

client = openai.OpenAI(api_key = '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')

def get_embeddings(chunk_data):
    """
    Get the embedding vectors for the chunk data

    Arg:
    - chunk_data: a list of chunk data

    Return:
    - Embedded vectors

    """
    
    client = OpenAI(api_key = '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')

    response = client.embeddings.create(
        input=chunk_data,
        model="text-embedding-3-small"
        )

    vectors_list = [item.embedding for item in response.data]
    return vectors_list

vectors_list = get_embeddings(chunk_data)


# Initialize the client and index(name of vector database)
pc = Pinecone(api_key='7658c7bf-1ee9-4e8d-9a5e-bc100843b51f')
index = pc.Index("assignment2")

# Store vectors in vector database
def vector_store(vectors_list):
    # Iterate over the vectors_list
    for i in range(len(vectors_list)):
        index.upsert(
            vectors=[
                {
                    'id': f'vec_{i}',
                    'values': vectors_list[i],
                    'metadata': {"text":chunk_data[i]}
                }
            ],
        )
    
vector_store(vectors_list)

query = 'What feature do only Jupiter, Saturn, Neptune and Uranus have in common'
query_embedding = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)
query_vector = [item.embedding for item in query_embedding.data]


def retrieve_embedding(index, num_embed):
    """
    Convert the information of vectors in the database into a panda dataframe
    
    Args:
    - index: Name of vector database(already set up)
    - num_embed: total number of vectors in the vector databse

    Return:
    - a dataframe which contains the embedded vectors and corresponding text
    """
    # Initialize a dictionary to store embedding data
    embedding_data = {"id":[], "values":[], "text":[]}
    
    # Fetch the embeddings 
    embedding = index.fetch([f'vec_{i}' for i in range(num_embed)])
    
    for i in range(num_embed):
        embedding_data["id"].append(i)
        idx = f"vec_{i}"
        embedding_data["text"].append(embedding['vectors'][idx]['metadata']['text'])
        embedding_data["values"].append(embedding['vectors'][idx]['values'])
        
    return pd.DataFrame(embedding_data)


embedding_data = retrieve_embedding(index,len(vectors_list))

def semantic_search(query_vector, db_embeddings):
    """
    Find the top three vectors which have the highest comsine similarity with the query vector

    Args:
    - query_vector: embedded vector of user query
    - db_embeddings: embedded vectors from vector database

    Return:
    - The indices of top three most similar vectors with the query vector
    """
    
    similarities = cosine_similarity(query_vector, db_embeddings)[0]
    # Get the indices of the top three similarity scores
    top_3_indices = np.argsort(similarities)[-3:][::-1]  # This sorts and then reverses to get top 3
    # Retrieve the top three most similar chunks and their similarity scores
    
    return top_3_indices

top_3_indices = semantic_search(query_vector, vectors_list)


def get_text(embedding_data, top_3_indices):
    """
    Extracts text corresponding to the given top vectors from embedding data.

    Args:
    - embedding_data (DataFrame): DataFrame containing columns 'id', 'values', and 'text'.
    - top_vectors (list): List of indices for which corresponding text needs to be extracted.

    Returns:
    - combined_text (str): Combined text corresponding to the top vectors.
    """
   # Extract text from selected rows
    selected_texts = embedding_data.loc[top_3_indices, 'text'].tolist()

    # Combine the selected texts into a single string
    combined_text = ' '.join(selected_texts)

    return combined_text

context = get_text(embedding_data, top_3_indices)

def response_wo_rag(user_input):
    """
    Generate response withou using rag

    Args:
    - questions given by users

    Returns:
    - answer without using rag  
    """

    response_wo_rag = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": user_input}
    ]
    )
    return response_wo_rag.choices[0].message.content

def response_rag(user_input):
    """
    Generate response using rag

    Args:
    - questions given by users

    Returns:
    - answer using rag  
    """
    
    response_rag = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "assistant", "content": context},
      {"role": "user", "content": user_input}
    ]
    )
    return response_rag.choices[0].message.content


# Deploy locally using Streamlit
st.title("Response Generator")

# Create a form for user input
with st.form("input_form"):
    user_input = st.text_area("Enter your message:")
    submitted = st.form_submit_button("Submit")

# Check if the question is sumbitedd by users
if submitted:
    st.text("Response with RAG:")
    st.write(response_rag(user_input))
    st.text("Response without RAG:")
    st.write(response_wo_rag(user_input))