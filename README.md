# Project Overview
ChatGPT is one of the most popular chatting tools nowadays, but it performs badly on unseen data. This project trains GPT 3.5-turbo with unseen data by using Retrieval Augmented Generation from scartch (without using langchain) to enhance the model's performance on the unseen data.

# Pipeline
- extract text from pdf files
- chunk the text
- get embeddings for the chunks using OpenAI's embedding model
- store the embedded vectors in vector database (Pinecone)
- extract the vectors and perform a retrieval using semantic search (use cosine similarity as metric)
- inset relevant context into LLM model
- create a user interface using Streamlit to show both results of using RAG and without using RAG

  # Conclusion
  The model using RAG performs better compared to the model without using RAG 
