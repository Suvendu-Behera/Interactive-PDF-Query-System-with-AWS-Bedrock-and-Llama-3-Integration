import os
import boto3
import streamlit as st
import signal
from time import sleep
import webbrowser

# Using Titan Embeddings Model to generate Embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data Ingestion: Load and split PDF documents into manageable chunks
def data_ingestion():
    # Load all PDFs in the 'data' directory
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split documents into smaller chunks (10,000 characters with 1,000 character overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding: Convert text documents into vector embeddings and store them using FAISS
def get_vector_store(docs):
    # Create vector store using FAISS and Bedrock embeddings
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")  # Save the FAISS index locally

# Retrieve the Llama 3 model from Bedrock
def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Define the prompt template for generating responses
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Generate a response using the LLM and the vector store
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat PDF")  # Set the title of the web page
    
    st.header("Chat with PDF using AWS Bedrock💁")  # Display the header in the app

    # Input box for the user to ask a question
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:  # Sidebar for additional controls
        st.title("Update or Create Vector Store:")
        
        # Button to update vector store
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()  # Load and split PDF documents
                get_vector_store(docs)  # Create and save vector embeddings
                st.success("Done")

    # Button to get output from Llama 3
    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            # Load FAISS index and get LLM response
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    # Button to close the application
    if st.button("Close"):
        st.write("Closing the application...")

        # Close the browser tab (attempts to close any open browser window with the Streamlit app)
        browser = webbrowser.get()
        browser.open("about:blank", new=0)  # Redirect to a blank page
        sleep(2)  # Give time for the browser to process

        # Get the process ID and terminate the application
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)

if __name__ == "__main__":
    main()
