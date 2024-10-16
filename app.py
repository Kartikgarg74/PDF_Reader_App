import fitz  # PyMuPDF for handling PDFs
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import cohere
import streamlit as st

# Cohere API key
cohere_client = cohere.Client('qFR72UQsWpnuLJW4cU7bP6R10X4sBd4lZNszGTJN')

# Load the sentence transformer model for embedding
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to create FAISS index from document embeddings
def create_faiss_index(documents):
    embeddings = model.encode(documents)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    return index, embeddings

# Function to search the FAISS index and retrieve relevant documents
def search_faiss_index(query, index, documents, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# Function to generate answer using Cohere's API
def generate_answer(query, context):
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100
    )
    return response.generations[0].text.strip()

# Streamlit app code
def main():
    st.title("Interactive QA Bot with PDF Upload")

    # Step 1: File uploader for PDFs
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # Extract text from the PDF
        st.write("Extracting text from the uploaded PDF...")
        document_text = extract_text_from_pdf(uploaded_file)
        documents = document_text.split("\n\n")  # Split the document into paragraphs
        st.write(f"Document processed. Found {len(documents)} sections.")

        # Step 2: Create FAISS index
        st.write("Creating document embeddings and FAISS index...")
        index, _ = create_faiss_index(documents)
        st.success("Document indexed successfully!")

        # Step 3: Input for user's question
        query = st.text_input("Ask a question based on the document:")
        
        if query:
            st.write("Retrieving relevant context from the document...")
            retrieved_docs = search_faiss_index(query, index, documents)
            context = " ".join(retrieved_docs)

            # Step 4: Show retrieved document sections
            st.subheader("Retrieved Document Sections:")
            for doc in retrieved_docs:
                st.write(doc)

            # Step 5: Generate answer using Cohere API
            st.write("Generating answer...")
            answer = generate_answer(query, context)
            st.subheader("Answer:")
            st.write(answer)

if __name__ == '__main__':
    main()
