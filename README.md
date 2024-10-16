# Interactive PDF QA Bot

This repository contains a two-step project that builds an interactive question-answering (QA) bot for PDF documents. The bot allows users to upload PDFs, ask questions about the content, and receive answers generated using a combination of FAISS (for document search) and Cohere's API (for text generation).

## Features
- **PDF Text Extraction**: Automatically extracts text from uploaded PDF documents.
- **Document Embedding**: Converts document sections into embeddings using Sentence Transformers for semantic search.
- **Efficient Search**: FAISS is used to index document embeddings and retrieve relevant sections based on user queries.
- **AI-Powered Answers**: Cohere's language model generates answers based on the retrieved sections.

## Setup

### Step 1: Install Required Packages

Before starting, ensure you have all the necessary dependencies installed. Run the following pip command:

```bash
pip install sentence-transformers faiss-cpu pymupdf cohere streamlit
```

### Step 2: Running the Application

1. **Step 1 – Prepare Embeddings and Index**  
   Open and run the `kwork.ipynb` Jupyter notebook. This step creates and saves document embeddings using Sentence Transformers and indexes them using FAISS.

2. **Step 2 – Interactive QA System**  
   After generating the FAISS index, you can run the Streamlit-based interactive QA system by executing:

   ```bash
   streamlit run app.py
   ```

## How It Works

### 1. **PDF Upload**
   - The system allows users to upload a PDF document. The text from the PDF is extracted and split into paragraphs or sections for processing.

### 2. **Document Embedding & Indexing**
   - Sentence embeddings are generated for each paragraph using a pre-trained Sentence Transformers model (e.g., `all-MiniLM-L6-v2`).
   - These embeddings are then indexed using FAISS for fast retrieval of relevant sections when a query is made.

### 3. **Ask a Question**
   - Users can input a query about the uploaded document. FAISS will search the indexed embeddings and return the most relevant document sections.

### 4. **Answer Generation**
   - Based on the retrieved sections, Cohere's language model generates a concise answer to the user's question.

## File Overview

- **`kwork.ipynb`**: Jupyter notebook that handles the creation of embeddings and FAISS index for document sections.
- **`app.py`**: Streamlit-based interactive application where users can upload PDFs, ask questions, and receive AI-generated answers.

## Example Workflow

1. **Prepare the Data**:  
   Run `kwork.ipynb` to process the document and generate the FAISS index.

2. **Run the Application**:  
   Start the Streamlit app using `app.py`, upload a PDF, and ask questions about its content.

## Requirements

- Python 3.7+
- Libraries:  
   - `sentence-transformers`
   - `faiss-cpu`
   - `pymupdf`
   - `cohere`
   - `streamlit`

## License
This project is licensed under the MIT License.
