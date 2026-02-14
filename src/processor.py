import os
import shutil
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.data_directory = "./data"
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the Chroma vector store. Loads existing if available."""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        else:
            pass

    def ingest_data(self):
        """Loads PDFs/TXTs from ./data, splits, and stores in ChromaDB."""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created {self.data_directory} directory. Please add files.")
            return

        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())

        # Load TXTs
        txt_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
        
        # Load Markdown 
        md_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.md",
            loader_cls=TextLoader
        )
        documents.extend(md_loader.load())

        if not documents:
            print("No documents found in ./data")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        # Store in Chroma
        if splits:
            if self.vector_store:
                self.vector_store.add_documents(splits)
            else:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory
                )
            
            print(f"Ingested {len(splits)} chunks from {len(documents)} documents.")

    def query(self, query_text: str) -> List[str]:
        """Returns the top 3 most relevant snippets for the query."""
        if not self.vector_store and os.path.exists(self.persist_directory):
             self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        
        if not self.vector_store:
            return ["No index found. Please ingest data first."]

        results = self.vector_store.similarity_search(query_text, k=3)
        return [doc.page_content for doc in results]

if __name__ == "__main__":
    # Simple test
    engine = RAGEngine()
    engine.ingest_data()
