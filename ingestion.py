import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
import argparse

# Load environment variables from .env file
load_dotenv()

class ingestion():
    def __init__(self, 
                ingest_directory: str = "",
                chroma_db_path: str = "./chroma_db",
                chroma_collection_name: str = "test_collection",
                embedding_model: str = "mxbai-embed-large:latest",
                overwrite_existing: bool = False
                ):
        # Set up environment variables
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        # Initialize embeddings and vector store
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
        )

        self.vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_db_path,
        )
        self.ingest_directory = ingest_directory
        self.overwrite_existing = overwrite_existing

    def process_files(self, dir_path: str = None):
        """
        Ingest a file or all files in the ingest directory.
        If file_path is provided, it will ingest that specific file.
        If not, it will ingest all files in the ingest_directory.
        """
        if dir_path:
            if dir_path.endswith(".txt"):
                self._ingest_file(dir_path)
        else:
            for root, _, files in os.walk(self.ingest_directory):
                for file in files:
                    if file.endswith(".txt"):
                        full_path = os.path.join(root, file)
                        self._ingest_file(full_path)
                        
    def _ingest_file(self, file_path: str):
        """
        Ingest a single file, splitting it into chunks and adding to the vector store.
        """
        # Load and split the document
        loader = TextLoader(
            file_path=file_path,
            autodetect_encoding=True
        )

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        all_splits = text_splitter.split_documents(docs)

        # Indexing the chunks into the vector store
        if self.vector_store:
            # Checking if Document is already indexed
            existing_docs = self.vector_store.get(where={"source": file_path}).get("ids", [])
            print(f"Found {len(existing_docs)} existing chunks for {file_path} in the vector store.")
            if existing_docs and not self.overwrite_existing:
                print(f"Document {file_path} is already indexed.")
            elif existing_docs and self.overwrite_existing:
                print(f"Deleting existing chunks from {file_path} in the vector store...")
                _ = self.vector_store.delete(where={"source": file_path})
                
                print(f"Adding new chunks from {file_path} to the vector store...")
                _ = self.vector_store.add_documents(documents=all_splits)
            else:
                print(f"Adding new chunks from {file_path} to the vector store...")
                _ = self.vector_store.add_documents(documents=all_splits)

def main():
    parser = argparse.ArgumentParser(description="Ingest files into Chroma vector store.")
    parser.add_argument("--ingest_directory", "-i", type=str, required=True, help="Directory containing files to ingest")
    parser.add_argument("--chroma_db_path", "-d", type=str, default="./chroma_db", help="Path to Chroma DB")
    parser.add_argument("--chroma_collection_name", "-c", type=str, default="test_collection", help="Chroma collection name")
    parser.add_argument("--embedding_model", "-e", type=str, default="mxbai-embed-large:latest", help="Embedding model name")
    parser.add_argument("--overwrite_existing", "-o", action="store_true", help="Overwrite existing indexed documents")

    args = parser.parse_args()

    ingestor = ingestion(
        ingest_directory=args.ingest_directory,
        chroma_db_path=args.chroma_db_path,
        chroma_collection_name=args.chroma_collection_name,
        embedding_model=args.embedding_model,
        overwrite_existing=args.overwrite_existing
    )
    
    ingestor.process_files()


if __name__ == "__main__":
    main()