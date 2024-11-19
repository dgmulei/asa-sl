from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, TypedDict, Set
import logging
from tqdm import tqdm
import os
import json
from .document_loader import Document

logger = logging.getLogger(__name__)

class ChromaMetadata(TypedDict):
    source: str
    chunk_id: int
    file_path: str

class EmbeddingsManager:
    def __init__(self, model_name: str, db_path: str):
        """Initialize the embeddings manager with a specified model and database path."""
        logger.info(f"Initializing EmbeddingsManager with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.processed_files_path = os.path.join(db_path, "processed_files.json")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="real_estate_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load processed files and perform cleanup
        self.processed_files = self._load_processed_files()
        self._cleanup_missing_files()

    def _load_processed_files(self) -> Set[str]:
        """Load the set of already processed files."""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading processed files list: {e}")
                return set()
        return set()

    def _save_processed_files(self) -> None:
        """Save the set of processed files."""
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.processed_files_path, 'w') as f:
            json.dump(list(self.processed_files), f)

    def _cleanup_missing_files(self) -> None:
        """Remove entries for files that no longer exist in the documents directory."""
        if not self.processed_files:
            return

        # Get the actual files in the documents directory
        docs_dir = os.getenv('DOCUMENTS_PATH', './data/real_estate_docs')
        existing_files = set(f for f in os.listdir(docs_dir) if f.endswith('.txt'))
        
        # Find files that have been processed but no longer exist
        missing_files = self.processed_files - existing_files
        
        if missing_files:
            logger.info(f"Found {len(missing_files)} files to clean up: {missing_files}")
            
            for filename in missing_files:
                try:
                    # Delete from Chroma DB
                    results = self.collection.get(
                        where={"source": filename}
                    )
                    
                    if results and results['ids']:
                        self.collection.delete(
                            ids=results['ids']
                        )
                        logger.info(f"Deleted entries for {filename} from Chroma DB")
                    
                    # Remove from processed files list
                    self.processed_files.remove(filename)
                    logger.info(f"Removed {filename} from processed files list")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up {filename}: {e}")
            
            # Save updated processed files list
            self._save_processed_files()
            logger.info("Cleanup completed")

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """
        Generate embeddings only for new documents and store them in the Chroma database.
        """
        if not documents:
            return

        # Group documents by source file
        documents_by_file: Dict[str, List[Document]] = {}
        for doc in documents:
            source = str(doc.metadata['source'])
            if source not in documents_by_file:
                documents_by_file[source] = []
            documents_by_file[source].append(doc)

        # Process only new files
        new_documents = []
        for source, docs in documents_by_file.items():
            if source not in self.processed_files:
                new_documents.extend(docs)
                logger.info(f"Found new file to process: {source}")

        if not new_documents:
            logger.info("No new documents to process")
            return

        logger.info(f"Processing {len(new_documents)} documents from {len(documents_by_file)} new files")
        
        # Process in batches
        for i in tqdm(range(0, len(new_documents), batch_size), desc="Creating embeddings"):
            batch = new_documents[i:i + batch_size]
            
            texts = [doc.text for doc in batch]
            metadatas: List[ChromaMetadata] = []
            
            for doc in batch:
                metadata: ChromaMetadata = {
                    'source': str(doc.metadata['source']),
                    'chunk_id': int(doc.metadata['chunk_id']),
                    'file_path': str(doc.metadata['file_path'])
                }
                metadatas.append(metadata)
                
            ids = [f"{metadata['source']}_{metadata['chunk_id']}" for metadata in metadatas]
            
            try:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,  # type: ignore
                    ids=ids
                )
                logger.debug(f"Successfully added batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error adding batch to collection: {str(e)}")
                raise

        # Update processed files list
        for source in documents_by_file.keys():
            if source not in self.processed_files:
                self.processed_files.add(source)
                logger.info(f"Marked {source} as processed")

        # Save updated processed files list
        self._save_processed_files()
    
    def get_collection(self) -> Any:
        """Return the Chroma collection for querying."""
        return self.collection