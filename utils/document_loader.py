import os
import logging
from typing import List, TypedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DocumentMetadata(TypedDict):
    source: str
    chunk_id: int
    file_path: str
    section_title: str

@dataclass
class Document:
    text: str
    metadata: DocumentMetadata

class DocumentLoader:
    def __init__(self, documents_path: str):
        """Initialize the DocumentLoader with the path to documents directory."""
        self.documents_path = documents_path
        logger.info(f"DocumentLoader initialized with path: {documents_path}")

    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[tuple[str, str]]:
        """
        Split text into chunks while preserving section context.
        Returns list of (section_title, chunk) tuples.
        """
        logger.info("Starting text chunking...")
        chunks = []
        current_section = "General"
        current_chunk = ""
        
        # Split into sections and process each
        sections = text.split('\n\n')
        logger.info(f"Found {len(sections)} sections to process")
        
        for section in sections:
            if not section.strip():
                continue
                
            # Check if this is a section title
            lines = section.split('\n')
            if len(lines) == 1 and not any(char in lines[0] for char in ':.,$-'):
                # If we have a current chunk, save it
                if current_chunk:
                    chunks.append((current_section, current_chunk.strip()))
                    current_chunk = ""
                
                current_section = lines[0].strip()
                logger.debug(f"New section found: {current_section}")
                continue
            
            # Add to current chunk or create new chunk if too large
            if len(current_chunk) + len(section) > max_chunk_size:
                if current_chunk:
                    chunks.append((current_section, current_chunk.strip()))
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append((current_section, current_chunk.strip()))
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def load_documents(self) -> List[Document]:
        """Load and chunk documents from the specified path."""
        logger.info(f"Loading documents from {self.documents_path}")
        documents: List[Document] = []
        
        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Documents directory not found: {self.documents_path}")

        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.documents_path, filename)
                logger.info(f"Processing file: {filename}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_size = len(content)
                        logger.info(f"File size: {content_size:,} bytes")
                        
                        # Split content into chunks while preserving section context
                        chunks = self.chunk_text(content)
                        logger.info(f"Created {len(chunks)} chunks for {filename}")
                        
                        for i, (section_title, chunk) in enumerate(chunks):
                            logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {filename}")
                            metadata: DocumentMetadata = {
                                'source': str(filename),
                                'chunk_id': i,
                                'file_path': str(file_path),
                                'section_title': section_title
                            }
                            doc = Document(
                                text=chunk,
                                metadata=metadata
                            )
                            documents.append(doc)
                            
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
        
        logger.info(f"Total documents created: {len(documents)}")
        return documents