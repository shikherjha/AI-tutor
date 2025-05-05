"""
Document processing utilities for Axon AI Tutor
"""
import os
import tempfile
from typing import Dict, Any, List, Optional
from fastapi import UploadFile, HTTPException

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    CSVLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

class DocumentProcessor:
    """Process and manage document uploads"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize document processor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.embeddings = FastEmbedEmbeddings()
    
    async def process_file(
        self, 
        file: UploadFile, 
        conversation_data: Dict[str, Any],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an uploaded file and add to conversation
        
        Args:
            file: The uploaded file
            conversation_data: Conversation data dictionary
            description: Optional file description
            
        Returns:
            Information about the processed file
        """
        # Create temp file path
        file_path = os.path.join(self.temp_dir, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Process the file based on extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        else:
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Load the document
        try:
            documents = loader.load()
        except Exception as e:
            # Clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Initialize files dict if needed
        if "files" not in conversation_data:
            conversation_data["files"] = {}
            
        # Add file info to conversation
        conversation_data["files"][file.filename] = {
            "path": file_path,
            "vectorstore": vectorstore,
            "description": description
        }
        
        return {
            "filename": file.filename,
            "path": file_path,
            "description": description
        }
    
    def find_relevant_context(
        self, 
        conversation_data: Dict[str, Any], 
        question: str, 
        max_docs: int = 3
    ) -> Optional[str]:
        """
        Find relevant context from uploaded files
        
        Args:
            conversation_data: Conversation data dictionary
            question: Question to find context for
            max_docs: Maximum number of document chunks to return
            
        Returns:
            Relevant context as string or None if no context found
        """
        if "files" not in conversation_data or not conversation_data["files"]:
            return None
            
        relevant_content = []
        
        # Check all files for relevant content
        for file_name, file_info in conversation_data["files"].items():
            if file_name.lower() in question.lower() or not relevant_content:
                # If file is mentioned or we have no content yet
                vectorstore = file_info["vectorstore"]
                docs = vectorstore.similarity_search(question, k=max_docs)
                
                for doc in docs:
                    source = f"From {file_name}"
                    if hasattr(doc, 'metadata') and doc.metadata.get('page'):
                        source += f" (page {doc.metadata['page']})"
                    
                    relevant_content.append(f"{source}:\n{doc.page_content}")
        
        if not relevant_content:
            return None
            
        return "\n\n".join(relevant_content)
    
    def cleanup_files(self, conversation_data: Dict[str, Any]) -> None:
        """
        Clean up temporary files
        
        Args:
            conversation_data: Conversation data dictionary
        """
        if "files" in conversation_data:
            for file_info in conversation_data["files"].values():
                try:
                    if os.path.exists(file_info["path"]):
                        os.remove(file_info["path"])
                except Exception:
                    pass