"""
Abstract base class for vector storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class VectorBackend(ABC):
    """Abstract base class for vector storage backends."""
    
    @abstractmethod
    def setup_store(self, session_id: str) -> str:
        """
        Create a new vector store for the session.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            Store identifier (vector store ID for OpenAI, collection name for MongoDB)
        """
        pass
    
    @abstractmethod
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """
        Archive messages to vector store.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            archive_id: Sequential archive number
            session_id: Session identifier
            
        Returns:
            True if archival was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search archived messages for relevant content.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of relevant archive results with score information
        """
        pass
    
    @abstractmethod
    def consolidate_archives(self, archive_ids: List[str]) -> Optional[str]:
        """
        Consolidate multiple archives into a single unit.
        
        Args:
            archive_ids: List of archive identifiers to consolidate
            
        Returns:
            Consolidated archive identifier, or None if consolidation failed
        """
        pass
    
    @abstractmethod
    def get_archive_count(self) -> int:
        """
        Get the number of archived message groups.
        
        Returns:
            Number of separate archives stored
        """
        pass
    
    @abstractmethod
    def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """
        Remove specified archives from storage.
        
        Args:
            archive_ids: List of archive identifiers to remove
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get backend-specific status information.
        
        Returns:
            Dictionary with status information (archive counts, storage size, etc.)
        """
        pass
    
    @abstractmethod
    def should_consolidate(self, max_archives: int) -> bool:
        """
        Check if consolidation should be triggered.
        
        Args:
            max_archives: Maximum number of archives before consolidation
            
        Returns:
            True if consolidation should be performed
        """
        pass
