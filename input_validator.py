"""
Input Validation and Sanitization Module
========================================

Provides security-focused input validation and sanitization for the infinite memory chat system.
Protects against injection attacks, validates API responses, and ensures data integrity.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import html

# Configure logging
logger = logging.getLogger("infinite_memory_chat")

# Security patterns to detect potentially dangerous content
DANGEROUS_PATTERNS = [
    # Script injection patterns
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
    r'javascript:',
    r'vbscript:',
    r'on\w+\s*=',
    
    # SQL injection patterns  
    r'(\bUNION\b.*\bSELECT\b)|(\bSELECT\b.*\bFROM\b)|(\bINSERT\b.*\bINTO\b)',
    r'(\bDROP\b.*\bTABLE\b)|(\bDELETE\b.*\bFROM\b)|(\bUPDATE\b.*\bSET\b)',
    r'(\bALTER\b.*\bTABLE\b)|(\bCREATE\b.*\bTABLE\b)',
    
    # NoSQL injection patterns
    r'\$where',
    r'\$ne',
    r'\$gt',
    r'\$lt',
    r'\$regex',
    r'\$eval',
    
    # Command injection patterns
    r'[;&|`]',
    r'\$\(.*\)',
    r'`.*`',
]

class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass

class APIResponseValidationError(Exception):
    """Raised when API response validation fails."""
    pass

class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, max_message_length: int = 10000, max_session_id_length: int = 100):
        self.max_message_length = max_message_length
        self.max_session_id_length = max_session_id_length
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]
        
    def validate_user_message(self, message: str) -> str:
        """
        Validate and sanitize user chat messages.
        
        Args:
            message: Raw user input message
            
        Returns:
            Sanitized message safe for processing
            
        Raises:
            InputValidationError: If message is invalid or dangerous
        """
        if not isinstance(message, str):
            raise InputValidationError("Message must be a string")
            
        if not message.strip():
            raise InputValidationError("Message cannot be empty")
            
        if len(message) > self.max_message_length:
            raise InputValidationError(f"Message too long (max {self.max_message_length} characters)")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(message):
                logger.warning(f"Potentially dangerous pattern detected in message: {pattern.pattern}")
                raise InputValidationError("Message contains potentially dangerous content")
        
        # Sanitize HTML entities and control characters
        sanitized = html.escape(message)
        sanitized = self._remove_control_characters(sanitized)
        
        logger.debug(f"User message validated and sanitized ({len(sanitized)} chars)")
        return sanitized
    
    def validate_session_id(self, session_id: str) -> str:
        """
        Validate and sanitize session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Validated session ID
            
        Raises:
            InputValidationError: If session ID is invalid
        """
        if not isinstance(session_id, str):
            raise InputValidationError("Session ID must be a string")
            
        if not session_id.strip():
            raise InputValidationError("Session ID cannot be empty")
            
        if len(session_id) > self.max_session_id_length:
            raise InputValidationError(f"Session ID too long (max {self.max_session_id_length} characters)")
        
        # Session IDs should only contain alphanumeric characters, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            raise InputValidationError("Session ID contains invalid characters")
        
        logger.debug(f"Session ID validated: {session_id}")
        return session_id.strip()
    
    def validate_file_path(self, file_path: str) -> str:
        """
        Validate file paths for export/import operations.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            InputValidationError: If file path is dangerous
        """
        if not isinstance(file_path, str):
            raise InputValidationError("File path must be a string")
            
        if not file_path.strip():
            raise InputValidationError("File path cannot be empty")
        
        # Check for path traversal attacks
        if '..' in file_path or file_path.startswith('/'):
            raise InputValidationError("File path contains invalid directory traversal")
        
        # Check for dangerous file extensions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.sh', '.ps1', '.js', '.vbs']
        if any(file_path.lower().endswith(ext) for ext in dangerous_extensions):
            raise InputValidationError("File extension not allowed")
        
        # Ensure reasonable length
        if len(file_path) > 255:
            raise InputValidationError("File path too long")
        
        logger.debug(f"File path validated: {file_path}")
        return file_path.strip()
    
    def validate_openai_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate OpenAI API response structure and content.
        
        Args:
            response_data: Raw response from OpenAI API
            
        Returns:
            Validated response data
            
        Raises:
            APIResponseValidationError: If response is invalid or malformed
        """
        if not isinstance(response_data, dict):
            raise APIResponseValidationError("OpenAI response must be a dictionary")
        
        # Validate required fields based on response type
        if 'choices' in response_data:
            # Chat completion response
            choices = response_data.get('choices', [])
            if not isinstance(choices, list) or not choices:
                raise APIResponseValidationError("OpenAI response missing valid choices")
            
            for choice in choices:
                if not isinstance(choice, dict):
                    raise APIResponseValidationError("Invalid choice format in OpenAI response")
                
                message = choice.get('message', {})
                if not isinstance(message, dict):
                    raise APIResponseValidationError("Invalid message format in OpenAI response")
                
                content = message.get('content', '')
                if not isinstance(content, str):
                    raise APIResponseValidationError("Invalid content format in OpenAI response")
                
                # Sanitize the content
                message['content'] = self._sanitize_api_response_content(content)
        
        elif 'data' in response_data:
            # Embeddings response
            data = response_data.get('data', [])
            if not isinstance(data, list) or not data:
                raise APIResponseValidationError("OpenAI embeddings response missing valid data")
            
            for item in data:
                if not isinstance(item, dict):
                    raise APIResponseValidationError("Invalid embedding item format")
                
                embedding = item.get('embedding', [])
                if not isinstance(embedding, list) or not embedding:
                    raise APIResponseValidationError("Invalid embedding format")
                
                # Validate embedding dimensions and values
                if len(embedding) not in [1536, 3072]:  # Common OpenAI embedding dimensions
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                
                for value in embedding:
                    if not isinstance(value, (int, float)):
                        raise APIResponseValidationError("Invalid embedding value type")
        
        else:
            raise APIResponseValidationError("Unknown OpenAI response format")
        
        logger.debug("OpenAI API response validated successfully")
        return response_data
    
    def validate_mongodb_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate MongoDB document structure before insertion.
        
        Args:
            document: Document to validate
            
        Returns:
            Validated document
            
        Raises:
            InputValidationError: If document is invalid
        """
        if not isinstance(document, dict):
            raise InputValidationError("MongoDB document must be a dictionary")
        
        # Check document size (MongoDB limit is 16MB, we'll be more conservative)
        document_size = len(json.dumps(document, default=str))
        if document_size > 10 * 1024 * 1024:  # 10MB limit
            raise InputValidationError("Document too large for MongoDB storage")
        
        # Validate required fields for archive documents
        if document.get('type') == 'archive':
            required_fields = ['session_id', 'archive_id', 'messages', 'content_text']
            for field in required_fields:
                if field not in document:
                    raise InputValidationError(f"Missing required field: {field}")
        
        # Sanitize text content in messages
        if 'messages' in document:
            for message in document['messages']:
                if isinstance(message, dict) and 'content' in message:
                    message['content'] = self._sanitize_api_response_content(message['content'])
        
        if 'content_text' in document:
            document['content_text'] = self._sanitize_api_response_content(document['content_text'])
        
        logger.debug("MongoDB document validated successfully")
        return document
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except common whitespace."""
        return ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    def _sanitize_api_response_content(self, content: str) -> str:
        """Sanitize content from API responses."""
        if not isinstance(content, str):
            return str(content)
        
        # Remove control characters but preserve formatting
        sanitized = self._remove_control_characters(content)
        
        # Limit length to prevent excessive memory usage
        if len(sanitized) > 50000:  # 50KB limit per response
            sanitized = sanitized[:50000] + "... [truncated]"
            logger.warning("API response content truncated due to length")
        
        return sanitized


class SecurityAuditLogger:
    """Logs security-related events for monitoring and analysis."""
    
    def __init__(self):
        self.security_logger = logging.getLogger("infinite_memory_chat.security")
        
    def log_validation_failure(self, validation_type: str, error_message: str, input_data: str = None):
        """Log validation failures for security monitoring."""
        self.security_logger.warning(
            f"Validation failure: {validation_type} - {error_message}",
            extra={
                'validation_type': validation_type,
                'error_message': error_message,
                'input_sample': input_data[:100] if input_data else None,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_suspicious_activity(self, activity_type: str, details: str):
        """Log suspicious activities that might indicate attacks."""
        self.security_logger.error(
            f"Suspicious activity detected: {activity_type} - {details}",
            extra={
                'activity_type': activity_type,
                'details': details,
                'timestamp': datetime.now().isoformat()
            }
        )


# Global instances
input_validator = InputValidator()
security_audit_logger = SecurityAuditLogger()
