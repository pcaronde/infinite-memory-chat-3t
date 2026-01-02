"""
Enhanced Error Recovery System
==============================

Provides graceful error handling, user-friendly error messages, and system resilience
features for the infinite memory chat system.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, AutoReconnect
from openai import RateLimitError, APIError, InternalServerError
from input_validator import InputValidationError, APIResponseValidationError

# Configure logging
logger = logging.getLogger("infinite_memory_chat")

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better handling."""
    VALIDATION = "validation"
    CONNECTION = "connection"
    API_LIMIT = "api_limit"
    AUTHENTICATION = "authentication"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    suggestion: str
    technical_details: Optional[str]
    recovery_actions: list
    timestamp: datetime
    is_recoverable: bool

class ErrorRecoveryManager:
    """Central error recovery and handling system."""
    
    def __init__(self):
        self.error_handlers: Dict[type, Callable] = {}
        self.recovery_strategies: Dict[ErrorCategory, list] = {}
        self.error_history = []
        self.max_error_history = 100
        
        # Register default error handlers
        self._register_default_handlers()
        self._register_recovery_strategies()
    
    def _register_default_handlers(self):
        """Register default error handlers for common exceptions."""
        
        # Input validation errors
        self.register_error_handler(InputValidationError, self._handle_validation_error)
        self.register_error_handler(APIResponseValidationError, self._handle_api_validation_error)
        
        # MongoDB errors
        self.register_error_handler(PyMongoError, self._handle_mongodb_error)
        self.register_error_handler(ServerSelectionTimeoutError, self._handle_mongodb_timeout)
        self.register_error_handler(AutoReconnect, self._handle_mongodb_reconnect)
        
        # OpenAI API errors
        self.register_error_handler(RateLimitError, self._handle_rate_limit_error)
        self.register_error_handler(APIError, self._handle_api_error)
        self.register_error_handler(InternalServerError, self._handle_api_server_error)
        
        # System errors
        self.register_error_handler(ConnectionError, self._handle_connection_error)
        self.register_error_handler(TimeoutError, self._handle_timeout_error)
        self.register_error_handler(ValueError, self._handle_value_error)
        self.register_error_handler(KeyError, self._handle_key_error)
        
    def _register_recovery_strategies(self):
        """Register recovery strategies for different error categories."""
        
        self.recovery_strategies[ErrorCategory.CONNECTION] = [
            "retry_with_backoff",
            "fallback_to_alternative",
            "graceful_degradation"
        ]
        
        self.recovery_strategies[ErrorCategory.API_LIMIT] = [
            "wait_and_retry",
            "use_alternative_model",
            "batch_operations"
        ]
        
        self.recovery_strategies[ErrorCategory.VALIDATION] = [
            "sanitize_input",
            "use_defaults",
            "request_user_correction"
        ]
        
        self.recovery_strategies[ErrorCategory.DATA_INTEGRITY] = [
            "validate_and_repair",
            "use_backup_data",
            "reinitialize_storage"
        ]
    
    def register_error_handler(self, exception_type: type, handler: Callable):
        """Register a custom error handler."""
        self.error_handlers[exception_type] = handler
        logger.debug(f"Registered error handler for {exception_type.__name__}")
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        Central error handling with recovery suggestions.
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error
            
        Returns:
            ErrorInfo with handling details and recovery suggestions
        """
        context = context or {}
        
        # Find the most specific handler
        handler = self._find_handler(type(exception))
        
        if handler:
            error_info = handler(exception, context)
        else:
            error_info = self._handle_unknown_error(exception, context)
        
        # Add to error history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Log the error
        self._log_error(error_info, exception)
        
        return error_info
    
    def _find_handler(self, exception_type: type) -> Optional[Callable]:
        """Find the most specific error handler."""
        # Check for exact match first
        if exception_type in self.error_handlers:
            return self.error_handlers[exception_type]
        
        # Check for parent class matches
        for registered_type, handler in self.error_handlers.items():
            if issubclass(exception_type, registered_type):
                return handler
        
        return None
    
    def _handle_validation_error(self, exception: InputValidationError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle input validation errors."""
        return ErrorInfo(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            user_message="Your input contains invalid or potentially unsafe content. Please try again with different text.",
            suggestion="Review your message for special characters, excessive length, or potentially harmful content.",
            technical_details=f"Validation failed: {exception}",
            recovery_actions=["sanitize_input", "request_correction"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_api_validation_error(self, exception: APIResponseValidationError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle API response validation errors."""
        return ErrorInfo(
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            user_message="There was an issue with the AI service response. Please try your request again.",
            suggestion="If the problem persists, try rephrasing your message or contact support.",
            technical_details=f"API response validation failed: {exception}",
            recovery_actions=["retry_request", "fallback_mode"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_mongodb_error(self, exception: PyMongoError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle general MongoDB errors."""
        return ErrorInfo(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            user_message="Database connection issue. Your message may not be saved properly.",
            suggestion="The system will attempt to reconnect automatically. You can continue chatting.",
            technical_details=f"MongoDB error: {exception}",
            recovery_actions=["retry_connection", "cache_locally", "fallback_storage"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_mongodb_timeout(self, exception: ServerSelectionTimeoutError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle MongoDB connection timeouts."""
        return ErrorInfo(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            message="Database connection timed out",
            user_message="Connection to the database is slow or unavailable. Your conversation will continue without long-term memory.",
            suggestion="Check your internet connection. The system will retry automatically.",
            technical_details=f"MongoDB timeout: {exception}",
            recovery_actions=["retry_with_backoff", "use_local_memory", "fallback_mode"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_mongodb_reconnect(self, exception: AutoReconnect, context: Dict[str, Any]) -> ErrorInfo:
        """Handle MongoDB auto-reconnection events."""
        return ErrorInfo(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.MEDIUM,
            message="Database reconnecting",
            user_message="Database connection was restored. Your conversation history is safe.",
            suggestion="No action needed. The system recovered automatically.",
            technical_details=f"MongoDB auto-reconnect: {exception}",
            recovery_actions=["connection_restored"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_rate_limit_error(self, exception: RateLimitError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle OpenAI rate limit errors."""
        return ErrorInfo(
            category=ErrorCategory.API_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            message="API rate limit exceeded",
            user_message="The AI service is temporarily busy. Please wait a moment and try again.",
            suggestion="Wait 30-60 seconds before sending another message.",
            technical_details=f"OpenAI rate limit: {exception}",
            recovery_actions=["wait_and_retry", "use_queue", "inform_user"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_api_error(self, exception: APIError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle general OpenAI API errors."""
        severity = ErrorSeverity.HIGH if "authentication" in str(exception).lower() else ErrorSeverity.MEDIUM
        
        return ErrorInfo(
            category=ErrorCategory.AUTHENTICATION if "authentication" in str(exception).lower() else ErrorCategory.SYSTEM,
            severity=severity,
            message=str(exception),
            user_message="AI service encountered an error. Please try your request again.",
            suggestion="If the problem persists, try rephrasing your message or check your API configuration.",
            technical_details=f"OpenAI API error: {exception}",
            recovery_actions=["retry_request", "check_credentials", "fallback_mode"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_api_server_error(self, exception: InternalServerError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle OpenAI internal server errors."""
        return ErrorInfo(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            message="AI service internal error",
            user_message="The AI service is experiencing technical difficulties. Please try again in a few minutes.",
            suggestion="This is a temporary issue with the AI service. Try again later.",
            technical_details=f"OpenAI server error: {exception}",
            recovery_actions=["wait_and_retry", "use_fallback", "notify_admin"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_connection_error(self, exception: ConnectionError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle general connection errors."""
        return ErrorInfo(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            message="Network connection error",
            user_message="Network connection issue. Please check your internet connection and try again.",
            suggestion="Verify your internet connection and firewall settings.",
            technical_details=f"Connection error: {exception}",
            recovery_actions=["retry_connection", "check_network", "use_offline_mode"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_timeout_error(self, exception: TimeoutError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle timeout errors."""
        return ErrorInfo(
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.MEDIUM,
            message="Operation timed out",
            user_message="The operation took too long to complete. Please try again.",
            suggestion="Try again with a shorter message or check your connection.",
            technical_details=f"Timeout error: {exception}",
            recovery_actions=["retry_with_shorter_timeout", "split_request"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_value_error(self, exception: ValueError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle value errors."""
        return ErrorInfo(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            user_message="Invalid input detected. Please check your input and try again.",
            suggestion="Ensure your input follows the expected format.",
            technical_details=f"Value error: {exception}",
            recovery_actions=["validate_input", "use_defaults"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _handle_key_error(self, exception: KeyError, context: Dict[str, Any]) -> ErrorInfo:
        """Handle key errors (missing configuration, etc.)."""
        return ErrorInfo(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            message=f"Missing required configuration: {exception}",
            user_message="System configuration error. Please check your setup.",
            suggestion="Verify all required configuration values are set in your environment.",
            technical_details=f"Key error: {exception}",
            recovery_actions=["check_config", "use_defaults", "notify_admin"],
            timestamp=datetime.now(),
            is_recoverable=False
        )
    
    def _handle_unknown_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorInfo:
        """Handle unknown/unregistered errors."""
        return ErrorInfo(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
            message=str(exception),
            user_message="An unexpected error occurred. The system will continue to function.",
            suggestion="If this error persists, please contact support with details of what you were doing.",
            technical_details=f"Unknown error ({type(exception).__name__}): {exception}",
            recovery_actions=["log_error", "continue_operation", "notify_admin"],
            timestamp=datetime.now(),
            is_recoverable=True
        )
    
    def _log_error(self, error_info: ErrorInfo, exception: Exception):
        """Log error information appropriately."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"Error handled: {error_info.category.value} - {error_info.message}",
            extra={
                'error_category': error_info.category.value,
                'error_severity': error_info.severity.value,
                'is_recoverable': error_info.is_recoverable,
                'recovery_actions': error_info.recovery_actions,
                'technical_details': error_info.technical_details,
                'exception_type': type(exception).__name__
            },
            exc_info=error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}
        
        category_counts = {}
        severity_counts = {}
        recoverable_count = 0
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            if error.is_recoverable:
                recoverable_count += 1
        
        return {
            "total_errors": len(self.error_history),
            "recoverable_errors": recoverable_count,
            "error_rate_percentage": (recoverable_count / len(self.error_history)) * 100,
            "categories": category_counts,
            "severities": severity_counts,
            "most_recent": self.error_history[-1].timestamp.isoformat() if self.error_history else None
        }

class GracefulDegradationManager:
    """Manages graceful degradation strategies when systems fail."""
    
    def __init__(self):
        self.degradation_modes = {
            "full_functionality": 0,
            "limited_memory": 1,
            "local_only": 2,
            "read_only": 3,
            "minimal": 4
        }
        self.current_mode = "full_functionality"
        self.mode_callbacks = []
    
    def add_mode_callback(self, callback: Callable[[str], None]):
        """Add callback for mode changes."""
        self.mode_callbacks.append(callback)
    
    def degrade_to_mode(self, mode: str, reason: str):
        """Degrade system to specified mode."""
        if mode not in self.degradation_modes:
            logger.error(f"Unknown degradation mode: {mode}")
            return
        
        if self.degradation_modes[mode] > self.degradation_modes[self.current_mode]:
            self.current_mode = mode
            logger.warning(f"System degraded to mode '{mode}' due to: {reason}")
            
            # Notify callbacks
            for callback in self.mode_callbacks:
                try:
                    callback(mode)
                except Exception as e:
                    logger.error(f"Mode callback error: {e}")
    
    def try_restore_mode(self, target_mode: str = "full_functionality") -> bool:
        """Attempt to restore to better functionality mode."""
        if self.current_mode == target_mode:
            return True
        
        # This would include actual health checks for the target mode
        # For now, we'll just log the attempt
        logger.info(f"Attempting to restore from '{self.current_mode}' to '{target_mode}'")
        
        # In a real implementation, this would test if the systems are healthy enough
        # to support the target mode
        return False
    
    def get_current_capabilities(self) -> Dict[str, bool]:
        """Get current system capabilities based on degradation mode."""
        capabilities = {
            "full_functionality": {
                "chat": True,
                "memory_storage": True,
                "memory_search": True,
                "session_management": True,
                "export_import": True
            },
            "limited_memory": {
                "chat": True,
                "memory_storage": False,
                "memory_search": False,
                "session_management": True,
                "export_import": True
            },
            "local_only": {
                "chat": True,
                "memory_storage": False,
                "memory_search": False,
                "session_management": False,
                "export_import": False
            },
            "read_only": {
                "chat": False,
                "memory_storage": False,
                "memory_search": True,
                "session_management": False,
                "export_import": False
            },
            "minimal": {
                "chat": False,
                "memory_storage": False,
                "memory_search": False,
                "session_management": False,
                "export_import": False
            }
        }
        
        return capabilities.get(self.current_mode, capabilities["minimal"])

# Global instances
error_recovery_manager = ErrorRecoveryManager()
graceful_degradation_manager = GracefulDegradationManager()
