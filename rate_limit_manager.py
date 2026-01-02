"""
Rate Limiting Manager
====================

Provides intelligent rate limiting, request throttling, and API quota management
for OpenAI and other external APIs in the infinite memory chat system.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Event
import queue
from collections import deque
import asyncio

# Configure logging
logger = logging.getLogger("infinite_memory_chat")

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    EXPONENTIAL_BACKOFF = "exponential_backoff"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 10000
    burst_allowance: int = 10
    backoff_base: float = 2.0
    max_backoff_seconds: float = 300.0
    min_interval_seconds: float = 1.0
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

@dataclass
class RequestMetrics:
    """Metrics for API requests."""
    timestamp: datetime
    tokens_used: int
    response_time_ms: float
    was_rate_limited: bool = False
    error_occurred: bool = False

class RateLimitManager:
    """Manages rate limiting for API requests."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_history: deque = deque()
        self.token_history: deque = deque()
        self.last_request_time = 0.0
        self.consecutive_rate_limits = 0
        self.current_backoff_seconds = 0.0
        self.lock = Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_rate_limits = 0
        self.total_tokens_used = 0
        self.average_response_time = 0.0
        
        logger.info(f"Rate limiting initialized: {config.requests_per_minute} req/min, {config.tokens_per_minute} tokens/min")
    
    def can_make_request(self, estimated_tokens: int = 1000) -> tuple[bool, float]:
        """
        Check if a request can be made now.
        
        Args:
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            Tuple of (can_make_request, wait_time_seconds)
        """
        with self.lock:
            now = time.time()
            current_minute = datetime.now()
            
            # Clean old entries
            self._cleanup_old_entries(current_minute)
            
            # Check if we're in backoff period
            if self.current_backoff_seconds > 0:
                time_since_last = now - self.last_request_time
                if time_since_last < self.current_backoff_seconds:
                    wait_time = self.current_backoff_seconds - time_since_last
                    logger.debug(f"In backoff period, wait {wait_time:.1f}s")
                    return False, wait_time
                else:
                    # Backoff period expired
                    self.current_backoff_seconds = 0.0
            
            # Check request rate limit
            recent_requests = len(self.request_history)
            if recent_requests >= self.config.requests_per_minute:
                wait_time = self._calculate_wait_time_for_requests()
                logger.debug(f"Request rate limit reached, wait {wait_time:.1f}s")
                return False, wait_time
            
            # Check token rate limit
            recent_tokens = sum(req.tokens_used for req in self.token_history)
            if recent_tokens + estimated_tokens > self.config.tokens_per_minute:
                wait_time = self._calculate_wait_time_for_tokens()
                logger.debug(f"Token rate limit would be exceeded, wait {wait_time:.1f}s")
                return False, wait_time
            
            # Check minimum interval
            time_since_last = now - self.last_request_time
            if time_since_last < self.config.min_interval_seconds:
                wait_time = self.config.min_interval_seconds - time_since_last
                logger.debug(f"Minimum interval not met, wait {wait_time:.1f}s")
                return False, wait_time
            
            return True, 0.0
    
    def record_request(self, tokens_used: int, response_time_ms: float, 
                      was_rate_limited: bool = False, error_occurred: bool = False):
        """Record a completed request for rate limiting calculations."""
        with self.lock:
            now = datetime.now()
            
            metrics = RequestMetrics(
                timestamp=now,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                was_rate_limited=was_rate_limited,
                error_occurred=error_occurred
            )
            
            self.request_history.append(metrics)
            self.token_history.append(metrics)
            self.last_request_time = time.time()
            
            # Update statistics
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            if was_rate_limited:
                self.total_rate_limits += 1
                self.consecutive_rate_limits += 1
                self._apply_backoff()
            else:
                self.consecutive_rate_limits = 0
            
            # Update average response time
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time_ms) 
                / self.total_requests
            )
            
            logger.debug(f"Request recorded: {tokens_used} tokens, {response_time_ms:.1f}ms, rate_limited={was_rate_limited}")
    
    def wait_if_needed(self, estimated_tokens: int = 1000) -> float:
        """
        Block until a request can be made, returning the wait time.
        
        Args:
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            Actual wait time in seconds
        """
        start_time = time.time()
        
        while True:
            can_proceed, wait_time = self.can_make_request(estimated_tokens)
            if can_proceed:
                total_wait = time.time() - start_time
                if total_wait > 0.1:  # Log waits longer than 100ms
                    logger.info(f"Rate limiting wait completed: {total_wait:.1f}s")
                return total_wait
            
            # Wait for the calculated time, but check periodically
            sleep_time = min(wait_time, 1.0)  # Check at least every second
            time.sleep(sleep_time)
    
    def _cleanup_old_entries(self, current_time: datetime):
        """Remove entries older than the rate limiting window."""
        cutoff_time = current_time - timedelta(minutes=1)
        
        while self.request_history and self.request_history[0].timestamp < cutoff_time:
            self.request_history.popleft()
        
        while self.token_history and self.token_history[0].timestamp < cutoff_time:
            self.token_history.popleft()
    
    def _calculate_wait_time_for_requests(self) -> float:
        """Calculate wait time based on request rate limit."""
        if not self.request_history:
            return 0.0
        
        oldest_request = self.request_history[0]
        time_until_window_reset = 60 - (datetime.now() - oldest_request.timestamp).total_seconds()
        return max(0.0, time_until_window_reset)
    
    def _calculate_wait_time_for_tokens(self) -> float:
        """Calculate wait time based on token rate limit."""
        if not self.token_history:
            return 0.0
        
        # Find when enough tokens will be available
        current_time = datetime.now()
        tokens_needed = self.config.tokens_per_minute - sum(req.tokens_used for req in self.token_history)
        
        if tokens_needed > 0:
            return 0.0
        
        # Find the oldest request that needs to expire
        oldest_request = self.token_history[0]
        time_until_available = 60 - (current_time - oldest_request.timestamp).total_seconds()
        return max(0.0, time_until_available)
    
    def _apply_backoff(self):
        """Apply exponential backoff after rate limiting."""
        if self.consecutive_rate_limits == 0:
            return
        
        backoff_seconds = min(
            self.config.backoff_base ** (self.consecutive_rate_limits - 1),
            self.config.max_backoff_seconds
        )
        
        self.current_backoff_seconds = backoff_seconds
        logger.warning(f"Rate limited {self.consecutive_rate_limits} times, backing off for {backoff_seconds:.1f}s")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        with self.lock:
            now = datetime.now()
            self._cleanup_old_entries(now)
            
            recent_requests = len(self.request_history)
            recent_tokens = sum(req.tokens_used for req in self.token_history)
            
            # Calculate rates
            request_rate_percent = (recent_requests / self.config.requests_per_minute) * 100
            token_rate_percent = (recent_tokens / self.config.tokens_per_minute) * 100
            
            # Recent error rate
            recent_errors = sum(1 for req in self.request_history if req.error_occurred)
            error_rate_percent = (recent_errors / max(1, recent_requests)) * 100
            
            return {
                "requests_per_minute": recent_requests,
                "tokens_per_minute": recent_tokens,
                "request_rate_percent": request_rate_percent,
                "token_rate_percent": token_rate_percent,
                "consecutive_rate_limits": self.consecutive_rate_limits,
                "current_backoff_seconds": self.current_backoff_seconds,
                "total_requests": self.total_requests,
                "total_rate_limits": self.total_rate_limits,
                "total_tokens_used": self.total_tokens_used,
                "average_response_time_ms": self.average_response_time,
                "error_rate_percent": error_rate_percent,
                "can_make_request_now": self.can_make_request()[0],
                "status": self._get_status_description(request_rate_percent, token_rate_percent)
            }
    
    def _get_status_description(self, request_rate_percent: float, token_rate_percent: float) -> str:
        """Get human-readable status description."""
        max_rate = max(request_rate_percent, token_rate_percent)
        
        if self.current_backoff_seconds > 0:
            return f"backing_off ({self.current_backoff_seconds:.1f}s remaining)"
        elif max_rate >= 90:
            return "near_limit"
        elif max_rate >= 70:
            return "moderate_usage"
        elif max_rate >= 30:
            return "low_usage"
        else:
            return "minimal_usage"

class RequestQueue:
    """Queue system for managing API requests with rate limiting."""
    
    def __init__(self, rate_limit_manager: RateLimitManager, max_queue_size: int = 100):
        self.rate_limit_manager = rate_limit_manager
        self.max_queue_size = max_queue_size
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.is_processing = False
        self.processing_event = Event()
        
        logger.info(f"Request queue initialized with max size {max_queue_size}")
    
    async def enqueue_request(self, request_func: Callable, estimated_tokens: int = 1000, 
                            priority: int = 0, timeout: float = 30.0) -> Any:
        """
        Enqueue a request for processing with rate limiting.
        
        Args:
            request_func: Function to execute (should return the API response)
            estimated_tokens: Estimated token usage
            priority: Request priority (higher = more important)
            timeout: Maximum wait time for the request
            
        Returns:
            The result of the request function
        """
        if self.request_queue.qsize() >= self.max_queue_size:
            raise queue.Full("Request queue is full")
        
        request_id = f"req_{int(time.time() * 1000)}"
        result_event = asyncio.Event()
        result_container = {"result": None, "error": None}
        
        request_item = {
            "id": request_id,
            "func": request_func,
            "estimated_tokens": estimated_tokens,
            "priority": priority,
            "timeout": timeout,
            "enqueued_at": time.time(),
            "result_event": result_event,
            "result_container": result_container
        }
        
        self.request_queue.put(request_item)
        logger.debug(f"Enqueued request {request_id} with {estimated_tokens} estimated tokens")
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
        
        # Wait for result
        try:
            await asyncio.wait_for(result_event.wait(), timeout=timeout)
            if result_container["error"]:
                raise result_container["error"]
            return result_container["result"]
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out after {timeout}s")
            raise TimeoutError(f"Request timed out after {timeout} seconds")
    
    async def _process_queue(self):
        """Process queued requests with rate limiting."""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info("Started processing request queue")
        
        try:
            while not self.request_queue.empty():
                try:
                    # Get next request (blocks if queue is empty)
                    request_item = self.request_queue.get(timeout=1.0)
                    
                    # Check if request has timed out while in queue
                    queue_time = time.time() - request_item["enqueued_at"]
                    if queue_time > request_item["timeout"]:
                        logger.warning(f"Request {request_item['id']} expired in queue after {queue_time:.1f}s")
                        request_item["result_container"]["error"] = TimeoutError("Request expired in queue")
                        request_item["result_event"].set()
                        continue
                    
                    # Wait for rate limiting
                    wait_time = self.rate_limit_manager.wait_if_needed(request_item["estimated_tokens"])
                    if wait_time > 0:
                        logger.debug(f"Waited {wait_time:.1f}s for rate limiting")
                    
                    # Execute the request
                    start_time = time.time()
                    try:
                        result = await request_item["func"]()
                        response_time = (time.time() - start_time) * 1000
                        
                        # Record successful request
                        self.rate_limit_manager.record_request(
                            tokens_used=request_item["estimated_tokens"],  # In real implementation, get actual tokens
                            response_time_ms=response_time,
                            was_rate_limited=False,
                            error_occurred=False
                        )
                        
                        request_item["result_container"]["result"] = result
                        logger.debug(f"Request {request_item['id']} completed successfully in {response_time:.1f}ms")
                        
                    except Exception as e:
                        response_time = (time.time() - start_time) * 1000
                        
                        # Determine if this was a rate limit error
                        was_rate_limited = "rate" in str(e).lower() and "limit" in str(e).lower()
                        
                        # Record failed request
                        self.rate_limit_manager.record_request(
                            tokens_used=0,  # No tokens used on error
                            response_time_ms=response_time,
                            was_rate_limited=was_rate_limited,
                            error_occurred=True
                        )
                        
                        request_item["result_container"]["error"] = e
                        logger.error(f"Request {request_item['id']} failed: {e}")
                    
                    # Signal completion
                    request_item["result_event"].set()
                    
                except queue.Empty:
                    # Queue is empty, we're done
                    break
                    
        finally:
            self.is_processing = False
            logger.info("Stopped processing request queue")

class AdaptiveRateLimitManager:
    """Advanced rate limiting with adaptive behavior based on API responses."""
    
    def __init__(self, initial_config: RateLimitConfig):
        self.base_manager = RateLimitManager(initial_config)
        self.config = initial_config
        self.adaptation_factor = 1.0  # Multiplier for limits
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 300  # 5 minutes
        
        # Performance tracking for adaptation
        self.recent_success_rate = 1.0
        self.recent_avg_response_time = 1000.0  # ms
        
    def adapt_limits(self):
        """Adapt rate limits based on recent performance."""
        now = time.time()
        if now - self.last_adaptation_time < self.adaptation_interval:
            return
        
        status = self.base_manager.get_current_status()
        
        # Calculate adaptation based on multiple factors
        adaptation_needed = False
        
        # If we're getting rate limited frequently, reduce limits
        if status["total_rate_limits"] > 0 and status["consecutive_rate_limits"] > 2:
            self.adaptation_factor *= 0.8  # Reduce by 20%
            adaptation_needed = True
            logger.info("Reducing rate limits due to frequent rate limiting")
        
        # If error rate is high, be more conservative
        elif status["error_rate_percent"] > 10:
            self.adaptation_factor *= 0.9  # Reduce by 10%
            adaptation_needed = True
            logger.info("Reducing rate limits due to high error rate")
        
        # If we're consistently under limits and performing well, increase slightly
        elif (status["request_rate_percent"] < 50 and 
              status["error_rate_percent"] < 2 and 
              status["consecutive_rate_limits"] == 0):
            self.adaptation_factor *= 1.1  # Increase by 10%
            adaptation_needed = True
            logger.info("Increasing rate limits due to good performance")
        
        # Apply bounds to adaptation factor
        self.adaptation_factor = max(0.1, min(2.0, self.adaptation_factor))
        
        if adaptation_needed:
            # Update the rate limiting configuration
            new_requests_per_minute = int(self.config.requests_per_minute * self.adaptation_factor)
            new_tokens_per_minute = int(self.config.tokens_per_minute * self.adaptation_factor)
            
            logger.info(f"Adapted rate limits: {new_requests_per_minute} req/min, {new_tokens_per_minute} tokens/min")
            
            # Create new config and manager
            new_config = RateLimitConfig(
                requests_per_minute=new_requests_per_minute,
                tokens_per_minute=new_tokens_per_minute,
                burst_allowance=self.config.burst_allowance,
                backoff_base=self.config.backoff_base,
                max_backoff_seconds=self.config.max_backoff_seconds,
                min_interval_seconds=self.config.min_interval_seconds,
                strategy=self.config.strategy
            )
            
            # Transfer state to new manager
            old_total_requests = self.base_manager.total_requests
            old_total_rate_limits = self.base_manager.total_rate_limits
            old_total_tokens = self.base_manager.total_tokens_used
            old_avg_response_time = self.base_manager.average_response_time
            
            self.base_manager = RateLimitManager(new_config)
            
            # Restore key statistics
            self.base_manager.total_requests = old_total_requests
            self.base_manager.total_rate_limits = old_total_rate_limits
            self.base_manager.total_tokens_used = old_total_tokens
            self.base_manager.average_response_time = old_avg_response_time
        
        self.last_adaptation_time = now
    
    def __getattr__(self, name):
        """Delegate other methods to the base manager."""
        return getattr(self.base_manager, name)

# Default configurations for different usage patterns
DEFAULT_CONFIGS = {
    "conservative": RateLimitConfig(
        requests_per_minute=30,
        tokens_per_minute=5000,
        burst_allowance=5,
        min_interval_seconds=2.0
    ),
    "moderate": RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=10000,
        burst_allowance=10,
        min_interval_seconds=1.0
    ),
    "aggressive": RateLimitConfig(
        requests_per_minute=100,
        tokens_per_minute=20000,
        burst_allowance=20,
        min_interval_seconds=0.5
    )
}

# Global rate limit manager (will be initialized by the application)
global_rate_limit_manager: Optional[AdaptiveRateLimitManager] = None
