"""
Connection Health Monitoring Module
===================================

Provides comprehensive connection health monitoring, auto-recovery, and resilience
features for database and API connections in the infinite memory chat system.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, AutoReconnect
import openai
from openai import RateLimitError, APIError

# Configure logging
logger = logging.getLogger("infinite_memory_chat")

class ConnectionStatus(Enum):
    """Connection status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"
    FAILED = "failed"

@dataclass
class ConnectionHealth:
    """Health metrics for a connection."""
    status: ConnectionStatus
    last_check: datetime
    consecutive_failures: int
    total_failures: int
    last_success: Optional[datetime]
    last_error: Optional[str]
    response_time_ms: float
    uptime_percentage: float

class ConnectionMonitor:
    """Base class for connection monitoring."""
    
    def __init__(self, name: str, check_interval: int = 30, max_failures: int = 3):
        self.name = name
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.health = ConnectionHealth(
            status=ConnectionStatus.DISCONNECTED,
            last_check=datetime.now(),
            consecutive_failures=0,
            total_failures=0,
            last_success=None,
            last_error=None,
            response_time_ms=0.0,
            uptime_percentage=0.0
        )
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def add_status_callback(self, callback: Callable[[str, ConnectionHealth], None]):
        """Add callback for status changes."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all callbacks of status change."""
        for callback in self.callbacks:
            try:
                callback(self.name, self.health)
            except Exception as e:
                logger.error(f"Callback error for {self.name}: {e}")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started health monitoring for {self.name}")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info(f"Stopped health monitoring for {self.name}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                self.check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitor loop error for {self.name}: {e}")
                time.sleep(self.check_interval)
    
    def check_health(self) -> ConnectionHealth:
        """Check connection health. Override in subclasses."""
        raise NotImplementedError
    
    def attempt_recovery(self) -> bool:
        """Attempt connection recovery. Override in subclasses."""
        raise NotImplementedError

class MongoDBConnectionMonitor(ConnectionMonitor):
    """MongoDB-specific connection health monitor."""
    
    def __init__(self, connection_string: str, database_name: str, **kwargs):
        super().__init__(name="MongoDB", **kwargs)
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.last_successful_operations = []
        
    def check_health(self) -> ConnectionHealth:
        """Check MongoDB connection health."""
        start_time = time.time()
        
        try:
            # Test connection if client exists
            if self.client:
                # Simple ping operation
                result = self.client.admin.command('hello')
                
                # Test database access
                db = self.client[self.database_name]
                collections = db.list_collection_names()
                
                response_time = (time.time() - start_time) * 1000
                
                # Update health metrics
                self.health.response_time_ms = response_time
                self.health.last_check = datetime.now()
                self.health.last_success = datetime.now()
                self.health.consecutive_failures = 0
                
                # Determine status based on response time
                if response_time < 1000:  # < 1 second
                    self.health.status = ConnectionStatus.HEALTHY
                elif response_time < 5000:  # < 5 seconds
                    self.health.status = ConnectionStatus.DEGRADED
                else:
                    self.health.status = ConnectionStatus.DEGRADED
                
                logger.debug(f"MongoDB health check passed ({response_time:.1f}ms)")
                
            else:
                # No client, attempt to connect
                self.health.status = ConnectionStatus.DISCONNECTED
                logger.warning("MongoDB client not initialized")
                
        except (PyMongoError, ServerSelectionTimeoutError, AutoReconnect) as e:
            self._handle_health_check_failure(str(e))
        except Exception as e:
            self._handle_health_check_failure(f"Unexpected error: {e}")
        
        # Calculate uptime percentage
        self._calculate_uptime()
        self._notify_callbacks()
        
        return self.health
    
    def attempt_recovery(self) -> bool:
        """Attempt MongoDB connection recovery."""
        logger.info("Attempting MongoDB connection recovery...")
        
        try:
            self.health.status = ConnectionStatus.RECOVERING
            self._notify_callbacks()
            
            # Close existing connection if it exists
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            
            # Create new connection with retry settings
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=10000,  # 10 second timeout
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=10,
                minPoolSize=1,
                retryWrites=True,
                retryReads=True
            )
            
            # Test the new connection
            self.client.admin.command('hello')
            db = self.client[self.database_name]
            db.list_collection_names()
            
            logger.info("MongoDB connection recovery successful")
            
            # Reset failure counters
            self.health.consecutive_failures = 0
            self.health.last_success = datetime.now()
            self.health.status = ConnectionStatus.HEALTHY
            self._notify_callbacks()
            
            return True
            
        except Exception as e:
            logger.error(f"MongoDB recovery failed: {e}")
            self.health.status = ConnectionStatus.FAILED
            self.health.last_error = str(e)
            self._notify_callbacks()
            return False
    
    def _handle_health_check_failure(self, error_message: str):
        """Handle health check failure."""
        self.health.consecutive_failures += 1
        self.health.total_failures += 1
        self.health.last_error = error_message
        self.health.last_check = datetime.now()
        
        if self.health.consecutive_failures >= self.max_failures:
            self.health.status = ConnectionStatus.FAILED
            logger.error(f"MongoDB connection failed after {self.max_failures} attempts: {error_message}")
            
            # Attempt automatic recovery
            if not self.attempt_recovery():
                logger.error("MongoDB automatic recovery failed")
        else:
            self.health.status = ConnectionStatus.DEGRADED
            logger.warning(f"MongoDB health check failed (attempt {self.health.consecutive_failures}): {error_message}")
    
    def _calculate_uptime(self):
        """Calculate connection uptime percentage."""
        # Simple calculation based on recent success/failure ratio
        total_checks = self.health.total_failures + max(1, self.health.consecutive_failures)
        if total_checks > 0:
            success_rate = max(0, total_checks - self.health.total_failures) / total_checks
            self.health.uptime_percentage = success_rate * 100
        else:
            self.health.uptime_percentage = 100.0

class OpenAIConnectionMonitor(ConnectionMonitor):
    """OpenAI API connection health monitor."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(name="OpenAI", **kwargs)
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limit_info = {
            'requests_per_minute': 0,
            'tokens_per_minute': 0,
            'last_rate_limit': None
        }
        
    def check_health(self) -> ConnectionHealth:
        """Check OpenAI API health with a minimal test request."""
        start_time = time.time()
        
        try:
            # Use a minimal embedding request as health check
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input="health check",
                encoding_format="float"
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Update health metrics
            self.health.response_time_ms = response_time
            self.health.last_check = datetime.now()
            self.health.last_success = datetime.now()
            self.health.consecutive_failures = 0
            
            # Determine status based on response time
            if response_time < 2000:  # < 2 seconds
                self.health.status = ConnectionStatus.HEALTHY
            elif response_time < 10000:  # < 10 seconds
                self.health.status = ConnectionStatus.DEGRADED
            else:
                self.health.status = ConnectionStatus.DEGRADED
            
            logger.debug(f"OpenAI health check passed ({response_time:.1f}ms)")
            
        except RateLimitError as e:
            self._handle_rate_limit(str(e))
        except APIError as e:
            self._handle_health_check_failure(f"API Error: {e}")
        except Exception as e:
            self._handle_health_check_failure(f"Unexpected error: {e}")
        
        self._calculate_uptime()
        self._notify_callbacks()
        
        return self.health
    
    def attempt_recovery(self) -> bool:
        """Attempt OpenAI API recovery (mainly waiting for rate limits)."""
        logger.info("Attempting OpenAI API recovery...")
        
        if self.rate_limit_info['last_rate_limit']:
            # Wait a bit more if we hit rate limits recently
            time_since_rate_limit = datetime.now() - self.rate_limit_info['last_rate_limit']
            if time_since_rate_limit < timedelta(minutes=1):
                logger.info("Waiting for rate limit recovery...")
                time.sleep(60)  # Wait 1 minute
        
        try:
            self.health.status = ConnectionStatus.RECOVERING
            self._notify_callbacks()
            
            # Test with a simple request
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input="recovery test",
                encoding_format="float"
            )
            
            logger.info("OpenAI API recovery successful")
            self.health.consecutive_failures = 0
            self.health.last_success = datetime.now()
            self.health.status = ConnectionStatus.HEALTHY
            self._notify_callbacks()
            
            return True
            
        except Exception as e:
            logger.error(f"OpenAI API recovery failed: {e}")
            self.health.status = ConnectionStatus.FAILED
            self.health.last_error = str(e)
            self._notify_callbacks()
            return False
    
    def _handle_rate_limit(self, error_message: str):
        """Handle rate limit specifically."""
        self.rate_limit_info['last_rate_limit'] = datetime.now()
        self.health.consecutive_failures += 1
        self.health.total_failures += 1
        self.health.last_error = f"Rate Limited: {error_message}"
        self.health.last_check = datetime.now()
        self.health.status = ConnectionStatus.DEGRADED
        
        logger.warning(f"OpenAI API rate limit hit: {error_message}")
    
    def _handle_health_check_failure(self, error_message: str):
        """Handle general health check failure."""
        self.health.consecutive_failures += 1
        self.health.total_failures += 1
        self.health.last_error = error_message
        self.health.last_check = datetime.now()
        
        if self.health.consecutive_failures >= self.max_failures:
            self.health.status = ConnectionStatus.FAILED
            logger.error(f"OpenAI API failed after {self.max_failures} attempts: {error_message}")
        else:
            self.health.status = ConnectionStatus.DEGRADED
            logger.warning(f"OpenAI API health check failed (attempt {self.health.consecutive_failures}): {error_message}")
    
    def _calculate_uptime(self):
        """Calculate API uptime percentage."""
        total_checks = self.health.total_failures + max(1, self.health.consecutive_failures)
        if total_checks > 0:
            success_rate = max(0, total_checks - self.health.total_failures) / total_checks
            self.health.uptime_percentage = success_rate * 100
        else:
            self.health.uptime_percentage = 100.0

class ConnectionHealthManager:
    """Manages multiple connection monitors."""
    
    def __init__(self):
        self.monitors: Dict[str, ConnectionMonitor] = {}
        self.status_callbacks = []
        
    def add_monitor(self, monitor: ConnectionMonitor):
        """Add a connection monitor."""
        self.monitors[monitor.name] = monitor
        monitor.add_status_callback(self._on_status_change)
        logger.info(f"Added connection monitor: {monitor.name}")
    
    def start_all_monitoring(self):
        """Start monitoring for all connections."""
        for monitor in self.monitors.values():
            monitor.start_monitoring()
        logger.info("Started monitoring for all connections")
    
    def stop_all_monitoring(self):
        """Stop monitoring for all connections."""
        for monitor in self.monitors.values():
            monitor.stop_monitoring()
        logger.info("Stopped monitoring for all connections")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.monitors:
            return {"status": "unknown", "monitors": {}}
        
        monitor_statuses = {}
        healthy_count = 0
        
        for name, monitor in self.monitors.items():
            monitor_statuses[name] = {
                "status": monitor.health.status.value,
                "last_check": monitor.health.last_check.isoformat(),
                "consecutive_failures": monitor.health.consecutive_failures,
                "response_time_ms": monitor.health.response_time_ms,
                "uptime_percentage": monitor.health.uptime_percentage,
                "last_error": monitor.health.last_error
            }
            
            if monitor.health.status == ConnectionStatus.HEALTHY:
                healthy_count += 1
        
        # Determine overall status
        if healthy_count == len(self.monitors):
            overall_status = "healthy"
        elif healthy_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "failed"
        
        return {
            "status": overall_status,
            "healthy_monitors": healthy_count,
            "total_monitors": len(self.monitors),
            "monitors": monitor_statuses,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for overall status changes."""
        self.status_callbacks.append(callback)
    
    def _on_status_change(self, monitor_name: str, health: ConnectionHealth):
        """Handle status change from individual monitors."""
        logger.info(f"Connection {monitor_name} status changed to {health.status.value}")
        
        # Notify overall status callbacks
        overall_health = self.get_overall_health()
        for callback in self.status_callbacks:
            try:
                callback(overall_health)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

# Global connection health manager
connection_health_manager = ConnectionHealthManager()
