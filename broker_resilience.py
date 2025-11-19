"""
Broker Connection Resilience Module
Handles connection pooling, auto-reconnection, failover, and session recovery
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import random
import hashlib
import json

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

@dataclass
class ConnectionStats:
    """Connection statistics"""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_attempts: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    last_connection_time: Optional[datetime] = None
    last_disconnection_time: Optional[datetime] = None
    uptime_seconds: float = 0
    downtime_seconds: float = 0

@dataclass
class BrokerConnection:
    """Individual broker connection"""
    broker_id: str
    host: str
    port: int
    state: ConnectionState = ConnectionState.DISCONNECTED
    connection: Optional[Any] = None
    last_heartbeat: Optional[datetime] = None
    reconnect_attempts: int = 0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BrokerConnectionPool:
    """
    Connection pool for managing multiple broker connections
    """
    
    def __init__(
        self,
        broker_configs: List[Dict[str, Any]],
        pool_size: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        heartbeat_interval: float = 30.0,
        connection_timeout: float = 10.0
    ):
        """
        Initialize connection pool
        
        Args:
            broker_configs: List of broker configurations
            pool_size: Maximum number of connections
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay (exponential backoff)
            heartbeat_interval: Heartbeat check interval
            connection_timeout: Connection timeout
        """
        self.broker_configs = broker_configs
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        
        self.connections: List[BrokerConnection] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.stats = ConnectionStats()
        self._heartbeat_task = None
        self._monitor_task = None
        self._shutdown = False
    
    async def initialize(self):
        """Initialize connection pool"""
        logger.info(f"Initializing connection pool with {self.pool_size} connections")
        
        # Create connections
        for i in range(min(self.pool_size, len(self.broker_configs))):
            config = self.broker_configs[i % len(self.broker_configs)]
            connection = BrokerConnection(
                broker_id=f"broker_{i}",
                host=config["host"],
                port=config["port"],
                priority=config.get("priority", 0),
                metadata=config.get("metadata", {})
            )
            self.connections.append(connection)
            
            # Try to establish connection
            if await self._connect(connection):
                await self.available_connections.put(connection)
        
        # Start monitoring tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._monitor_task = asyncio.create_task(self._connection_monitor())
        
        logger.info(f"Connection pool initialized with {self.available_connections.qsize()} available connections")
    
    async def _connect(self, connection: BrokerConnection) -> bool:
        """Establish connection to broker"""
        connection.state = ConnectionState.CONNECTING
        
        try:
            # Simulate connection establishment
            # In real implementation, this would connect to actual broker
            await asyncio.sleep(0.1)  # Simulate connection time
            
            connection.connection = f"MockConnection_{connection.broker_id}"
            connection.state = ConnectionState.CONNECTED
            connection.last_heartbeat = datetime.now()
            
            self.stats.total_connections += 1
            self.stats.successful_connections += 1
            self.stats.last_connection_time = datetime.now()
            
            logger.info(f"Connected to {connection.broker_id} at {connection.host}:{connection.port}")
            return True
            
        except Exception as e:
            connection.state = ConnectionState.FAILED
            self.stats.failed_connections += 1
            logger.error(f"Failed to connect to {connection.broker_id}: {e}")
            return False
    
    async def acquire(self, timeout: Optional[float] = None) -> Optional[BrokerConnection]:
        """
        Acquire connection from pool
        
        Args:
            timeout: Timeout for acquiring connection
            
        Returns:
            Available connection or None
        """
        try:
            if timeout:
                connection = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=timeout
                )
            else:
                connection = await self.available_connections.get()
            
            # Verify connection is still valid
            if connection.state == ConnectionState.CONNECTED:
                return connection
            else:
                # Try to reconnect
                if await self._reconnect(connection):
                    return connection
                return None
                
        except asyncio.TimeoutError:
            logger.warning("Timeout acquiring connection from pool")
            return None
    
    async def release(self, connection: BrokerConnection):
        """Release connection back to pool"""
        if connection.state == ConnectionState.CONNECTED:
            await self.available_connections.put(connection)
        else:
            # Try to reconnect before returning to pool
            asyncio.create_task(self._reconnect_and_return(connection))
    
    async def _reconnect_and_return(self, connection: BrokerConnection):
        """Reconnect and return connection to pool"""
        if await self._reconnect(connection):
            await self.available_connections.put(connection)
    
    async def _reconnect(self, connection: BrokerConnection) -> bool:
        """Reconnect with exponential backoff"""
        connection.state = ConnectionState.RECONNECTING
        
        for attempt in range(self.max_retries):
            delay = self.retry_delay * (2 ** attempt)
            logger.info(f"Reconnection attempt {attempt + 1}/{self.max_retries} for {connection.broker_id} (delay: {delay}s)")
            
            await asyncio.sleep(delay)
            
            if await self._connect(connection):
                connection.reconnect_attempts = 0
                self.stats.reconnection_attempts += 1
                return True
            
            connection.reconnect_attempts += 1
        
        connection.state = ConnectionState.FAILED
        logger.error(f"Failed to reconnect to {connection.broker_id} after {self.max_retries} attempts")
        return False
    
    async def _heartbeat_monitor(self):
        """Monitor connection health with heartbeats"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                for connection in self.connections:
                    if connection.state == ConnectionState.CONNECTED:
                        # Send heartbeat
                        if await self._send_heartbeat(connection):
                            connection.last_heartbeat = datetime.now()
                        else:
                            logger.warning(f"Heartbeat failed for {connection.broker_id}")
                            connection.state = ConnectionState.DISCONNECTED
                            asyncio.create_task(self._reconnect(connection))
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
    
    async def _send_heartbeat(self, connection: BrokerConnection) -> bool:
        """Send heartbeat to broker"""
        try:
            # Simulate heartbeat
            # In real implementation, this would send actual heartbeat
            await asyncio.sleep(0.01)
            return random.random() > 0.05  # 95% success rate for simulation
        except Exception:
            return False
    
    async def _connection_monitor(self):
        """Monitor overall connection health"""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Calculate statistics
                connected = sum(1 for c in self.connections if c.state == ConnectionState.CONNECTED)
                disconnected = sum(1 for c in self.connections if c.state == ConnectionState.DISCONNECTED)
                failed = sum(1 for c in self.connections if c.state == ConnectionState.FAILED)
                
                logger.info(f"Connection pool status: {connected} connected, {disconnected} disconnected, {failed} failed")
                
                # Try to recover failed connections
                for connection in self.connections:
                    if connection.state == ConnectionState.FAILED:
                        asyncio.create_task(self._reconnect(connection))
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        connected = sum(1 for c in self.connections if c.state == ConnectionState.CONNECTED)
        
        return {
            "total_connections": len(self.connections),
            "connected": connected,
            "available": self.available_connections.qsize(),
            "stats": {
                "total_connections": self.stats.total_connections,
                "successful_connections": self.stats.successful_connections,
                "failed_connections": self.stats.failed_connections,
                "reconnection_attempts": self.stats.reconnection_attempts
            },
            "connections": [
                {
                    "broker_id": c.broker_id,
                    "state": c.state.value,
                    "last_heartbeat": c.last_heartbeat.isoformat() if c.last_heartbeat else None
                }
                for c in self.connections
            ]
        }
    
    async def shutdown(self):
        """Shutdown connection pool"""
        logger.info("Shutting down connection pool")
        self._shutdown = True
        
        # Cancel monitoring tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        
        # Close all connections
        for connection in self.connections:
            await self._disconnect(connection)
    
    async def _disconnect(self, connection: BrokerConnection):
        """Disconnect from broker"""
        try:
            # Simulate disconnection
            # In real implementation, this would close actual connection
            connection.connection = None
            connection.state = ConnectionState.DISCONNECTED
            self.stats.last_disconnection_time = datetime.now()
            logger.info(f"Disconnected from {connection.broker_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from {connection.broker_id}: {e}")

class ConnectionManager:
    """
    High-level connection manager with failover support
    """
    
    def __init__(
        self,
        primary_brokers: List[Dict[str, Any]],
        backup_brokers: Optional[List[Dict[str, Any]]] = None,
        session_recovery: bool = True
    ):
        """
        Initialize connection manager
        
        Args:
            primary_brokers: Primary broker configurations
            backup_brokers: Backup broker configurations for failover
            session_recovery: Enable session recovery
        """
        self.primary_brokers = primary_brokers
        self.backup_brokers = backup_brokers or []
        self.session_recovery = session_recovery
        
        self.current_broker = None
        self.session_data = {}
        self.message_queue = deque(maxlen=1000)
        self._failover_in_progress = False
    
    async def connect(self) -> bool:
        """Establish connection with failover support"""
        # Try primary brokers first
        for broker in self.primary_brokers:
            if await self._try_connect(broker):
                self.current_broker = broker
                await self._recover_session()
                return True
        
        # Try backup brokers
        for broker in self.backup_brokers:
            if await self._try_connect(broker):
                self.current_broker = broker
                await self._recover_session()
                return True
        
        logger.error("Failed to connect to any broker")
        return False
    
    async def _try_connect(self, broker: Dict[str, Any]) -> bool:
        """Try to connect to a specific broker"""
        try:
            logger.info(f"Attempting connection to {broker['host']}:{broker['port']}")
            # Simulate connection attempt
            await asyncio.sleep(0.1)
            
            # Random success for simulation
            if random.random() > 0.3:
                logger.info(f"Successfully connected to {broker['host']}:{broker['port']}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def _recover_session(self):
        """Recover session after reconnection"""
        if not self.session_recovery or not self.session_data:
            return
        
        logger.info("Recovering session...")
        
        try:
            # Restore subscriptions
            if "subscriptions" in self.session_data:
                for subscription in self.session_data["subscriptions"]:
                    await self._resubscribe(subscription)
            
            # Resend pending messages
            while self.message_queue:
                message = self.message_queue.popleft()
                await self._resend_message(message)
            
            logger.info("Session recovery completed")
            
        except Exception as e:
            logger.error(f"Session recovery failed: {e}")
    
    async def _resubscribe(self, subscription: Dict[str, Any]):
        """Resubscribe to market data"""
        logger.info(f"Resubscribing to {subscription}")
        # Implementation would resubscribe to actual market data
        pass
    
    async def _resend_message(self, message: Dict[str, Any]):
        """Resend pending message"""
        logger.info(f"Resending message: {message}")
        # Implementation would resend actual message
        pass
    
    async def failover(self):
        """Perform failover to backup broker"""
        if self._failover_in_progress:
            logger.warning("Failover already in progress")
            return
        
        self._failover_in_progress = True
        logger.info("Initiating failover...")
        
        try:
            # Save current session
            self._save_session()
            
            # Disconnect from current broker
            if self.current_broker:
                await self._disconnect_current()
            
            # Try to connect to new broker
            if await self.connect():
                logger.info("Failover completed successfully")
            else:
                logger.error("Failover failed")
                
        finally:
            self._failover_in_progress = False
    
    def _save_session(self):
        """Save current session data"""
        self.session_data = {
            "timestamp": datetime.now().isoformat(),
            "broker": self.current_broker,
            "subscriptions": [],  # Would contain actual subscriptions
            "pending_orders": []   # Would contain pending orders
        }
    
    async def _disconnect_current(self):
        """Disconnect from current broker"""
        logger.info(f"Disconnecting from current broker")
        # Implementation would disconnect from actual broker
        self.current_broker = None
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message with automatic queuing on failure"""
        try:
            if self.current_broker:
                # Simulate message sending
                await asyncio.sleep(0.01)
                
                if random.random() > 0.1:  # 90% success rate
                    return True
                else:
                    raise Exception("Send failed")
            else:
                raise Exception("No active connection")
                
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
            
            # Queue message for retry
            self.message_queue.append(message)
            
            # Trigger reconnection
            asyncio.create_task(self.connect())
            
            return False

# Example usage
if __name__ == "__main__":
    async def main():
        # Configure brokers
        broker_configs = [
            {"host": "broker1.example.com", "port": 8080, "priority": 1},
            {"host": "broker2.example.com", "port": 8080, "priority": 2},
            {"host": "broker3.example.com", "port": 8080, "priority": 3}
        ]
        
        # Create connection pool
        pool = BrokerConnectionPool(broker_configs, pool_size=3)
        await pool.initialize()
        
        # Use connections
        for i in range(5):
            connection = await pool.acquire(timeout=5.0)
            if connection:
                print(f"Got connection: {connection.broker_id}")
                # Do something with connection
                await asyncio.sleep(0.1)
                await pool.release(connection)
            else:
                print("Failed to acquire connection")
        
        # Print statistics
        print(json.dumps(pool.get_stats(), indent=2))
        
        # Shutdown
        await pool.shutdown()
    
    asyncio.run(main())