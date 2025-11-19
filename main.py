"""
Improved main.py with multiple real-time data source options:
- Polygon WebSocket
- Redis PubSub
- AWS MSK (Kafka)
- Alpaca/IBKR real-time feeds
"""
import asyncio
import signal
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Available data source types"""
    POLYGON_REST = "polygon_rest"           # REST API (high latency)
    POLYGON_WEBSOCKET = "polygon_websocket" # WebSocket (low latency)
    REDIS_PUBSUB = "redis_pubsub"          # Redis PubSub
    AWS_MSK = "aws_msk"                     # Kafka on AWS
    ALPACA_STREAM = "alpaca_stream"         # Alpaca real-time
    IBKR_STREAM = "ibkr_stream"             # Interactive Brokers

class DataSourceManager:
    """Manages multiple data source connections"""
    
    def __init__(self, source_type: DataSourceType, config: Dict):
        self.source_type = source_type
        self.config = config
        self.connection = None
        self.callbacks = []
        
        logger.info(f"Initializing data source: {source_type.value}")
    
    async def initialize(self):
        """Initialize the selected data source"""
        if self.source_type == DataSourceType.POLYGON_WEBSOCKET:
            await self._init_polygon_websocket()
        elif self.source_type == DataSourceType.REDIS_PUBSUB:
            await self._init_redis_pubsub()
        elif self.source_type == DataSourceType.AWS_MSK:
            await self._init_aws_msk()
        elif self.source_type == DataSourceType.ALPACA_STREAM:
            await self._init_alpaca_stream()
        elif self.source_type == DataSourceType.IBKR_STREAM:
            await self._init_ibkr_stream()
        else:
            await self._init_polygon_rest()
    
    async def _init_polygon_websocket(self):
        """Initialize Polygon WebSocket connection"""
        try:
            from polygon import WebSocketClient
            from polygon.websocket import Market
            
            api_key = self.config.get('polygon_api_key')
            symbols = self.config.get('symbols', [])
            
            def handle_msg(msgs):
                for msg in msgs:
                    self._process_message(msg)
            
            self.connection = WebSocketClient(
                api_key=api_key,
                market=Market.Stocks,
                feed='delayed'  # or 'realtime' for paid tier
            )
            
            # Subscribe to trades and quotes
            self.connection.subscribe_trades(handle_msg, *symbols)
            self.connection.subscribe_quotes(handle_msg, *symbols)
            
            logger.info(f"✅ Polygon WebSocket connected for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polygon WebSocket: {e}")
            raise
    
    async def _init_redis_pubsub(self):
        """Initialize Redis PubSub connection"""
        try:
            import redis.asyncio as redis
            
            redis_host = self.config.get('redis_host', 'localhost')
            redis_port = self.config.get('redis_port', 6379)
            redis_db = self.config.get('redis_db', 0)
            channels = self.config.get('channels', ['market_data'])
            
            self.connection = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            
            pubsub = self.connection.pubsub()
            await pubsub.subscribe(*channels)
            
            # Start listening task
            asyncio.create_task(self._redis_listener(pubsub))
            
            logger.info(f"✅ Redis PubSub connected to {len(channels)} channels")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis PubSub: {e}")
            raise
    
    async def _redis_listener(self, pubsub):
        """Listen to Redis messages"""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    self._process_message(data)
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
    
    async def _init_aws_msk(self):
        """Initialize AWS MSK (Kafka) connection"""
        try:
            from aiokafka import AIOKafkaConsumer
            
            bootstrap_servers = self.config.get('kafka_brokers', ['localhost:9092'])
            topics = self.config.get('kafka_topics', ['market_data'])
            group_id = self.config.get('kafka_group_id', 'ares7_consumer')
            
            self.connection = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            await self.connection.start()
            
            # Start consuming task
            asyncio.create_task(self._kafka_consumer())
            
            logger.info(f"✅ AWS MSK connected to {len(topics)} topics")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS MSK: {e}")
            raise
    
    async def _kafka_consumer(self):
        """Consume Kafka messages"""
        try:
            async for msg in self.connection:
                self._process_message(msg.value)
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
    
    async def _init_alpaca_stream(self):
        """Initialize Alpaca real-time stream"""
        try:
            from alpaca_trade_api.stream import Stream
            
            api_key = self.config.get('alpaca_api_key')
            api_secret = self.config.get('alpaca_api_secret')
            base_url = self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
            symbols = self.config.get('symbols', [])
            
            self.connection = Stream(
                api_key,
                api_secret,
                base_url=base_url,
                data_feed='iex'  # or 'sip' for paid tier
            )
            
            # Subscribe to trades
            for symbol in symbols:
                self.connection.subscribe_trades(self._process_alpaca_trade, symbol)
                self.connection.subscribe_quotes(self._process_alpaca_quote, symbol)
            
            # Start stream
            asyncio.create_task(self.connection._run_forever())
            
            logger.info(f"✅ Alpaca stream connected for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca stream: {e}")
            raise
    
    async def _process_alpaca_trade(self, trade):
        """Process Alpaca trade message"""
        data = {
            'type': 'trade',
            'symbol': trade.symbol,
            'price': trade.price,
            'size': trade.size,
            'timestamp': trade.timestamp
        }
        self._process_message(data)
    
    async def _process_alpaca_quote(self, quote):
        """Process Alpaca quote message"""
        data = {
            'type': 'quote',
            'symbol': quote.symbol,
            'bid': quote.bid_price,
            'ask': quote.ask_price,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'timestamp': quote.timestamp
        }
        self._process_message(data)
    
    async def _init_ibkr_stream(self):
        """Initialize Interactive Brokers stream"""
        try:
            from ib_insync import IB, Stock, util
            
            host = self.config.get('ibkr_host', '127.0.0.1')
            port = self.config.get('ibkr_port', 7497)
            client_id = self.config.get('ibkr_client_id', 1)
            symbols = self.config.get('symbols', [])
            
            self.connection = IB()
            await self.connection.connectAsync(host, port, clientId=client_id)
            
            # Subscribe to market data
            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                self.connection.reqMktData(contract, '', False, False)
            
            # Set up callbacks
            self.connection.pendingTickersEvent += self._process_ibkr_tickers
            
            logger.info(f"✅ IBKR stream connected for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to initialize IBKR stream: {e}")
            raise
    
    def _process_ibkr_tickers(self, tickers):
        """Process IBKR ticker updates"""
        for ticker in tickers:
            data = {
                'type': 'quote',
                'symbol': ticker.contract.symbol,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'volume': ticker.volume,
                'timestamp': datetime.now().isoformat()
            }
            self._process_message(data)
    
    async def _init_polygon_rest(self):
        """Initialize Polygon REST API (fallback, high latency)"""
        try:
            from polygon import RESTClient
            
            api_key = self.config.get('polygon_api_key')
            self.connection = RESTClient(api_key)
            
            logger.warning("⚠️  Using Polygon REST API - high latency mode")
            logger.info("✅ Polygon REST client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polygon REST: {e}")
            raise
    
    def _process_message(self, data: Dict):
        """Process incoming market data message"""
        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def register_callback(self, callback):
        """Register a callback for market data"""
        self.callbacks.append(callback)
    
    async def close(self):
        """Close data source connection"""
        if self.connection:
            try:
                if self.source_type == DataSourceType.REDIS_PUBSUB:
                    await self.connection.close()
                elif self.source_type == DataSourceType.AWS_MSK:
                    await self.connection.stop()
                elif self.source_type == DataSourceType.IBKR_STREAM:
                    self.connection.disconnect()
                
                logger.info(f"✅ {self.source_type.value} connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

class AresMainOrchestrator:
    """Main orchestrator with adaptive data source selection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mode = config.get('mode', 'BACKTEST')
        self.data_source = None
        self.running = False
        
        # Select data source based on mode and config
        self.source_type = self._select_data_source()
        
        logger.info(f"ARES-7 v73 initialized in {self.mode} mode")
        logger.info(f"Data source: {self.source_type.value}")
    
    def _select_data_source(self) -> DataSourceType:
        """Select optimal data source based on configuration"""
        
        if self.mode == 'BACKTEST':
            return DataSourceType.POLYGON_REST
        
        # For LIVE/PAPER mode, check available sources in priority order
        if self.config.get('redis_host'):
            logger.info("✅ Redis available - using Redis PubSub (lowest latency)")
            return DataSourceType.REDIS_PUBSUB
        
        elif self.config.get('kafka_brokers'):
            logger.info("✅ Kafka available - using AWS MSK")
            return DataSourceType.AWS_MSK
        
        elif self.config.get('alpaca_api_key'):
            logger.info("✅ Alpaca available - using Alpaca stream")
            return DataSourceType.ALPACA_STREAM
        
        elif self.config.get('ibkr_host'):
            logger.info("✅ IBKR available - using Interactive Brokers")
            return DataSourceType.IBKR_STREAM
        
        elif self.config.get('polygon_api_key'):
            logger.info("⚠️  Using Polygon WebSocket (moderate latency)")
            return DataSourceType.POLYGON_WEBSOCKET
        
        else:
            logger.warning("⚠️  No optimal data source found, using Polygon REST (high latency)")
            return DataSourceType.POLYGON_REST
    
    async def initialize(self):
        """Initialize data source and components"""
        self.data_source = DataSourceManager(self.source_type, self.config)
        await self.data_source.initialize()
        
        # Register data callback
        self.data_source.register_callback(self.on_market_data)
    
    def on_market_data(self, data: Dict):
        """Handle incoming market data"""
        # Process market data and generate signals
        symbol = data.get('symbol')
        price = data.get('price') or data.get('last')
        
        if symbol and price:
            logger.debug(f"{symbol}: ${price:.2f}")
            # TODO: Feed to engines and generate signals
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        try:
            await self.initialize()
            
            logger.info("🚀 ARES-7 v73 started")
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        self.running = False
        
        if self.data_source:
            await self.data_source.close()
        
        logger.info("✅ Shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ARES-7 v73 Trading System')
    parser.add_argument('--mode', choices=['BACKTEST', 'PAPER', 'LIVE'], 
                       default='BACKTEST', help='Trading mode')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--data-source', type=str, 
                       choices=[s.value for s in DataSourceType],
                       help='Force specific data source')
    
    args = parser.parse_args()
    
    # Load config
    config = {
        'mode': args.mode,
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        
        # API keys (from environment)
        'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        'alpaca_api_key': os.getenv('ALPACA_API_KEY'),
        'alpaca_api_secret': os.getenv('ALPACA_API_SECRET'),
        
        # Redis config
        'redis_host': os.getenv('REDIS_HOST'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        
        # Kafka config
        'kafka_brokers': os.getenv('KAFKA_BROKERS', '').split(',') if os.getenv('KAFKA_BROKERS') else None,
        
        # IBKR config
        'ibkr_host': os.getenv('IBKR_HOST'),
        'ibkr_port': int(os.getenv('IBKR_PORT', 7497)),
    }
    
    # Override data source if specified
    if args.data_source:
        config['force_data_source'] = DataSourceType(args.data_source)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ares_main.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run orchestrator
    orchestrator = AresMainOrchestrator(config)
    
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

if __name__ == "__main__":
    main()
