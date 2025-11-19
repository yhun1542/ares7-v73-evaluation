"""
Improved Database Manager with separated backtest and live trading schemas
"""
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Database types"""
    BACKTEST = "backtest"
    LIVE = "live"

class DatabaseManager:
    """Manages separate databases for backtest and live trading"""
    
    def __init__(self, db_dir: str = "./db"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate database paths
        self.backtest_db = self.db_dir / "backtest.sqlite"
        self.live_db = self.db_dir / "live_trading.sqlite"
        
        # Initialize both databases
        self._init_backtest_db()
        self._init_live_db()
        
        logger.info(f"✅ DatabaseManager initialized")
        logger.info(f"  Backtest DB: {self.backtest_db}")
        logger.info(f"  Live DB: {self.live_db}")
    
    def _init_backtest_db(self):
        """Initialize backtest database schema"""
        conn = sqlite3.connect(str(self.backtest_db))
        cursor = conn.cursor()
        
        # Backtest runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                config TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Backtest trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                commission REAL,
                slippage REAL,
                engine TEXT,
                signal_strength REAL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            )
        ''')
        
        # Backtest metrics table (time series)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_return REAL,
                cumulative_return REAL,
                drawdown REAL,
                volatility REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bt_trades_symbol ON backtest_trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bt_metrics_run ON backtest_metrics(run_id)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Backtest database schema initialized")
    
    def _init_live_db(self):
        """Initialize live trading database schema"""
        conn = sqlite3.connect(str(self.live_db))
        cursor = conn.cursor()
        
        # Live trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                filled_quantity REAL DEFAULT 0,
                filled_avg_price REAL,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                filled_at TEXT,
                cancelled_at TEXT,
                commission REAL,
                slippage REAL,
                pnl REAL,
                engine TEXT,
                signal_strength REAL,
                metadata TEXT
            )
        ''')
        
        # Live positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                quantity REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0,
                last_updated TEXT NOT NULL
            )
        ''')
        
        # Live portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_portfolio_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_pnl REAL,
                daily_pnl_pct REAL,
                total_pnl REAL,
                total_pnl_pct REAL,
                num_positions INTEGER,
                metadata TEXT
            )
        ''')
        
        # Live risk events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_risk_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                symbol TEXT,
                description TEXT NOT NULL,
                action_taken TEXT,
                metadata TEXT
            )
        ''')
        
        # Live system logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_system_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_symbol ON live_trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_status ON live_trades(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_time ON live_trades(submitted_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_snapshots_time ON live_portfolio_snapshots(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_risk_time ON live_risk_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_logs_time ON live_system_logs(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Live trading database schema initialized")
    
    # ===== BACKTEST METHODS =====
    
    def save_backtest_run(self, run_data: Dict[str, Any]) -> str:
        """Save backtest run summary"""
        conn = sqlite3.connect(str(self.backtest_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_runs 
            (run_id, start_date, end_date, initial_capital, final_capital, 
             total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, 
             config, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_data['run_id'],
            run_data['start_date'],
            run_data['end_date'],
            run_data['initial_capital'],
            run_data['final_capital'],
            run_data['total_return'],
            run_data.get('sharpe_ratio'),
            run_data.get('max_drawdown'),
            run_data.get('win_rate'),
            run_data.get('total_trades', 0),
            json.dumps(run_data.get('config', {})),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Saved backtest run: {run_data['run_id']}")
        return run_data['run_id']
    
    def save_backtest_trade(self, trade_data: Dict[str, Any]):
        """Save backtest trade"""
        conn = sqlite3.connect(str(self.backtest_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_trades 
            (run_id, symbol, entry_time, exit_time, side, entry_price, exit_price,
             quantity, pnl, pnl_pct, commission, slippage, engine, signal_strength, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['run_id'],
            trade_data['symbol'],
            trade_data['entry_time'],
            trade_data.get('exit_time'),
            trade_data['side'],
            trade_data['entry_price'],
            trade_data.get('exit_price'),
            trade_data['quantity'],
            trade_data.get('pnl'),
            trade_data.get('pnl_pct'),
            trade_data.get('commission', 0),
            trade_data.get('slippage', 0),
            trade_data.get('engine'),
            trade_data.get('signal_strength'),
            json.dumps(trade_data.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    # ===== LIVE TRADING METHODS =====
    
    def save_live_trade(self, trade_data: Dict[str, Any]) -> int:
        """Save live trade"""
        conn = sqlite3.connect(str(self.live_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO live_trades 
            (order_id, symbol, side, order_type, quantity, price, status, 
             submitted_at, engine, signal_strength, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['order_id'],
            trade_data['symbol'],
            trade_data['side'],
            trade_data['order_type'],
            trade_data['quantity'],
            trade_data.get('price'),
            trade_data['status'],
            trade_data['submitted_at'],
            trade_data.get('engine'),
            trade_data.get('signal_strength'),
            json.dumps(trade_data.get('metadata', {}))
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"✅ Saved live trade: {trade_data['order_id']}")
        return trade_id
    
    def update_live_trade(self, order_id: str, updates: Dict[str, Any]):
        """Update live trade status"""
        conn = sqlite3.connect(str(self.live_db))
        cursor = conn.cursor()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [order_id]
        
        cursor.execute(f'''
            UPDATE live_trades 
            SET {set_clause}
            WHERE order_id = ?
        ''', values)
        
        conn.commit()
        conn.close()
    
    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """Save live portfolio snapshot"""
        conn = sqlite3.connect(str(self.live_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO live_portfolio_snapshots 
            (timestamp, total_value, cash, positions_value, daily_pnl, daily_pnl_pct,
             total_pnl, total_pnl_pct, num_positions, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot['timestamp'],
            snapshot['total_value'],
            snapshot['cash'],
            snapshot['positions_value'],
            snapshot.get('daily_pnl'),
            snapshot.get('daily_pnl_pct'),
            snapshot.get('total_pnl'),
            snapshot.get('total_pnl_pct'),
            snapshot.get('num_positions', 0),
            json.dumps(snapshot.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def log_risk_event(self, event: Dict[str, Any]):
        """Log risk event"""
        conn = sqlite3.connect(str(self.live_db))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO live_risk_events 
            (timestamp, event_type, severity, symbol, description, action_taken, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event['timestamp'],
            event['event_type'],
            event['severity'],
            event.get('symbol'),
            event['description'],
            event.get('action_taken'),
            json.dumps(event.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
        logger.warning(f"⚠️  Risk event: {event['event_type']} - {event['description']}")
    
    def get_live_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent live trades"""
        conn = sqlite3.connect(str(self.live_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM live_trades 
            ORDER BY submitted_at DESC 
            LIMIT ?
        ''', (limit,))
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    
    def get_backtest_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest run summary"""
        conn = sqlite3.connect(str(self.backtest_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM backtest_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database manager
    db = DatabaseManager(db_dir="./db")
    
    # Test backtest
    run_data = {
        'run_id': 'test_run_001',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': 100000.0,
        'final_capital': 115000.0,
        'total_return': 0.15,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.08,
        'win_rate': 0.62,
        'total_trades': 150
    }
    db.save_backtest_run(run_data)
    
    # Test live trade
    trade_data = {
        'order_id': 'ORD_001',
        'symbol': 'AAPL',
        'side': 'BUY',
        'order_type': 'LIMIT',
        'quantity': 100,
        'price': 180.50,
        'status': 'SUBMITTED',
        'submitted_at': datetime.now().isoformat(),
        'engine': 'momentum'
    }
    db.save_live_trade(trade_data)
    
    print("✅ Database test complete")
