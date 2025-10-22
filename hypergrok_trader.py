#!/usr/bin/env python3
"""
HyperGrok Trading Bot - Fixed Version
AI-powered automated trading on Hyperliquid using xAI Grok for decision making.
Uses WebSocket connections for real-time market data streaming to bypass API rate limits.
"""

import asyncio
import json
import os
import sys
import logging
import requests
import websockets
import time
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Callable
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
import openai  # For Grok API integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class HyperliquidWebSocketClient:
    """WebSocket client for real-time Hyperliquid market data - eliminates rate limits."""

    def __init__(self, testnet: bool = True):
        self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws" if testnet else "wss://api.hyperliquid.xyz/ws"
        self.market_data: Dict[str, Dict] = {}
        self.callbacks: list[Callable[[dict], None]] = []

    def add_callback(self, callback: Callable[[dict], None]):
        """Add callback function for market data updates."""
        self.callbacks.append(callback)

    def _notify_callbacks(self, data: dict):
        """Send market data to all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def subscribe_to_market_data(self, coins: list[str] = ["SOL"]):
        """Connect to WebSocket and subscribe to real-time market data."""
        try:
            logger.info(f"🔌 Connecting to Hyperliquid WebSocket: {self.ws_url}")
            async with websockets.connect(self.ws_url) as websocket:
                logger.info(f"✅ WebSocket connection established - no REST API rate limits!")

                # Subscribe to level 2 orderbook data (individual coin subscriptions)
                for coin in coins:
                    subscribe_msg = {
                        "method": "subscribe",
                        "subscription": {
                            "type": "l2Book",
                            "coin": coin
                        }
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"📊 Subscribed to real-time L2 orderbook for: {coin}")

                    # Subscribe to trades for this coin
                    trades_sub = {
                        "method": "subscribe",
                        "subscription": {
                            "type": "trades",
                            "coin": coin
                        }
                    }
                    await websocket.send(json.dumps(trades_sub))
                    logger.info(f"📈 Subscribed to trades for: {coin}")

                # Listen for continuous market data updates
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_market_update(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def _handle_market_update(self, data: dict):
        """Process incoming real-time market data."""
        try:
            if data.get("channel") == "l2Book" and data.get("data"):
                book_data = data["data"]
                coin = book_data.get("coin")

                if coin and "levels" in book_data:
                    levels = book_data["levels"]
                    if isinstance(levels, list) and len(levels) >= 2:
                        bids = levels[0]  # Best bids
                        asks = levels[1]  # Best asks

                        # Extract top prices
                        bid_price = float(bids[0]["px"]) if bids and len(bids) > 0 and "px" in bids[0] else 0
                        ask_price = float(asks[0]["px"]) if asks and len(asks) > 0 and "px" in asks[0] else 0

                        if bid_price > 0 and ask_price > 0:
                            mid_price = (bid_price + ask_price) / 2
                            spread = ask_price - bid_price

                            # Store market data
                            self.market_data[coin] = {
                                "best_bid": bid_price,
                                "best_ask": ask_price,
                                "mid_price": mid_price,
                                "spread": spread,
                                "timestamp": time.time()
                            }

                            # Send to callbacks for real-time processing
                            market_update = {
                                "coin": coin,
                                "best_bid": bid_price,
                                "best_ask": ask_price,
                                "mid_price": mid_price,
                                "spread": spread,
                                "timestamp": time.time()
                            }
                            self._notify_callbacks(market_update)

            elif data.get("channel") == "trades" and data.get("data"):
                # Process trade updates
                for trade in data["data"]:
                    coin = trade.get("coin")
                    if coin:
                        trade_update = {
                            "type": "trade",
                            "coin": coin,
                            "px": float(trade.get("px", 0)),
                            "sz": float(trade.get("sz", 0)),
                            "side": trade.get("side"),
                            "time": trade.get("time"),
                            "hash": trade.get("hash"),
                            "timestamp": time.time()
                        }
                        self._notify_callbacks(trade_update)

        except Exception as e:
            logger.error(f"Market update processing error: {e}")

    def get_current_price(self, coin: str = "SOL") -> Optional[float]:
        """Get current mid price from WebSocket data (zero API calls required)."""
        market_info = self.market_data.get(coin)
        if market_info:
            return market_info.get("mid_price")
        return None

    def get_market_data(self, coin: str = "SOL") -> Optional[Dict]:
        """Get full market data for a coin."""
        return self.market_data.get(coin)

class HyperliquidAPI:
    """REST API for order placement only - market data handled via WebSocket."""

    def __init__(self, api_url: str, wallet_address: str, private_key: str, vault_address: str = None):
        self.api_url = api_url.rstrip('/')
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.vault_address = vault_address
        self.session = requests.Session()
        self.headers = {'Content-Type': 'application/json'}

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> Dict[str, Any]:
        """Make HTTP request to Hyperliquid API."""
        url = f"{self.api_url}{endpoint}"
        try:
            if method == 'GET':
                response = self.session.get(url, headers=self.headers, params=data, timeout=10)
            else:
                response = self.session.post(url, headers=self.headers, json=data, timeout=10)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_perpetuals_metadata(self) -> Dict[str, Any]:
        """Get metadata for all perpetual contracts."""
        data = {"type": "meta"}
        return self._make_request('POST', '/info', data)

    def get_user_state(self, user_address: str) -> Dict[str, Any]:
        """Get user positions and balances."""
        data = {"type": "clearinghouseState", "user": user_address}
        return self._make_request('POST', '/info', data)

    def place_order(self, coin: str, is_buy: bool, sz: float, limit_px: float = None,
                   order_type: str = "Limit", reduce_only: bool = False) -> Dict[str, Any]:
        """Place properly signed order."""
        nonce = int(time.time() * 1000)

        order = {
            "coin": coin,
            "side": "buy" if is_buy else "sell",
            "sz": str(sz),
            "order_type": {"type": order_type}
        }

        if limit_px is not None:
            order["limit_px"] = str(limit_px)
        if reduce_only:
            order["reduce_only"] = reduce_only

        action = {"type": "order", "orders": [order], "grouping": "na"}
        action_json = json.dumps(action, separators=(',', ':'))
        action_hash = Web3.keccak(text=action_json)
        message_hash = Web3.solidity_keccak(
            ['bytes32', 'uint64'],
            [action_hash, nonce]
        )

        message_to_sign = encode_defunct(message_hash)
        signed_message = Account.sign_message(message_to_sign, self.private_key)
        signature_hex = "0x" + signed_message.signature.hex()

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": signature_hex
        }

        if self.vault_address and self.vault_address != self.wallet_address:
            payload["vaultAddress"] = self.vault_address

        return self._make_request('POST', '/exchange', payload)

class HyperGrokTrader:
    """Trading bot for testing Hyperliquid trades with WebSocket market data."""

    def __init__(self):
        # Load configuration
        self.api_url = os.getenv('HYPERLIQUID_API_URL')
        self.wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.vault_address = os.getenv('HYPERLIQUID_VAULT_ADDRESS')
        self.grok_api_key = os.getenv('XAI_API_KEY')

        if not all([self.api_url, self.wallet_address, self.private_key]):
            raise ValueError("Missing required environment variables")

        # Initialize components
        self.api = HyperliquidAPI(
            self.api_url,
            self.wallet_address,
            self.private_key,
            self.vault_address
        )

        self.ws_client = HyperliquidWebSocketClient(testnet=True)
        self.market_task = None

        # Initialize Grok AI client if available
        if self.grok_api_key:
            openai.api_key = self.grok_api_key
            openai.api_base = "https://api.x.ai/v1"
            logger.info("🤖 Grok AI integration initialized")
        else:
            logger.warning("⚠️ XAI_API_KEY not found - running without AI analysis")

        self.trading_enabled = bool(self.grok_api_key)
        self.trade_history = []

        logger.info("HyperGrok Trader initialized with WebSocket support")
        logger.info("✅ Configuration loaded successfully")

    async def get_grok_analysis(self, coin: str = "SOL") -> Optional[Dict[str, Any]]:
        """Get AI-powered trading analysis from Grok."""
        if not self.trading_enabled:
            return None

        try:
            market_data = self.ws_client.get_market_data(coin)
            if not market_data:
                return None

            current_price = market_data.get("mid_price", 0)
            spread = market_data.get("spread", 0)
            timestamp = market_data.get("timestamp", time.time())

            # Prepare context for Grok
            context = f"""
            Current market data for {coin} perpetual contract:
            - Price: ${current_price:.2f}
            - Spread: ${spread:.4f}
            - Timestamp: {time.strftime('%H:%M:%S', time.localtime(timestamp))}

            Analyze this very short-term price movement and provide a trading decision.
            Consider market psychology, momentum, and risk.

            Respond with JSON format only:
            {{
                "action": "buy" or "sell" or "hold",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation",
                "risk_level": "low" or "medium" or "high"
            }}
            """

            response = await openai.ChatCompletion.acreate(
                model="grok-beta",  # Use Grok model
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency trading assistant analyzing real-time market data. Provide concise, actionable insights."},
                    {"role": "user", "content": context}
                ],
                max_tokens=150,
                temperature=0.3  # Keep it focused and less random
            )

            result_text = response.choices[0].message.content
            logger.info(f"🤖 Grok analysis for {coin}: {result_text}")

            # Parse JSON response
            try:
                analysis = json.loads(result_text)
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Failed to parse Grok response: {result_text}")
                return None

        except Exception as e:
            logger.error(f"❌ Grok analysis failed: {e}")
            return None

    async def execute_safe_trade(self, analysis: Dict[str, Any]) -> bool:
        """Execute a safe, small test trade based on AI analysis."""
        try:
            coin = "SOL"  # Start with SOL for testing
            action = analysis.get("action", "hold")
            confidence = analysis.get("confidence", 0)
            reason = analysis.get("reasoning", "No reasoning provided")

            if action == "hold" or confidence < 0.6:
                logger.info("⏸️  Holding position - low confidence or no clear signal")
                return False

            # Very small trade for safety (0.01 contracts)
            trade_size = 0.01
            is_buy = action == "buy"

            current_price = self.ws_client.get_current_price(coin)
            if not current_price:
                logger.warning("⚠️ No current price available - skipping trade")
                return False

            logger.info(f"🎯 Grok recommends {coin} {action.upper()} with {confidence:.1%} confidence")
            logger.info(f"💡 Reasoning: {reason}")
            logger.info(f"📊 Current price: ${current_price:.2f}")
            logger.info(f"💰 Executing {trade_size} {coin} {action} order at market")

            try:
                # Place limit order slightly better than mid price
                if is_buy:
                    limit_price = current_price * 0.9995  # Buy 0.05% below mid
                else:
                    limit_price = current_price * 1.0005  # Sell 0.05% above mid

                result = self.api.place_order(
                    coin=coin,
                    is_buy=is_buy,
                    sz=trade_size,
                    limit_px=limit_price,
                    order_type="Limit"
                )

                logger.info("✅ ORDER PLACED SUCCESSFULLY!")
                logger.info(f"📋 Response: {json.dumps(result, indent=2)}")

                # Record trade in history
                self.trade_history.append({
                    "timestamp": time.time(),
                    "coin": coin,
                    "action": action,
                    "size": trade_size,
                    "price": limit_price,
                    "grok_confidence": confidence,
                    "grok_reasoning": reason,
                    "api_response": result
                })

                return True

            except Exception as order_error:
                logger.error(f"❌ Trade execution failed: {order_error}")
                return False

        except Exception as e:
            logger.error(f"❌ Safe trade execution error: {e}")
            return False

    async def start_automated_trading(self, duration_minutes: int = 5):
        """Start AI-powered automated trading loop."""
        logger.info(f"🤖 Starting AI-powered automated trading for {duration_minutes} minutes...")
        logger.info("🎯 Trading SOL perpetual contracts using Grok AI analysis")

        # Start WebSocket market data
        self.ws_client.add_callback(self.on_market_update)
        self.market_task = asyncio.create_task(
            self.ws_client.subscribe_to_market_data(["SOL"])
        )

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_analysis = 0
        analysis_interval = 60  # Analyze every 60 seconds

        logger.info("⏰ Trading session started - monitoring market conditions...")

        try:
            while time.time() < end_time:
                current_time = time.time()

                # Get AI analysis every interval (not on every price update)
                if current_time - last_analysis > analysis_interval:
                    logger.info("🔍 Requesting AI analysis from Grok...")
                    analysis = await self.get_grok_analysis("SOL")

                    if analysis:
                        success = await self.execute_safe_trade(analysis)
                        if success:
                            logger.info("🎯 Trade executed based on Grok analysis")
                        else:
                            logger.info("⏸️  No trade executed")

                    last_analysis = current_time

                # Brief pause to prevent overwhelming the exchange
                await asyncio.sleep(5)

            logger.info("🏁 Trading session completed")
            logger.info(f"📊 Total trades executed: {len(self.trade_history)}")

        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
        finally:
            if self.market_task:
                self.market_task.cancel()
                logger.info("🔌 WebSocket connection closed")

    def on_market_update(self, data: dict):
        """Handle market data updates during trading."""
        if data.get("type") == "trade":
            px = data.get("px")
            sz = data.get("sz")
            side = "SELL" if data.get("side") == "A" else "BUY"
            logger.info(f"💰 SOL Trade: {side} {sz} @ ${px:.2f}")
        else:
            # Orderbook update - occasional logging
            mid_price = data.get("mid_price")
            spread = data.get("spread")
            if time.time() % 30 < 1:  # Log roughly every 30 seconds
                logger.info(f"📊 SOL @ ${mid_price:.2f} (Spread: ${spread:.4f})")

    async def test_websocket_connection(self) -> bool:
        """Test WebSocket data streaming."""
        logger.info("🧪 Testing WebSocket connection and data streaming...")

        start_time = time.time()
        received_updates = 0

        def count_updates(data):
            nonlocal received_updates
            received_updates += 1
            if received_updates <= 3:  # Log first few updates
                logger.info(f"📈 Update #{received_updates}: {data.get('coin', 'Unknown')} @ ${data.get('mid_price', 0):.2f}")

        self.ws_client.add_callback(count_updates)

        try:
            # Start WebSocket streaming
            self.market_task = asyncio.create_task(
                self.ws_client.subscribe_to_market_data(["SOL"])
            )

            # Wait for data and check connectivity
            await asyncio.sleep(5)

            if received_updates > 0:
                logger.info("✅ WebSocket test successful! Data streaming confirmed.")
                return True
            else:
                logger.warning("⚠️ WebSocket test inconclusive - limited or no data received")
                return False

        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            return False
        finally:
            if self.market_task:
                self.market_task.cancel()

    async def run_demo(self):
        """Run demo showing WebSocket market data."""
        logger.info("🚀 Starting HyperGrok WebSocket Demo...")
        logger.info("🎯 This demonstration shows real-time market data streaming")
        logger.info("Note: No REST API calls needed - all data comes from WebSocket")

        try:
            success = await self.test_websocket_connection()
            if success:
                logger.info("🎉 WebSocket implementation successful!")
                logger.info("💡 You can now build trading strategies using self.get_current_market_price()")
            else:
                logger.warning("⚠️ WebSocket connection needs improvement")

            return success
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

async def main():
    """Main entry point."""
    logger.info("🚀 Starting HyperGrok Trading Bot...")

    try:
        trader = HyperGrokTrader()

        # First test the WebSocket connection
        logger.info("🧪 Testing WebSocket connectivity...")
        success = await trader.test_websocket_connection()

        if success:
            logger.info("✅ WebSocket test successful!")
            logger.info("🤖 Starting AI-powered automated trading...")

            # Start automated trading for 3 minutes for safety
            await trader.start_automated_trading(duration_minutes=3)
            logger.info("✅ Trading completed successfully!")
        else:
            logger.error("❌ WebSocket connection failed - aborting trading")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
