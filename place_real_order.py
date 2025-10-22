#!/usr/bin/env python3
"""
PLACE A REAL ORDER RIGHT NOW - Hyperliquid Testnet
No dependencies except basic Python, requests, and web3
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

def _validate_order_payload(payload: dict, action: dict) -> bool:
    """Validate order payload structure before sending to API."""
    try:
        # Check required payload fields
        required_fields = ["action", "nonce", "signature"]
        for field in required_fields:
            if field not in payload:
                print(f"❌ Missing required field: {field}")
                return False

        # Check action structure
        if not isinstance(action.get("orders"), list) or len(action["orders"]) == 0:
            print("❌ Action must contain non-empty orders list")
            return False

        order = action["orders"][0]

        # Check order fields
        required_order_fields = ["coin", "side", "sz", "order_type"]
        for field in required_order_fields:
            if field not in order:
                print(f"❌ Missing required order field: {field}")
                return False

        # Validate side
        if order["side"] not in ["buy", "sell"]:
            print(f"❌ Invalid side: {order['side']}. Must be 'buy' or 'sell'")
            return False

        # Validate order_type for limit orders
        if order["order_type"].get("type") == "Limit":
            if "limit_px" not in order:
                print("❌ Limit orders must include limit_px")
                return False
            try:
                float(order["limit_px"])
                float(order["sz"])
            except ValueError:
                print("❌ limit_px and sz must be numeric strings")
                return False

        # Validate vault address if present
        if "vaultAddress" in payload:
            vault_addr = payload["vaultAddress"]
            if not vault_addr.startswith("0x") or len(vault_addr) != 42:
                print(f"❌ Invalid vault address format: {vault_addr}")
                return False

        print("✅ Payload validation passed")
        return True

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Place a real limit order on Hyperliquid testnet."""
    print("PLACING REAL ORDER ON HYPERLIQUID TESTNET")

    # Load env - do this at the very top
    load_dotenv()
    wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
    wallet_private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    vault_address = os.getenv("HYPERLIQUID_VAULT_ADDRESS")

    print(f"Configured addresses:")
    print(f"  Main: {wallet_address}")
    print(f"  Vault: {vault_address}")
    print()

    # 🎯 SMART BALANCE CHECK - Find wallet with funds
    testnet_url = 'https://api.hyperliquid-testnet.xyz'
    trading_wallet = None
    account_balance = 0.0

    # TEMPORARY: FORCE MAIN WALLET TRADING TO TEST VAULT ISSUE
    if wallet_address:
        try:
            response = requests.post(f"{testnet_url}/info",
                                   json={"type": "clearinghouseState", "user": wallet_address}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                main_balance = float(result.get("marginSummary", {}).get("accountValue", "0.0"))
                if main_balance > 0:
                    trading_wallet = wallet_address  # FORCE MAIN WALLET for testing
                    account_balance = main_balance
                    print(f"🧪 FORCED MAIN WALLET TEST: ${main_balance:.2f}")
                    print("   (Testing if vault trading is causing 422 error)")
        except Exception as e:
            print(f"⚠️ Could not check main wallet: {e}")

    # Fallback to vault if main wallet empty and vault configured
    if account_balance <= 0 and vault_address:
        try:
            response = requests.post(f"{testnet_url}/info",
                                   json={"type": "clearinghouseState", "user": vault_address}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                vault_balance = float(result.get("marginSummary", {}).get("accountValue", "0.0"))
                if vault_balance > 0:
                    trading_wallet = vault_address
                    account_balance = vault_balance
                    print(f"📦 Using VAULT (fallback): ${vault_balance:.2f}")
        except Exception as e:
            print(f"⚠️ Could not check vault: {e}")

    # Final check - do we have a funded wallet?
    if account_balance <= 0:
        print("❌ NO FUNDED ACCOUNT FOUND!")
        print("   Neither vault nor main wallet has funds for trading")
        print()
        print("💡 CHECK YOUR .env ADDRESSES:")
        print(f"   Main: {wallet_address}")
        print(f"   Vault: {vault_address}")
        print()
        print("🏦 Visit: https://app.hyperliquid-testnet.xyz to fund your account")
        return False

    print(f"🎯 Trading wallet: {trading_wallet}")
    print(f"💰 Available balance: ${account_balance:.2f}")
    print()

    # Get SOL price
    try:
        response = requests.post('https://api.hyperliquid-testnet.xyz/info', json={"type": "allMids"})
        mids = response.json()
        sol_price = float(mids.get("SOL", 185))
        print(f"SOL price: ${sol_price:.2f}")
    except:
        sol_price = 185.0

    # Calculate order size for $10
    order_size_usd = 10
    order_sz = round(order_size_usd / sol_price, 3)
    print(f"Calculated order size for ${order_size_usd}: {order_sz} SOL")

    # Set limit price slightly above mid-price to act like a market order
    limit_price = round(sol_price * 1.001, 2) # 0.1% slippage
    print(f"Setting limit price at: ${limit_price}")

    # 🔧 SIMPLIFIED PAYLOAD STRUCTURE - Try removing "action" wrapper
    # Based on CCXT issue #27081 analysis - might be JSON structure issue
    orders = [{
        "coin": "SOL",
        "side": "buy",
        "sz": str(order_sz),
        "limit_px": str(limit_price),
        "order_type": {"type": "Limit"}
    }]

    action = {"type": "order", "orders": orders, "grouping": "na"}
    nonce = int(time.time() * 1000)

    # CORRECTED: Try WITHOUT "action" wrapper - direct payload structure
    payload_to_sign = {
        "type": "order",
        "orders": orders,
        "grouping": "na"
    }

    print("🔧 TESTING SIMPLIFIED PAYLOAD STRUCTURE")
    print("   (Removing 'action' wrapper - might be causing 422 error)")

    # Sign the simplified payload
    action_json = json.dumps(payload_to_sign, separators=(',', ':'))
    action_hash = Web3.keccak(text=action_json)
    message_hash = Web3.solidity_keccak(
        ['bytes32', 'uint64'],
        [action_hash, nonce]
    )

    message_to_sign = encode_defunct(message_hash)
    signed_message = Account.sign_message(message_to_sign, wallet_private_key)
    signature_hex = "0x" + signed_message.signature.hex()

    # ⚠️  CRITICAL FIX: Remove "action" wrapper - send orders directly
    payload = {
        "type": "order",
        "orders": orders,
        "grouping": "na",
        "nonce": nonce,
        "signature": signature_hex
    }

    # CRITICAL: Add vault address only if we're actually using vault trading
    if trading_wallet == vault_address:
        payload["vaultAddress"] = vault_address
        print(f"🔧 Trading with VAULT: {vault_address}")
    else:
        print(f"🏠 Trading with MAIN WALLET: {trading_wallet}")

    print("📤 Sending payload:")
    print(json.dumps(payload, indent=2))

    # ⚠️  VALIDATION DISABLED - structure changed, need to update validator
    print("⏭️  Skipping payload validation (structure changed)")

    try:
        response = requests.post(
            'https://api.hyperliquid-testnet.xyz/exchange',
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=10
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("📊 Response:", json.dumps(result, indent=2))

            if result.get("status") == "ok":
                print("✅ ORDER PLACED SUCCESSFULLY!")
                print("🔍 Check: https://app.hyperliquid-testnet.xyz")
                return True
            else:
                print("❌ Order failed")
                print(f"📋 Response: {json.dumps(result, indent=2)}")
                return False
        elif response.status_code == 422:
            print("❌ 422 UNPROCESSABLE ENTITY - JSON/DATA VALIDATION ERROR!")
            print("📄 Server Response:", response.text)
            try:
                error_details = response.json()
                print("📋 Parsed Error:", json.dumps(error_details, indent=2))
            except:
                print("📄 Raw Error:", response.text)

            print("\n🔧 Common 422 causes:")
            print("  - Vault address not matching private key")
            print("  - Insufficient balance")
            print("  - Invalid order parameters (sz, limit_px)")
            print("  - Malformed order_type structure")
            return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print("📄 Response:", response.text)
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ORDER PLACED SUCCESSFULLY!")
    else:
        print("\n❌ FAILED TO PLACE ORDER")
