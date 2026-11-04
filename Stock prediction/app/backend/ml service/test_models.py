"""
Test script to validate models are working correctly
Run this after training to verify predictions
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import model_manager
import pandas as pd

def test_models():
    """Test all models with sample predictions"""
    
    print("\n" + "="*60)
    print("  MTR-QNet Lite Model Testing")
    print("="*60 + "\n")
    
    # Check if models are loaded
    if not model_manager.models_available():
        print("✗ FAILED: Models not loaded!")
        print("\nPlease run: python train_models.py")
        return False
    
    print("✓ All models loaded successfully\n")
    
    # Test with a few tickers
    test_tickers = [
        ("RELIANCE.NS", "Reliance Industries"),
        ("TCS.NS", "Tata Consultancy Services"),
        ("INFY.NS", "Infosys"),
    ]
    
    print("-" * 60)
    print("Testing Predictions")
    print("-" * 60 + "\n")
    
    successful = 0
    failed = 0
    
    for ticker, name in test_tickers:
        print(f"Testing: {ticker} ({name})")
        
        try:
            prediction = model_manager.predict(ticker, name)
            
            if prediction is None:
                print(f"  ✗ FAILED: Could not generate prediction\n")
                failed += 1
                continue
            
            # Validate prediction structure
            required_keys = ["symbol", "trend", "confidence", "atr", "signal", "priceRange"]
            missing_keys = [k for k in required_keys if k not in prediction]
            
            if missing_keys:
                print(f"  ✗ FAILED: Missing keys: {missing_keys}\n")
                failed += 1
                continue
            
            # Validate prediction values
            if prediction["confidence"] < 0 or prediction["confidence"] > 100:
                print(f"  ✗ FAILED: Invalid confidence: {prediction['confidence']}\n")
                failed += 1
                continue
            
            if prediction["signal"] not in ["BUY", "SELL", "HOLD"]:
                print(f"  ✗ FAILED: Invalid signal: {prediction['signal']}\n")
                failed += 1
                continue
            
            if prediction["trend"] not in ["UP", "DOWN", "NEUTRAL"]:
                print(f"  ✗ FAILED: Invalid trend: {prediction['trend']}\n")
                failed += 1
                continue
            
            # Print prediction details
            print(f"  ✓ SUCCESS")
            print(f"    Trend: {prediction['trend']}")
            print(f"    Signal: {prediction['signal']}")
            print(f"    Confidence: {prediction['confidence']:.1f}%")
            print(f"    ATR: {prediction['atr']:.2f}")
            print(f"    Price Range: {prediction['priceRange']['low']:.2f} - {prediction['priceRange']['high']:.2f}")
            print(f"    Volatility: {prediction['volatility']['risk_level']}")
            print()
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}\n")
            failed += 1
    
    # Summary
    print("-" * 60)
    print("Test Summary")
    print("-" * 60)
    print(f"Successful: {successful}/{len(test_tickers)}")
    print(f"Failed: {failed}/{len(test_tickers)}")
    
    # Overall result
    if failed == 0:
        print("\n✓ All tests passed! Models are working correctly.")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")
        return False


def test_api_integration():
    """Test FastAPI endpoints"""
    try:
        import requests
    except ImportError:
        print("\nNote: requests library not installed. Skipping API tests.")
        print("Install with: pip install requests")
        return True
    
    print("\n" + "="*60)
    print("  API Endpoint Testing")
    print("="*60 + "\n")
    
    base_url = "http://localhost:8001"
    
    # Test health endpoint
    print("Testing GET /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("  ✓ Health check passed\n")
        else:
            print(f"  ✗ Health check failed: {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print("  ℹ ML Service not running (this is normal if server not started)\n")
        return True
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        return False
    
    # Test predict endpoint
    print("Testing POST /predict...")
    try:
        payload = {"symbol": "RELIANCE.NS", "name": "Reliance Industries"}
        response = requests.post(f"{base_url}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Prediction successful")
            print(f"    Signal: {data.get('signal')}")
            print(f"    Confidence: {data.get('confidence')}%\n")
        else:
            print(f"  ✗ Prediction failed: {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print("  ℹ ML Service not running (this is normal if server not started)\n")
        return True
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  MTR-QNet Lite - Model & API Validation".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    
    # Run model tests
    model_test = test_models()
    
    # Run API tests
    api_test = test_api_integration()
    
    # Final result
    print("\n" + "="*60)
    if model_test and api_test:
        print("✓ All validation tests passed!")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("✗ Some validation tests failed")
        print("="*60 + "\n")
        sys.exit(1)
