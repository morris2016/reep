# Advanced ML Binance Trading Application

This project provides an advanced, GUI-based trading bot for Binance that leverages machine learning techniques. The application is written in Python and uses PyQt5 for the interface. It integrates multiple technical indicators and supports testnet trading for safe experimentation.

## Features
- Real-time Binance market data integration
- Multiple ML models (Random Forest, XGBoost, Advanced Ensemble, rule-based fallback)
- Council ensemble model combining Random Forest and Logistic Regression
- Over 25 technical indicators from the `ta` library
- Professional dark-themed UI with live updates
- Real-time price chart widget visualizing recent prices
- Candlestick chart for OHLC visualization
- Export of signal history and model persistence
- Manual trading interface for quick order execution
 - Optional auto trading mode to automatically execute ML signals
 - Configurable confidence threshold for safer auto trading
- Trade history table showing executed manual and auto orders
- Trade history persisted to a local SQLite database with CSV export

- Built-in "Copy Source" button for easy code sharing
- Detailed authentication error messages for easier API troubleshooting
- Optional advanced risk management and regime detection modules
- Experimental online learning optimizer for real-time adaptation
- Built-in testnet support for safe development and testing
- Advanced risk management and market regime detection utilities

These features are summarized in the entry point docstring of [`trading_ui.py`](./trading_ui.py) around the `if __name__ == "__main__":` block.

## Setup
1. Ensure **Python 3.7+** is installed.
2. Install required dependencies:
   ```bash
   pip install PyQt5 pandas numpy scikit-learn ta requests
   # Optional but recommended
   pip install xgboost lightgbm joblib
   ```
3. Clone this repository and navigate to the project folder.

## Running the Application
Start the application with:
```bash
python reep.py
```
The `main()` function defined in [`trading_ui.py`](./trading_ui.py) performs dependency checks and launches the PyQt interface. Configure your Binance API credentials in the UI (use the Binance **testnet** initially) and select your preferred ML model to start trading.

## Notes
- Always test with the Binance testnet before trading live funds.
- This project is for educational purposes and does not constitute financial advice.

## License
This project is licensed under the [MIT License](LICENSE).

