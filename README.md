# Advanced ML Binance Trading Application v2.2

This project provides an advanced, GUI-based trading bot for Binance that leverages machine learning techniques. The application is written in Python and uses PyQt5 for the interface. It integrates multiple technical indicators, supports testnet trading for safe experimentation, and now features enhanced error handling, logging, and a more robust user experience.

## Key Features
- **Real-time Binance Market Data:** Integration for live price and 24hr statistics.
- **Multiple ML Models:** Supports Random Forest, XGBoost (if installed), a custom Council Ensemble, and a rule-based fallback system.
- **Comprehensive Technical Analysis:** Utilizes over 25 technical indicators from the `ta` library for feature engineering.
- **Professional Dark-Themed UI:** Built with PyQt5, featuring live updates, clear information hierarchy, and improved user feedback.
- **Live Price Chart:** Visualizes recent price movements.
- **Persistent Trade History:** Executed trades are logged to a local SQLite database (`trades.db`) and can be exported to CSV.
- **ML Model Management:** Save, load, and retrain ML models directly from the UI. View feature importances for supported models.
- **Manual & Automated Trading:**
    - Interface for quick manual order execution (MARKET and LIMIT).
    - Optional auto-trading mode to execute trades based on ML signals meeting a configurable confidence threshold.
- **Enhanced Error Handling & Logging:** Robust error catching, informative user dialogs, and detailed logging to `trading_app.log` and console for easier debugging and monitoring.
- **Resilient API Interaction:** Implemented retry mechanisms with exponential backoff for API calls to handle transient network issues and basic rate limit responses.
- **Testnet Support:** Full support for Binance Testnet, highly recommended for initial setup and strategy testing.
- **Code Accessibility:** "Copy Source Code" button in the "About" tab for easy review of the main UI logic.

## Setup

1.  **Python:** Ensure **Python 3.8+** is installed.
2.  **Dependencies:** Install required packages. It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install PyQt5 pandas numpy scikit-learn ta requests joblib
    ```
3.  **Optional ML Libraries:** For XGBoost support:
    ```bash
    pip install xgboost
    ```
    The application will fall back to other models if XGBoost is not installed.
4.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

## Running the Application

Execute the main script:
```bash
python reep.py
```

Upon first launch, or if `trading_app.log` or `trades.db` don't exist, they will be created in the same directory as the scripts.

**Initial Configuration:**
1.  Navigate to the **⚙️ API Configuration** tab.
2.  Enter your Binance API Key and Secret. **IMPORTANT:** Start with **Testnet** credentials.
3.  Ensure the "🔗 Use Binance Testnet" checkbox is checked for initial testing.
4.  Click "💾 Save Keys" and then "📡 Test API Connection". You should see a success message.
5.  Navigate to the **📈 Live Trading** tab.
6.  Select a symbol (e.g., BTCUSDT) and an ML Model.
7.  Click "🚀 Start" to begin data fetching and ML model initialization/training.

## Logging

The application generates detailed logs in `trading_app.log` located in the application's root directory. This file records:
- Application startup and shutdown events.
- API requests and responses (errors are logged with details).
- ML model training progress and outcomes.
- Generated trading signals.
- Executed trades (manual and auto).
- UI interactions and status changes.
- Errors and warnings from all modules.

Review this log file for troubleshooting and monitoring application activity.

## Important Notes
-   **RISK WARNING:** Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. All trading decisions are your own.
-   **TESTNET FIRST:** Always use the Binance Testnet environment for testing any new configurations, models, or strategies before considering live trading.
-   **EDUCATIONAL PURPOSES:** This project is provided for educational and experimental purposes only. It does not constitute financial advice.
-   **API KEY SECURITY:** Protect your API keys. Do not share them and ensure they have appropriate permissions (e.g., enable trading, disable withdrawals if not needed for the bot's function).

## License
This project is licensed under the [MIT License](LICENSE). (Assuming `LICENSE` file exists, if not, one should be added).
