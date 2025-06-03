import sys
import datetime
import json
import hashlib
import hmac
import time
import requests
import threading
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
import ta  # Technical Analysis library
from PyQt5.QtCore import QThread, pyqtSignal


class EnhancedBinanceAPI:
    """Enhanced Binance API client with robust error handling and historical data"""
    
    def __init__(self, api_key="", api_secret=""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.testnet_url = "https://testnet.binance.vision"
        self.use_testnet = True
        
        self.config = {
            "key_type": "HMAC-SHA-256",
            "description": "Enhanced ML Trading Bot"
        }

    def place_order(self, symbol, side, quantity, order_type="MARKET", price=None):
        """Place a simple order on Binance."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if params["type"] == "LIMIT" and price is not None:
            params["timeInForce"] = "GTC"
            params["price"] = price
        return self._send_request("POST", "/api/v3/order", params=params, signed=True)
    
    def get_server_time(self):
        """Get Binance server time with error handling"""
        try:
            url = f"{self.testnet_url if self.use_testnet else self.base_url}/api/v3/time"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()['serverTime']
        except Exception as e:
            print(f"Error getting server time: {e}")
            return int(time.time() * 1000)
    
    def _send_request(self, method, endpoint, params=None, signed=False):
        """Enhanced request handler with comprehensive error handling"""
        url = f"{self.testnet_url if self.use_testnet else self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        
        if signed and (not self.api_key or not self.api_secret):
            return {"error": "API credentials required for signed request"}
            
        if signed:
            if params is None:
                params = {}
            params['timestamp'] = self.get_server_time()
            params['recvWindow'] = 5000
            
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, data=params, headers=headers, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}"}
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection failed"}
        except requests.exceptions.HTTPError as e:
            try:
                error_data = e.response.json()
                msg = error_data.get('msg', 'Unknown error')
                if 'API-key' in msg or 'permissions' in msg:
                    return {"error": "Authentication failed: Invalid API key or permissions"}
                return {"error": f"API Error: {msg}"}
            except Exception:
                return {"error": f"HTTP Error {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_account_info(self):
        """Get account information"""
        return self._send_request('GET', "/api/v3/account", signed=True)
    
    def get_ticker_price(self, symbol="BTCUSDT"):
        """Get current price for a symbol"""
        params = {'symbol': symbol}
        return self._send_request('GET', "/api/v3/ticker/price", params=params)
    
    def get_24hr_ticker(self, symbol="BTCUSDT"):
        """Get 24hr ticker statistics"""
        params = {'symbol': symbol}
        return self._send_request('GET', "/api/v3/ticker/24hr", params=params)
    
    def get_historical_klines(self, symbol="BTCUSDT", interval="1m", limit=500):
        """Fetch historical candlestick data for ML training"""
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = self._send_request('GET', "/api/v3/klines", params=params)
        
        if "error" in response:
            return response
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(response, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_volume', 'Trades', 'Taker_buy_base', 
                'Taker_buy_quote', 'Ignore'
            ])
            
            # Convert timestamps and numeric columns
            df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            return df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Open_time')
            
        except Exception as e:
            return {"error": f"Data processing error: {str(e)}"}

class TechnicalAnalyzer:
    """Advanced technical analysis and feature engineering"""
    
    def __init__(self):
        self.feature_names = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram', 'rsi', 'stoch_k', 'stoch_d',
            'bb_upper', 'bb_lower', 'bb_position', 'bb_width', 'williams_r',
            'volume_sma', 'volume_ratio', 'obv', 'price_change_1', 'price_change_5',
            'price_change_10', 'hl_spread', 'volatility', 'atr', 'cci'
        ]
        
    def calculate_features(self, df):
        """Calculate comprehensive technical indicators"""
        if len(df) < 50:  # Need sufficient data for indicators
            return None
            
        features = {}
        
        try:
            # Moving Averages
            features['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10).iloc[-1]
            features['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
            features['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1]
            features['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12).iloc[-1]
            features['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26).iloc[-1]
            features['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(df['Close'])
            macd_signal_line = ta.trend.macd_signal(df['Close'])
            features['macd'] = macd_line.iloc[-1] if not macd_line.empty else 0
            features['macd_signal'] = macd_signal_line.iloc[-1] if not macd_signal_line.empty else 0
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Momentum Indicators
            features['rsi'] = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
            
            stoch_k = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            stoch_d = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            features['stoch_k'] = stoch_k.iloc[-1] if not stoch_k.empty else 50
            features['stoch_d'] = stoch_d.iloc[-1] if not stoch_d.empty else 50
            
            williams_r = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            features['williams_r'] = williams_r.iloc[-1] if not williams_r.empty else -50
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(df['Close'])
            bb_low = ta.volatility.bollinger_lband(df['Close'])
            bb_mid = ta.volatility.bollinger_mavg(df['Close'])
            
            if not bb_high.empty and not bb_low.empty and not bb_mid.empty:
                features['bb_upper'] = bb_high.iloc[-1]
                features['bb_lower'] = bb_low.iloc[-1]
                bb_range = features['bb_upper'] - features['bb_lower']
                features['bb_position'] = (df['Close'].iloc[-1] - features['bb_lower']) / bb_range if bb_range > 0 else 0.5
                features['bb_width'] = bb_range / bb_mid.iloc[-1] if bb_mid.iloc[-1] > 0 else 0
            else:
                features['bb_upper'] = features['bb_lower'] = features['bb_position'] = features['bb_width'] = 0
            
            # Volume Indicators
            features['volume_sma'] = df['Volume'].rolling(10).mean().iloc[-1]
            features['volume_ratio'] = df['Volume'].iloc[-1] / features['volume_sma'] if features['volume_sma'] > 0 else 1
            
            obv = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            features['obv'] = obv.iloc[-1] if not obv.empty else 0
            
            # Price Changes
            features['price_change_1'] = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) >= 2 else 0
            features['price_change_5'] = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) >= 6 else 0
            features['price_change_10'] = (df['Close'].iloc[-1] - df['Close'].iloc[-11]) / df['Close'].iloc[-11] * 100 if len(df) >= 11 else 0
            
            # Volatility and Range
            features['hl_spread'] = (df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Close'].iloc[-1] * 100
            
            returns = df['Close'].pct_change().dropna()
            features['volatility'] = returns.rolling(20).std().iloc[-1] * 100 if len(returns) >= 20 else 0
            
            # Average True Range
            atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            features['atr'] = atr.iloc[-1] if not atr.empty else 0
            
            # Commodity Channel Index
            cci = ta.trend.cci(df['High'], df['Low'], df['Close'])
            features['cci'] = cci.iloc[-1] if not cci.empty else 0
            
            # Clean NaN and infinite values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0
                    
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
            
        return features

    def advanced_feature_engineering(self, df, market_data):
        """Enhanced feature engineering for 80% accuracy target"""
        try:
            df = df.copy()

            # Market microstructure features
            if {'bid_volume', 'ask_volume'}.issubset(market_data.columns):
                imbalance = (
                    market_data['bid_volume'] - market_data['ask_volume']
                ) / (market_data['bid_volume'] + market_data['ask_volume'])
                df['order_book_imbalance'] = imbalance
            else:
                df['order_book_imbalance'] = 0.0

            # Cross-asset correlations
            if 'btc_dominance' in market_data.columns:
                df['btc_dominance_change'] = market_data['btc_dominance'].pct_change()
            else:
                df['btc_dominance_change'] = 0.0

            if 'fear_greed' in market_data.columns:
                df['fear_greed_normalized'] = (market_data['fear_greed'] - 50) / 50
            else:
                df['fear_greed_normalized'] = 0.0

            # Time-series decomposition
            from scipy import signal
            df['trend_component'] = signal.detrend(df['close']) if 'close' in df.columns else 0.0
            if 'close' in df.columns:
                fourier = np.abs(np.fft.fft(df['close'].values))[: len(df) // 2]
                df['fourier_transform'] = np.mean(fourier)
            else:
                df['fourier_transform'] = 0.0

            # Sentiment features - placeholders for real data sources
            df['sentiment_score'] = 0.0
            df['whale_movement'] = 0.0

            return df
        except Exception as e:
            print(f"Error in advanced feature engineering: {e}")
            return df


class MLSignalGenerator:
    """Advanced ML-based trading signal generator with multiple model support"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram', 'rsi', 'stoch_k', 'stoch_d',
            'bb_position', 'bb_width', 'williams_r', 'volume_ratio', 'obv',
            'price_change_1', 'price_change_5', 'price_change_10', 'hl_spread',
            'volatility', 'atr', 'cci'
        ]
        self.is_trained = False
        self.training_accuracy = 0.0
        self.feature_importance = []
        
    def prepare_features(self, feature_dict):
        """Convert feature dictionary to model input array"""
        features = []
        for name in self.feature_names:
            value = feature_dict.get(name, 0)
            if pd.isna(value) or np.isinf(value):
                value = 0
            features.append(float(value))
        return np.array(features).reshape(1, -1)
    
    def _prepare_training_data(self, historical_data):
        """Enhanced training data preparation with better labeling strategy"""
        features_list = []
        labels = []
        
        analyzer = TechnicalAnalyzer()
        
        # Process multiple DataFrames if provided
        if isinstance(historical_data, list):
            all_segments = []
            for df in historical_data:
                for i in range(50, len(df) - 1, 3):  # Step by 3 for efficiency
                    segment = df.iloc[i-50:i+1]
                    if len(segment) >= 50:
                        all_segments.append((segment, df.iloc[i+1] if i+1 < len(df) else None))
        else:
            all_segments = []
            for i in range(50, len(historical_data) - 1, 3):
                segment = historical_data.iloc[i-50:i+1]
                if len(segment) >= 50:
                    all_segments.append((segment, historical_data.iloc[i+1] if i+1 < len(historical_data) else None))
        
        for segment, next_row in all_segments:
            try:
                # Calculate features
                feature_dict = analyzer.calculate_features(segment)
                if feature_dict is None:
                    continue
                    
                feature_vector = self.prepare_features(feature_dict).flatten()
                features_list.append(feature_vector)
                
                # Enhanced labeling strategy
                current_price = segment['Close'].iloc[-1]
                if next_row is not None:
                    next_price = next_row['Close']
                    price_change = (next_price - current_price) / current_price
                    
                    # Dynamic thresholds based on volatility
                    volatility = feature_dict.get('volatility', 1.0) / 100
                    threshold = max(0.003, volatility * 0.5)  # Minimum 0.3% or half the volatility
                    
                    if price_change > threshold:
                        labels.append(2)  # BUY
                    elif price_change < -threshold:
                        labels.append(0)  # SELL
                    else:
                        labels.append(1)  # HOLD
                else:
                    labels.append(1)  # Default to HOLD if no next price
                    
            except Exception as e:
                print(f"Error processing training sample: {e}")
                continue
                
        return np.array(features_list), np.array(labels)
    
    def train_model(self, historical_data, model_type="random_forest"):
        """Train ML model with enhanced validation"""
        try:
            X, y = self._prepare_training_data(historical_data)
            
            if len(X) < 100:
                print(f"Insufficient training samples: {len(X)}")
                return False
            
            print(f"Training with {len(X)} samples, {len(self.feature_names)} features")
            print(f"Label distribution: BUY: {np.sum(y==2)}, HOLD: {np.sum(y==1)}, SELL: {np.sum(y==0)}")
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scalers[model_type] = StandardScaler()
            X_train_scaled = self.scalers[model_type].fit_transform(X_train)
            X_test_scaled = self.scalers[model_type].transform(X_test)
            
            # Initialize model
            if model_type == "advanced_ensemble":
                self.models[model_type] = AdvancedEnsemble()
                self.models[model_type].fit(X_train_scaled, y_train)
            elif model_type == "random_forest":
                self.models[model_type] = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            elif model_type == "xgboost":
                try:
                    import xgboost as xgb
                    self.models[model_type] = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric='mlogloss'
                    )
                except ImportError:
                    print("XGBoost not available, using Random Forest")
                    return self.train_model(historical_data, "random_forest")
            elif model_type == "advanced_ensemble":
                self.models[model_type] = AdvancedEnsemble()
                self.models[model_type].fit(X_train_scaled, y_train)
                # Evaluate separately
                train_pred = (self.models[model_type].predict(X_train_scaled) > 0.5).astype(int)
                test_pred = (self.models[model_type].predict(X_test_scaled) > 0.5).astype(int)
                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)
                self.training_accuracy = test_accuracy
                self.feature_importance = []
                self.is_trained = True
                print(f"Training accuracy: {train_accuracy:.3f}")
                print(f"Test accuracy: {test_accuracy:.3f}")
                return True
            
            # Train model
            if model_type != "advanced_ensemble":
                self.models[model_type].fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = self.models[model_type].predict(X_train_scaled)
            test_pred = self.models[model_type].predict(X_test_scaled)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            self.training_accuracy = test_accuracy
            
            print(f"Training accuracy: {train_accuracy:.3f}")
            print(f"Test accuracy: {test_accuracy:.3f}")
            
            # Feature importance
            if hasattr(self.models[model_type], 'feature_importances_'):
                importances = self.models[model_type].feature_importances_
                self.feature_importance = list(zip(self.feature_names, importances))
                self.feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print("Top 5 most important features:")
                for name, importance in self.feature_importance[:5]:
                    print(f"  {name}: {importance:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def predict_signal(self, feature_dict, model_type="random_forest"):
        """Generate enhanced trading signal"""
        if not self.is_trained or model_type not in self.models:
            return self._fallback_signal(feature_dict)
            
        try:
            features = self.prepare_features(feature_dict)
            features_scaled = self.scalers[model_type].transform(features)

            if model_type == "advanced_ensemble":
                prob_buy = float(self.models[model_type].predict(features_scaled))
                if prob_buy > 0.55:
                    prediction = 2
                elif prob_buy < 0.45:
                    prediction = 0
                else:
                    prediction = 1
                probabilities = [1 - prob_buy, 0, prob_buy]

            else:
                prediction = self.models[model_type].predict(features_scaled)[0]
                probabilities = self.models[model_type].predict_proba(features_scaled)[0]
            
            base_confidence = int(np.max(probabilities) * 100)
            
            # Enhanced confidence calculation
            prob_spread = np.max(probabilities) - np.mean(probabilities)
            if prob_spread > 0.3:
                confidence = min(base_confidence + 15, 95)
            elif prob_spread > 0.2:
                confidence = min(base_confidence + 10, 90)
            else:
                confidence = base_confidence
            
            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            signal = signal_map.get(prediction, "HOLD")
            
            return signal, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_signal(feature_dict)
    
    def _fallback_signal(self, feature_dict):
        """Enhanced rule-based fallback system"""
        try:
            # Multi-indicator scoring system
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            rsi = feature_dict.get('rsi', 50)
            if rsi < 25:
                buy_score += 2
            elif rsi < 35:
                buy_score += 1
            elif rsi > 75:
                sell_score += 2
            elif rsi > 65:
                sell_score += 1
            
            # MACD signals
            macd_hist = feature_dict.get('macd_histogram', 0)
            if macd_hist > 0:
                buy_score += 1
            elif macd_hist < 0:
                sell_score += 1
            
            # Bollinger Band signals
            bb_position = feature_dict.get('bb_position', 0.5)
            if bb_position < 0.1:
                buy_score += 1
            elif bb_position > 0.9:
                sell_score += 1
            
            # Volume confirmation
            volume_ratio = feature_dict.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume
                if buy_score > sell_score:
                    buy_score += 1
                elif sell_score > buy_score:
                    sell_score += 1
            
            # Generate signal
            if buy_score >= 3:
                return "BUY", min(70 + buy_score * 5, 85)
            elif sell_score >= 3:
                return "SELL", min(70 + sell_score * 5, 85)
            elif buy_score > sell_score:
                return "BUY", 60
            elif sell_score > buy_score:
                return "SELL", 60
            else:
                return "HOLD", 50
                
        except Exception as e:
            print(f"Fallback signal error: {e}")
            return "HOLD", 50
    
    def get_feature_importance(self, model_type="random_forest"):
        """Get feature importance from trained model"""
        return self.feature_importance
    
    def save_model(self, filepath, model_type="random_forest"):
        """Save trained model with metadata"""
        try:
            if model_type in self.models:
                model_data = {
                    'model': self.models[model_type],
                    'scaler': self.scalers[model_type],
                    'feature_names': self.feature_names,
                    'training_accuracy': self.training_accuracy,
                    'feature_importance': self.feature_importance,
                    'model_type': model_type,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                joblib.dump(model_data, filepath)
                return True
        except Exception as e:
            print(f"Error saving model: {e}")
        return False
    
    def load_model(self, filepath):
        """Load pre-trained model with validation"""
        try:
            model_data = joblib.load(filepath)
            model_type = model_data.get('model_type', 'random_forest')
            
            self.models[model_type] = model_data['model']
            self.scalers[model_type] = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.feature_importance = model_data.get('feature_importance', [])
            self.is_trained = True
            
            print(f"Loaded model: {model_type}, Accuracy: {self.training_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
class EnhancedDataWorker(QThread):
    """Enhanced data worker with sophisticated ML integration and comprehensive error handling"""
    data_updated = pyqtSignal(dict)
    ml_signal_generated = pyqtSignal(dict)
    ml_status_updated = pyqtSignal(str)
    training_progress = pyqtSignal(str)
    training_message = pyqtSignal(str)  # Previously missing signal - now properly defined
    
    def __init__(self, binance_api, symbol="BTCUSDT"):
        super().__init__()
        self.binance_api = binance_api
        self.symbol = symbol
        self.running = False
        
        # ML components
        self.ml_generator = MLSignalGenerator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.historical_buffer = None
        self.model_type = "random_forest"
        
        # Performance tracking
        self.signal_history = []
        self.update_counter = 0
        self.last_error_time = None
        self.error_count = 0
    
    def initialize_ml_system(self):
        """Initialize ML system in background thread with comprehensive progress tracking"""
        threading.Thread(target=self._fetch_and_train, daemon=True).start()
    
    def _fetch_and_train(self):
        """Fetch historical data and train ML model with detailed progress reporting"""
        try:
            self.ml_status_updated.emit("🔄 Fetching historical data...")
            self.training_message.emit("Starting comprehensive data collection...")
            self.training_progress.emit("Fetching historical data")
            
            # Fetch multiple timeframes for robust training
            timeframes = ["5m", "15m", "1h"]
            all_data = []
            
            for i, tf in enumerate(timeframes):
                progress_msg = f"Fetching {tf} data... ({i+1}/{len(timeframes)})"
                self.training_progress.emit(f"Loaded {tf}")
                self.training_message.emit(progress_msg)
                
                df = self.binance_api.get_historical_klines(self.symbol, tf, 500)
                
                if "error" not in df and not df.empty:
                    all_data.append(df)
                    success_msg = f"✅ Loaded {tf}: {len(df)} candles"
                    self.ml_status_updated.emit(success_msg)
                    self.training_message.emit(success_msg)
                else:
                    error_msg = f"⚠️ Failed to load {tf} data"
                    self.ml_status_updated.emit(error_msg)
                    self.training_message.emit(error_msg)
            
            if all_data:
                self.historical_buffer = all_data[0]  # Use 5m as primary buffer
                
                self.training_progress.emit("Training ML model")
                self.training_message.emit("Processing features and training model...")
                self.ml_status_updated.emit("🤖 Training machine learning model...")
                
                success = self.ml_generator.train_model(all_data, self.model_type)
                
                if success:
                    accuracy = self.ml_generator.training_accuracy
                    success_msg = f"✅ Model trained! Accuracy: {accuracy:.1%}"
                    self.ml_status_updated.emit(success_msg)
                    self.training_progress.emit("Training completed")
                    self.training_message.emit(f"Model ready with {accuracy:.1%} accuracy")
                else:
                    failure_msg = "❌ Training failed - using fallback rules"
                    self.ml_status_updated.emit(failure_msg)
                    self.training_message.emit("Training failed, fallback rules active")
            else:
                no_data_msg = "❌ No historical data available"
                self.ml_status_updated.emit(no_data_msg)
                self.training_message.emit("Data collection failed - check API connection")
                
        except Exception as e:
            error_msg = f"❌ Initialization error: {str(e)}"
            self.ml_status_updated.emit(error_msg)
            self.training_message.emit(f"System error: {str(e)}")
    
    def set_model_type(self, model_type):
        """Change the ML model type with comprehensive validation"""
        old_model = self.model_type
        self.model_type = model_type
        
        if self.historical_buffer is not None and model_type != "fallback":
            self.training_message.emit(f"Switching from {old_model} to {model_type}")
            self.initialize_ml_system()
        elif model_type == "fallback":
            self.training_message.emit("Switched to rule-based fallback system")
    
    def run(self):
        """Main data processing loop with enhanced error handling and recovery"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Get current market data with comprehensive error handling
                price_data = self.binance_api.get_ticker_price(self.symbol)
                stats_data = self.binance_api.get_24hr_ticker(self.symbol)
                
                if "error" not in price_data and "error" not in stats_data:
                    # Reset error counter on successful data fetch
                    consecutive_errors = 0
                    
                    # Update historical buffer periodically
                    if self.update_counter % 12 == 0:  # Every 12 iterations (1 minute)
                        self._update_historical_buffer()
                    
                    # Generate ML signal with comprehensive feature analysis
                    if self.historical_buffer is not None and len(self.historical_buffer) >= 50:
                        features = self.technical_analyzer.calculate_features(self.historical_buffer)
                        
                        if features:
                            signal, confidence = self.ml_generator.predict_signal(features, self.model_type)
                            
                            # Enhanced signal data with comprehensive metadata
                            signal_data = {
                                'symbol': self.symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'price': float(price_data['price']),
                                'timestamp': datetime.datetime.now(),
                                'features': features,
                                'model_type': self.model_type,
                                'is_ml_trained': self.ml_generator.is_trained,
                                'update_count': self.update_counter,
                                'buffer_size': len(self.historical_buffer),
                                'error_count': self.error_count
                            }
                            
                            self.signal_history.append(signal_data)
                            if len(self.signal_history) > 100:
                                self.signal_history.pop(0)
                            
                            self.ml_signal_generated.emit(signal_data)
                    
                    # Emit regular data update with comprehensive metadata
                    combined_data = {
                        'symbol': self.symbol,
                        'price': price_data,
                        'stats': stats_data,
                        'timestamp': datetime.datetime.now(),
                        'update_count': self.update_counter,
                        'ml_active': self.ml_generator.is_trained,
                        'buffer_ready': self.historical_buffer is not None,
                        'consecutive_errors': consecutive_errors,
                        'total_errors': self.error_count
                    }
                    
                    self.data_updated.emit(combined_data)
                    self.update_counter += 1
                
                else:
                    # Handle API errors with intelligent recovery
                    consecutive_errors += 1
                    self.error_count += 1
                    
                    error_msg = price_data.get('error', stats_data.get('error', 'Unknown API error'))
                    self.ml_status_updated.emit(f"⚠️ API Error ({consecutive_errors}/{max_consecutive_errors}): {error_msg}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.ml_status_updated.emit("❌ Too many consecutive errors - entering recovery mode")
                        self.msleep(30000)  # Wait 30 seconds before retry
                        consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                error_msg = f"Data processing error: {str(e)}"
                print(f"⚠️ {error_msg}")
                self.ml_status_updated.emit(f"⚠️ {error_msg}")
                self.training_message.emit(f"Processing error: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.ml_status_updated.emit("❌ System entering recovery mode due to repeated errors")
                    self.msleep(60000)  # Wait 1 minute before retry
                    consecutive_errors = 0
            
            # Intelligent sleep timing based on system state and error count
            if consecutive_errors > 0:
                sleep_time = min(10000 + (consecutive_errors * 2000), 30000)  # Exponential backoff
            else:
                sleep_time = 5000 if self.ml_generator.is_trained else 3000
            
            self.msleep(sleep_time)
    
    def _update_historical_buffer(self):
        """Update historical buffer with latest market data and maintain optimal size"""
        try:
            latest_data = self.binance_api.get_historical_klines(self.symbol, "5m", 2)
            if "error" not in latest_data and not latest_data.empty:
                # Intelligently merge new data and maintain buffer size
                self.historical_buffer = pd.concat([self.historical_buffer, latest_data]).tail(200)
                
                # Remove any duplicate timestamps to maintain data integrity
                self.historical_buffer = self.historical_buffer[~self.historical_buffer.index.duplicated(keep='last')]
                
        except Exception as e:
            print(f"⚠️ Error updating historical buffer: {e}")
            self.training_message.emit(f"Buffer update error: {str(e)}")
    
    def start_updates(self):
        """Start the data processing thread with validation"""
        self.running = True
        self.error_count = 0
        self.start()
    
    def stop_updates(self):
        """Stop the data processing thread with proper cleanup and timeout"""
        print("⏹️ Stopping data processing thread...")
        self.running = False
        self.quit()
        
        # Wait for thread to finish with timeout
        if not self.wait(5000):  # Wait up to 5 seconds
            print("⚠️ Thread did not stop gracefully, terminating...")
            self.terminate()
            self.wait()  # Wait for termination to complete
        
        print("✅ Data processing thread stopped successfully")

