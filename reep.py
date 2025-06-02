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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QGroupBox, QFormLayout,
    QTextEdit, QTabWidget, QMessageBox, QFileDialog,
    QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QPolygonF
from collections import deque


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
                return {"error": f"API Error: {error_data.get('msg', 'Unknown error')}"}
            except:
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
            elif model_type == "council":
                rf = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42,
                    class_weight='balanced',
                )
                lr = LogisticRegression(max_iter=200)
                rf.fit(X_train_scaled, y_train)
                lr.fit(X_train_scaled, y_train)
                self.models[model_type] = {'rf': rf, 'lr': lr}
                # Evaluate council
                def vote(pred_a, pred_b):
                    return [pa if pa==pb else pa for pa,pb in zip(pred_a,pred_b)]
                train_pred = vote(rf.predict(X_train_scaled), lr.predict(X_train_scaled))
                test_pred = vote(rf.predict(X_test_scaled), lr.predict(X_test_scaled))
                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)
                self.training_accuracy = test_accuracy
                if hasattr(rf, 'feature_importances_'):
                    importances = rf.feature_importances_
                    self.feature_importance = list(zip(self.feature_names, importances))
                    self.feature_importance.sort(key=lambda x: x[1], reverse=True)
                self.is_trained = True
                print(f"Training accuracy: {train_accuracy:.3f}")
                print(f"Test accuracy: {test_accuracy:.3f}")
                return True
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
            elif model_type == "council":
                rf = self.models[model_type]['rf']
                lr = self.models[model_type]['lr']
                pred_rf = rf.predict(features_scaled)[0]
                pred_lr = lr.predict(features_scaled)[0]
                prediction = pred_rf if pred_rf == pred_lr else pred_rf
                prob_rf = rf.predict_proba(features_scaled)[0]
                prob_lr = lr.predict_proba(features_scaled)[0]
                probabilities = (prob_rf + prob_lr) / 2
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


class PriceChartWidget(QWidget):
    """Lightweight widget to display live price history as a line chart."""

    def __init__(self, parent=None, max_points=120):
        super().__init__(parent)
        self.prices = deque(maxlen=max_points)
        self.setMinimumHeight(150)

    def add_price(self, price: float) -> None:
        """Add a new price point and trigger a repaint."""
        self.prices.append(price)
        self.update()

    def paintEvent(self, event):  # noqa: D401 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        if len(self.prices) < 2:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(rect, Qt.AlignCenter, "No data")
            return

        minimum = min(self.prices)
        maximum = max(self.prices)
        span = maximum - minimum or 1
        step = w / (len(self.prices) - 1)

        poly = QPolygonF()
        for i, price in enumerate(self.prices):
            x = i * step
            y = h - ((price - minimum) / span) * h
            poly.append(QPointF(x, y))

        painter.setPen(QPen(QColor(0, 255, 136), 2))
        painter.drawPolyline(poly)


class CandlestickChartWidget(QWidget):
    """Widget to display OHLC data as candlesticks."""

    def __init__(self, parent=None, max_candles=120):
        super().__init__(parent)
        self.candles = deque(maxlen=max_candles)
        self.setMinimumHeight(150)

    def add_candle(self, o, h, l, c):
        """Add a candlestick and repaint."""
        self.candles.append((o, h, l, c))
        self.update()

    def paintEvent(self, event):  # noqa: D401 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        if not self.candles:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(rect, Qt.AlignCenter, "No data")
            return

        high = max(c[1] for c in self.candles)
        low = min(c[2] for c in self.candles)
        span = high - low or 1
        step = w / len(self.candles)

        for i, (o, hi, lo, c) in enumerate(self.candles):
            x = i * step + step / 2
            y_open = h - ((o - low) / span) * h
            y_close = h - ((c - low) / span) * h
            y_high = h - ((hi - low) / span) * h
            y_low = h - ((lo - low) / span) * h

            color = QColor(0, 255, 136) if c >= o else QColor(255, 68, 68)
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(x, y_high), QPointF(x, y_low))

            rect_x = x - step * 0.3
            rect_w = step * 0.6
            rect_y = min(y_open, y_close)
            rect_h = abs(y_close - y_open) or 1
            painter.fillRect(rect_x, rect_y, rect_w, rect_h, color)


class BinanceTradingApp(QMainWindow):
    """Main application with comprehensive ML trading capabilities"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Advanced ML Binance Trading Bot - Real Intelligence")
        self.setGeometry(100, 100, 1600, 1000)
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.binance_api = EnhancedBinanceAPI()
        self.enhanced_data_worker = None
        
        # Trading state
        self.is_trading = False
        self.current_symbol = "BTCUSDT"
        self.current_model = "random_forest"
        
        # Performance tracking
        self.signal_performance = []
        
        # Setup UI
        self.setup_ui()
        self.apply_styling()
        
        # Initialize status
        self.update_connection_status("🔄 Ready - Configure API for live trading")
    
    def setup_ui(self):
        """Setup the complete user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout()
        
        # Create enhanced tab system
        self.tabs = QTabWidget()
        
        # Trading Tab
        self.trading_tab = QWidget()
        self.setup_trading_tab()
        self.tabs.addTab(self.trading_tab, "📈 Live Trading")
        
        # ML Configuration Tab
        self.ml_tab = QWidget()
        self.setup_ml_tab()
        self.tabs.addTab(self.ml_tab, "🤖 Machine Learning")
        
        # API Configuration Tab
        self.config_tab = QWidget()
        self.setup_config_tab()
        self.tabs.addTab(self.config_tab, "⚙️ API Configuration")
        
        # Account Tab
        self.account_tab = QWidget()
        self.setup_account_tab()
        self.tabs.addTab(self.account_tab, "💰 Account Info")
        
        # Analytics Tab
        self.analytics_tab = QWidget()
        self.setup_analytics_tab()
        self.tabs.addTab(self.analytics_tab, "📊 Analytics")

        # About Tab with source code access
        self.about_tab = QWidget()
        self.setup_about_tab()
        self.tabs.addTab(self.about_tab, "ℹ️ About")
        
        self.main_layout.addWidget(self.tabs)
        self.central_widget.setLayout(self.main_layout)
    
    def setup_trading_tab(self):
        """Setup enhanced trading interface"""
        layout = QVBoxLayout()
        
        # Enhanced status bar
        status_layout = QHBoxLayout()
        
        self.connection_status = QLabel("Connection: Disconnected")
        self.connection_status.setFont(QFont('Arial', 11, QFont.Bold))
        
        self.ml_status = QLabel("ML: Initializing...")
        self.ml_status.setFont(QFont('Arial', 11, QFont.Bold))
        
        self.update_counter_label = QLabel("Updates: 0")
        
        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.ml_status)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.update_counter_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # Enhanced controls
        controls_layout = QHBoxLayout()
        
        # Symbol selection with popular pairs
        controls_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems([
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", 
            "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
            "EOSUSDT", "TRXUSDT", "XLMUSDT", "ATOMUSDT", "VETUSDT"
        ])
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        controls_layout.addWidget(self.symbol_combo)
        
        # Model selection
        controls_layout.addWidget(QLabel("ML Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Random Forest",
            "XGBoost",
            "Advanced Ensemble",
            "Council",
            "Fallback Rules",
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        controls_layout.addWidget(self.model_combo)
        
        # Auto-retrain option
        self.auto_retrain_checkbox = QCheckBox("Auto-retrain")
        self.auto_retrain_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_retrain_checkbox)
        
        # Trading controls
        self.start_button = QPushButton("🚀 Start ML Trading")
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setEnabled(False)
        
        self.start_button.clicked.connect(self.start_ml_trading)
        self.stop_button.clicked.connect(self.stop_ml_trading)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Enhanced price display
        price_group = QGroupBox("Live Market Data")
        price_layout = QVBoxLayout()
        
        self.price_display = QLabel("💰 Live Price: Waiting for data...")
        self.price_display.setFont(QFont('Arial', 18, QFont.Bold))
        self.price_display.setAlignment(Qt.AlignCenter)
        self.price_display.setMinimumHeight(60)
        self.price_display.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                stop:0 #1e1e1e, stop:1 #2a2a2a);
            color: #00ff88; 
            border: 2px solid #444; 
            border-radius: 8px;
            padding: 10px;
        """)
        
        # Market statistics
        stats_layout = QHBoxLayout()
        
        self.stats_24h_change = QLabel("📊 24h Change: --")
        self.stats_24h_volume = QLabel("📈 Volume: --")
        self.stats_high_low = QLabel("⬆️⬇️ High/Low: --")
        self.stats_trades = QLabel("🔄 Trades: --")
        
        for widget in [self.stats_24h_change, self.stats_24h_volume, self.stats_high_low, self.stats_trades]:
            widget.setFont(QFont('Arial', 10))
        
        stats_layout.addWidget(self.stats_24h_change)
        stats_layout.addWidget(self.stats_24h_volume)
        stats_layout.addWidget(self.stats_high_low)
        stats_layout.addWidget(self.stats_trades)

        price_layout.addWidget(self.price_display)
        price_layout.addLayout(stats_layout)
        self.price_chart = PriceChartWidget()
        self.candlestick_chart = CandlestickChartWidget()
        price_layout.addWidget(self.price_chart)
        price_layout.addWidget(self.candlestick_chart)
        price_group.setLayout(price_layout)

        layout.addWidget(price_group)

        # Auto trading controls
        auto_group = QGroupBox("🤖 Auto Trading")
        auto_layout = QHBoxLayout()
        self.auto_trade_checkbox = QCheckBox("Enable Auto")
        self.auto_trade_qty = QLineEdit()
        self.auto_trade_qty.setPlaceholderText("Qty")
        auto_layout.addWidget(self.auto_trade_checkbox)
        auto_layout.addWidget(QLabel("Qty:"))
        auto_layout.addWidget(self.auto_trade_qty)
        auto_group.setLayout(auto_layout)

        layout.addWidget(auto_group)

        # Manual trading controls
        manual_group = QGroupBox("🖐️ Manual Trading")
        manual_layout = QHBoxLayout()

        self.manual_qty = QLineEdit()
        self.manual_qty.setPlaceholderText("Quantity")
        self.manual_price = QLineEdit()
        self.manual_price.setPlaceholderText("Price (for limit)")
        self.manual_order_type = QComboBox()
        self.manual_order_type.addItems(["MARKET", "LIMIT"])

        self.manual_buy_button = QPushButton("Buy")
        self.manual_sell_button = QPushButton("Sell")
        self.manual_buy_button.clicked.connect(lambda: self.execute_manual_order("BUY"))
        self.manual_sell_button.clicked.connect(lambda: self.execute_manual_order("SELL"))

        manual_layout.addWidget(QLabel("Qty:"))
        manual_layout.addWidget(self.manual_qty)
        manual_layout.addWidget(QLabel("Price:"))
        manual_layout.addWidget(self.manual_price)
        manual_layout.addWidget(QLabel("Type:"))
        manual_layout.addWidget(self.manual_order_type)
        manual_layout.addWidget(self.manual_buy_button)
        manual_layout.addWidget(self.manual_sell_button)
        manual_group.setLayout(manual_layout)

        layout.addWidget(manual_group)

        # Enhanced data tables
        self.setup_enhanced_tables(layout)
        
        self.trading_tab.setLayout(layout)
    
    def setup_ml_tab(self):
        """Setup comprehensive ML configuration and monitoring"""
        layout = QVBoxLayout()
        
        # ML System Status
        status_group = QGroupBox("🤖 ML System Status")
        status_layout = QVBoxLayout()
        
        self.ml_detailed_status = QLabel("Status: Initializing ML system...")
        self.ml_detailed_status.setFont(QFont('Arial', 11, QFont.Bold))
        
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        
        self.ml_accuracy_label = QLabel("Model Accuracy: Not trained")
        self.ml_model_info = QLabel("Current Model: None loaded")
        
        status_layout.addWidget(self.ml_detailed_status)
        status_layout.addWidget(self.training_progress)
        status_layout.addWidget(self.ml_accuracy_label)
        status_layout.addWidget(self.ml_model_info)
        status_group.setLayout(status_layout)
        
        # Model Management
        model_group = QGroupBox("📁 Model Management")
        model_layout = QFormLayout()
        
        # Model file operations
        file_layout = QHBoxLayout()
        self.model_file_input = QLineEdit()
        self.model_file_input.setPlaceholderText("Select model file path...")
        
        self.browse_button = QPushButton("📂 Browse")
        self.browse_button.clicked.connect(self.browse_model_file)
        
        file_layout.addWidget(self.model_file_input)
        file_layout.addWidget(self.browse_button)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_model_button = QPushButton("💾 Save Model")
        self.load_model_button = QPushButton("📤 Load Model")
        self.retrain_button = QPushButton("🔄 Retrain Now")
        self.export_signals_button = QPushButton("📊 Export Signals")
        
        self.save_model_button.clicked.connect(self.save_ml_model)
        self.load_model_button.clicked.connect(self.load_ml_model)
        self.retrain_button.clicked.connect(self.retrain_ml_model)
        self.export_signals_button.clicked.connect(self.export_signal_history)
        
        button_layout.addWidget(self.save_model_button)
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.retrain_button)
        button_layout.addWidget(self.export_signals_button)
        
        model_layout.addRow("Model File:", file_layout)
        model_layout.addRow("Actions:", button_layout)
        model_group.setLayout(model_layout)
        
        # Feature Importance Display
        features_group = QGroupBox("🎯 Feature Importance")
        features_layout = QVBoxLayout()
        
        self.feature_importance_table = QTableWidget(0, 2)
        self.feature_importance_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        self.feature_importance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feature_importance_table.setMaximumHeight(200)
        
        features_layout.addWidget(self.feature_importance_table)
        features_group.setLayout(features_layout)
        
        # ML Signals History
        signals_group = QGroupBox("🎯 Recent ML Signals")
        signals_layout = QVBoxLayout()
        
        self.ml_signals_table = QTableWidget(0, 7)
        self.ml_signals_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Signal", "Confidence", "Price", "Model", "Trained"
        ])
        self.ml_signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ml_signals_table.setMaximumHeight(250)
        
        signals_layout.addWidget(self.ml_signals_table)
        signals_group.setLayout(signals_layout)
        
        # Add all groups
        layout.addWidget(status_group)
        layout.addWidget(model_group)
        layout.addWidget(features_group)
        layout.addWidget(signals_group)
        layout.addStretch()
        
        self.ml_tab.setLayout(layout)
    
    def setup_config_tab(self):
        """Setup enhanced API configuration"""
        layout = QVBoxLayout()
        
        # API Configuration
        api_group = QGroupBox("🔐 Binance API Configuration")
        api_layout = QFormLayout()
        
        # Enhanced API inputs
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Binance API Key...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setPlaceholderText("Enter your Binance API Secret...")
        self.api_secret_input.setEchoMode(QLineEdit.Password)
        
        # Environment selection
        self.testnet_checkbox = QCheckBox("Use Testnet (Recommended for testing)")
        self.testnet_checkbox.setChecked(True)
        
        # Configuration display
        self.config_info = QTextEdit()
        self.config_info.setMaximumHeight(120)
        self.config_info.setPlainText(
            f"🔧 Configuration Details:\n"
            f"• Key Type: {self.binance_api.config['key_type']}\n"
            f"• Description: {self.binance_api.config['description']}\n"
            f"• Environment: {'Testnet' if self.binance_api.use_testnet else 'Live Trading'}\n"
            f"• Status: Not configured\n\n"
            f"⚠️  IMPORTANT: Start with testnet for safety!"
        )
        self.config_info.setReadOnly(True)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_config_button = QPushButton("💾 Save Configuration")
        self.test_connection_button = QPushButton("🔗 Test Connection")
        self.clear_config_button = QPushButton("🗑️ Clear")
        
        self.save_config_button.clicked.connect(self.save_api_config)
        self.test_connection_button.clicked.connect(self.test_api_connection)
        self.clear_config_button.clicked.connect(self.clear_api_config)
        
        button_layout.addWidget(self.save_config_button)
        button_layout.addWidget(self.test_connection_button)
        button_layout.addWidget(self.clear_config_button)
        button_layout.addStretch()
        
        # Form assembly
        api_layout.addRow("API Key:", self.api_key_input)
        api_layout.addRow("API Secret:", self.api_secret_input)
        api_layout.addRow("Options:", self.testnet_checkbox)
        api_layout.addRow("Info:", self.config_info)
        api_layout.addRow("Actions:", button_layout)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        layout.addStretch()
        
        self.config_tab.setLayout(layout)
    
    def setup_account_tab(self):
        """Setup enhanced account information display"""
        layout = QVBoxLayout()
        
        # Account overview
        overview_group = QGroupBox("💰 Account Overview")
        overview_layout = QVBoxLayout()
        
        self.account_info_display = QTextEdit()
        self.account_info_display.setReadOnly(True)
        self.account_info_display.setPlainText(
            "🔐 Account Information\n" + "="*50 + "\n\n"
            "Please configure API credentials to view account details.\n\n"
            "Features available after configuration:\n"
            "• Real-time balance information\n"
            "• Trading permissions status\n"
            "• Account type and limits\n"
            "• Recent trading activity"
        )
        
        # Refresh controls
        refresh_layout = QHBoxLayout()
        self.refresh_account_button = QPushButton("🔄 Refresh Account Info")
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh every 30s")
        
        self.refresh_account_button.clicked.connect(self.refresh_account_info)
        
        refresh_layout.addWidget(self.refresh_account_button)
        refresh_layout.addWidget(self.auto_refresh_checkbox)
        refresh_layout.addStretch()
        
        overview_layout.addWidget(self.account_info_display)
        overview_layout.addLayout(refresh_layout)
        overview_group.setLayout(overview_layout)
        
        layout.addWidget(overview_group)
        
        self.account_tab.setLayout(layout)
    
    def setup_analytics_tab(self):
        """Setup analytics and performance tracking"""
        layout = QVBoxLayout()
        
        # Performance metrics
        metrics_group = QGroupBox("📈 Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        # Summary stats
        stats_layout = QHBoxLayout()
        
        self.total_signals_label = QLabel("Total Signals: 0")
        self.buy_signals_label = QLabel("Buy Signals: 0")
        self.sell_signals_label = QLabel("Sell Signals: 0")
        self.hold_signals_label = QLabel("Hold Signals: 0")
        
        for label in [self.total_signals_label, self.buy_signals_label, 
                     self.sell_signals_label, self.hold_signals_label]:
            label.setFont(QFont('Arial', 10, QFont.Bold))
        
        stats_layout.addWidget(self.total_signals_label)
        stats_layout.addWidget(self.buy_signals_label)
        stats_layout.addWidget(self.sell_signals_label)
        stats_layout.addWidget(self.hold_signals_label)
        stats_layout.addStretch()
        
        # Performance table
        self.performance_table = QTableWidget(0, 5)
        self.performance_table.setHorizontalHeaderLabels([
            "Time Period", "Signals", "Avg Confidence", "Best Signal", "Model Used"
        ])
        self.performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.performance_table.setMaximumHeight(200)
        
        metrics_layout.addLayout(stats_layout)
        metrics_layout.addWidget(QLabel("Recent Performance:"))
        metrics_layout.addWidget(self.performance_table)
        metrics_group.setLayout(metrics_layout)
        
        layout.addWidget(metrics_group)
        layout.addStretch()
        
        self.analytics_tab.setLayout(layout)

    def setup_about_tab(self):
        """Setup about page with a button to copy source code."""
        layout = QVBoxLayout()
        layout.addWidget(QLabel(
            "<b>Advanced ML Binance Trading Bot</b><br>Source available for review."))
        self.copy_code_button = QPushButton("Copy Source Code")
        self.copy_code_button.clicked.connect(self.copy_source_code)
        layout.addWidget(self.copy_code_button)
        layout.addStretch()
        self.about_tab.setLayout(layout)

    def copy_source_code(self):
        """Copy full source file to clipboard."""
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                code = f.read()
            QApplication.clipboard().setText(code)
            QMessageBox.information(self, "Copy Code", "Source code copied to clipboard")
        except Exception as e:
            QMessageBox.warning(self, "Copy Code", f"Failed to copy code: {e}")
    
    def setup_enhanced_tables(self, parent_layout):
        """Setup enhanced data display tables"""
        # Live Signals Table
        signals_group = QGroupBox("🎯 Live Trading Signals")
        signals_layout = QVBoxLayout()
        
        self.signals_table = QTableWidget(0, 6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Signal", "Price", "Confidence", "Source"
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.signals_table.setAlternatingRowColors(True)
        self.signals_table.setMaximumHeight(200)
        
        signals_layout.addWidget(self.signals_table)
        signals_group.setLayout(signals_layout)
        parent_layout.addWidget(signals_group)
        
        # Price History Table
        history_group = QGroupBox("📊 Recent Price Updates")
        history_layout = QVBoxLayout()
        
        self.price_history_table = QTableWidget(0, 5)
        self.price_history_table.setHorizontalHeaderLabels([
            "Time", "Price", "Change %", "Volume", "Trend"
        ])
        self.price_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.price_history_table.setAlternatingRowColors(True)
        self.price_history_table.setMaximumHeight(150)
        
        history_layout.addWidget(self.price_history_table)
        history_group.setLayout(history_layout)
        parent_layout.addWidget(history_group)
    
    def apply_styling(self):
        """Apply comprehensive modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2b2b2b, stop:1 #1e1e1e);
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 2px solid #444;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3c3c3c, stop:1 #2a2a2a);
                border-radius: 8px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
                color: #ffffff;
                padding: 12px 20px;
                margin-right: 3px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #606060, stop:1 #4a4a4a);
                border-bottom: 3px solid #00ff88;
            }
            
            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #555555, stop:1 #404040);
            }
            
            QLabel {
                color: #ffffff;
                padding: 3px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
                color: white;
                border: none;
                padding: 10px 16px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
                min-width: 100px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #388E3C, stop:1 #2E7D32);
            }
            
            QPushButton:disabled {
                background: #666666;
                color: #999999;
            }
            
            QComboBox, QLineEdit {
                padding: 8px 12px;
                font-size: 12px;
                border: 2px solid #555;
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
                color: #ffffff;
                min-width: 120px;
            }
            
            QComboBox:focus, QLineEdit:focus {
            border: 2px solid #00ff88;
            }
            
            QComboBox::drop-down {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #555, stop:1 #444);
                width: 20px;
                border-radius: 4px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid #ffffff;
                margin: 2px;
            }
            
            QTableWidget {
                gridline-color: #555;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3c3c3c, stop:1 #2a2a2a);
                alternate-background-color: #4a4a4a;
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 8px;
                selection-background-color: #00ff88;
            }
            
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #606060, stop:1 #4a4a4a);
                padding: 10px;
                border: 1px solid #555;
                font-weight: bold;
                color: #ffffff;
                border-radius: 4px;
            }
            
            QHeaderView::section:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #707070, stop:1 #5a5a5a);
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #00ff88;
                font-size: 14px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3c3c3c, stop:1 #4a4a4a);
                border-radius: 6px;
            }
            
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #00ff88;
                selection-color: #000000;
            }
            
            QCheckBox {
                color: #ffffff;
                font-weight: bold;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #555;
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
            }
            
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00ff88, stop:1 #00cc66);
                border: 2px solid #00ff88;
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #00ff88;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
            }
            
            QProgressBar {
                border: 2px solid #555;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                min-height: 25px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:0.5 #00cc66, stop:1 #00aa44);
                border-radius: 6px;
                margin: 2px;
            }
            
            QScrollBar:vertical {
                background: #3a3a3a;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
        """)
    
    # ===== CORE EVENT HANDLERS AND FUNCTIONALITY =====
    
    def on_symbol_changed(self, symbol):
        """Handle symbol change with intelligent restart logic"""
        old_symbol = self.current_symbol
        self.current_symbol = symbol
        
        self.update_connection_status(f"🔄 Symbol changed: {old_symbol} → {symbol}")
        
        if self.enhanced_data_worker and self.enhanced_data_worker.running:
            reply = QMessageBox.question(self, "Symbol Change", 
                f"Changing symbol will restart data collection.\n"
                f"Current progress will be lost. Continue?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_ml_trading()
                # Use QTimer to restart after UI updates
                QTimer.singleShot(2000, self.start_ml_trading)
            else:
                # Revert symbol change
                self.symbol_combo.blockSignals(True)
                self.symbol_combo.setCurrentText(old_symbol)
                self.symbol_combo.blockSignals(False)
                self.current_symbol = old_symbol
    
    def on_model_changed(self, model_text):
        """Handle ML model selection with validation"""
        model_map = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Advanced Ensemble": "advanced_ensemble",
            "Council": "council",
            "Fallback Rules": "fallback",
        }
        
        new_model = model_map.get(model_text, "random_forest")
        
        if new_model == self.current_model:
            return  # No change needed
        
        old_model = self.current_model
        self.current_model = new_model
        
        if self.enhanced_data_worker:
            if new_model == "fallback":
                # Switch to fallback mode immediately
                self.enhanced_data_worker.ml_generator.is_trained = False
                self.update_ml_status("🔧 Using Fallback Rules (No training required)")
                self.ml_accuracy_label.setText("Model Accuracy: N/A (Rule-based)")
                self.ml_model_info.setText("Current Model: Fallback Rules")
                self.update_feature_importance_display([])
            else:
                # Check if we need to train the new model type
                if (not self.enhanced_data_worker.ml_generator.is_trained or 
                    new_model not in self.enhanced_data_worker.ml_generator.models):
                    
                    if self.auto_retrain_checkbox.isChecked():
                        self.update_ml_status(f"🔄 Switching to {model_text}...")
                        self.enhanced_data_worker.set_model_type(new_model)
                    else:
                        reply = QMessageBox.question(self, "Model Change",
                            f"Switching to {model_text} requires training.\n"
                            f"Start training now?",
                            QMessageBox.Yes | QMessageBox.No)
                        
                        if reply == QMessageBox.Yes:
                            self.enhanced_data_worker.set_model_type(new_model)
                        else:
                            # Revert model change
                            self.model_combo.blockSignals(True)
                            self.model_combo.setCurrentText(old_model.replace('_', ' ').title())
                            self.model_combo.blockSignals(False)
                            self.current_model = old_model
                            return
                else:
                    # Model already trained, switch immediately
                    self.enhanced_data_worker.set_model_type(new_model)
                    accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
                    self.update_ml_status(f"✅ Switched to {model_text} (Accuracy: {accuracy:.1%})")
        
        self.ml_model_info.setText(f"Current Model: {model_text}")
    
    def start_ml_trading(self):
        """Start comprehensive ML trading system with validation"""
        # Validate API configuration
        if not self.binance_api.api_key and not self.binance_api.api_secret:
            reply = QMessageBox.question(self, "Demo Mode", 
                "No API credentials configured.\n\n"
                "Run in demo mode with simulated data?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply != QMessageBox.Yes:
                return
        
        # Validate symbol selection
        self.current_symbol = self.symbol_combo.currentText()
        if not self.current_symbol:
            QMessageBox.warning(self, "Invalid Symbol", "Please select a trading symbol.")
            return
        
        try:
            # Create enhanced data worker with comprehensive error handling
            self.enhanced_data_worker = EnhancedDataWorker(self.binance_api, self.current_symbol)
            
            # Connect all signals with error handling
            self.enhanced_data_worker.data_updated.connect(self.update_live_data)
            self.enhanced_data_worker.ml_signal_generated.connect(self.add_ml_signal)
            self.enhanced_data_worker.ml_status_updated.connect(self.update_ml_status)
            self.enhanced_data_worker.training_progress.connect(self.update_training_progress)
            self.enhanced_data_worker.training_message.connect(self.update_training_message)
            
            # Set model type and initialize
            self.enhanced_data_worker.set_model_type(self.current_model)
            
            # Initialize ML system if auto-retrain is enabled and not using fallback
            if (self.auto_retrain_checkbox.isChecked() and 
                self.current_model != "fallback"):
                self.training_progress.setVisible(True)
                self.enhanced_data_worker.initialize_ml_system()
            
            # Start data processing
            self.enhanced_data_worker.start_updates()
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.symbol_combo.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.auto_retrain_checkbox.setEnabled(False)
            self.is_trading = True
            
            self.update_connection_status(f"🚀 ML Trading active: {self.current_symbol}")
            
            if self.current_model == "fallback":
                self.update_ml_status("🔧 Using Fallback Rules")
            else:
                self.update_ml_status("🔄 Initializing ML system...")
            
        except Exception as e:
            QMessageBox.critical(self, "Startup Error", 
                f"Failed to start ML trading:\n{str(e)}")
            self.update_connection_status(f"❌ Startup failed: {str(e)}")
    
    def stop_ml_trading(self):
        """Stop ML trading with proper cleanup"""
        try:
            if self.enhanced_data_worker:
                self.enhanced_data_worker.stop_updates()
                self.enhanced_data_worker = None
            
            # Reset UI state
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.symbol_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.auto_retrain_checkbox.setEnabled(True)
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)
            self.is_trading = False
            
            self.update_connection_status("⏹️ ML Trading stopped")
            self.update_ml_status("🔄 Idle")
            
        except Exception as e:
            QMessageBox.warning(self, "Stop Error",
                f"Error during shutdown:\n{str(e)}")

    def execute_manual_order(self, side):
        """Execute a manual trade using current API settings."""
        qty_text = self.manual_qty.text().strip()
        if not qty_text:
            QMessageBox.warning(self, "Manual Trade", "Quantity is required")
            return
        try:
            qty = float(qty_text)
            order_type = self.manual_order_type.currentText()
            price = self.manual_price.text().strip()
            price_val = float(price) if price and order_type == "LIMIT" else None
            result = self.binance_api.place_order(
                self.symbol_combo.currentText(), side, qty, order_type, price_val
            )
            if "error" in result:
                QMessageBox.warning(self, "Manual Trade", f"Order failed: {result['error']}")
            else:
                status = result.get('status', 'Placed')
                QMessageBox.information(self, "Manual Trade", f"Order {status}")
        except ValueError:
            QMessageBox.warning(self, "Manual Trade", "Invalid number format")

    def auto_trade_signal(self, side):
        """Place an automatic market order based on ML signal."""
        qty_text = self.auto_trade_qty.text().strip()
        if not qty_text:
            return
        try:
            qty = float(qty_text)
        except ValueError:
            return
        result = self.binance_api.place_order(
            self.symbol_combo.currentText(), side, qty
        )
        if "error" in result:
            print(f"Auto trade failed: {result['error']}")
    
    # ===== DATA PROCESSING AND UI UPDATES =====
    
    def update_live_data(self, data):
        """Process and display live market data with trend analysis"""
        try:
            if 'price' in data and 'price' in data['price']:
                current_price = float(data['price']['price'])
                
                # Enhanced price display with formatting
                if current_price >= 1000:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.2f}"
                else:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.4f}"

                self.price_display.setText(price_text)
                if hasattr(self, 'price_chart'):
                    self.price_chart.add_price(current_price)
                if 'candle' in data and data['candle'] and hasattr(self, 'candlestick_chart'):
                    c = data['candle']
                    self.candlestick_chart.add_candle(
                        float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
                    )

                # Add to price history with trend analysis
                self.add_enhanced_price_history(current_price, data.get('stats', {}))
            
            # Update comprehensive market statistics
            if 'stats' in data and data['stats']:
                self.update_market_statistics(data['stats'])
            
            # Update counter
            if 'update_count' in data:
                self.update_counter_label.setText(f"Updates: {data['update_count']}")
                
        except Exception as e:
            print(f"Error updating live data: {e}")
            self.update_connection_status(f"⚠️ Data processing error: {str(e)}")
    
    def update_market_statistics(self, stats):
        """Update comprehensive market statistics with color coding"""
        try:
            # 24h Price Change
            if 'priceChangePercent' in stats:
                change_pct = float(stats['priceChangePercent'])
                change_text = f"📊 24h: {change_pct:+.2f}%"
                
                self.stats_24h_change.setText(change_text)
                if change_pct > 2.0:
                    self.stats_24h_change.setStyleSheet("color: #00ff88; font-weight: bold;")
                elif change_pct > 0:
                    self.stats_24h_change.setStyleSheet("color: #88ff88; font-weight: bold;")
                elif change_pct < -2.0:
                    self.stats_24h_change.setStyleSheet("color: #ff4444; font-weight: bold;")
                elif change_pct < 0:
                    self.stats_24h_change.setStyleSheet("color: #ff8888; font-weight: bold;")
                else:
                    self.stats_24h_change.setStyleSheet("color: #ffffff; font-weight: bold;")
            
            # 24h Volume with intelligent formatting
            if 'volume' in stats:
                volume = float(stats['volume'])
                if volume >= 1_000_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000_000:.1f}B"
                elif volume >= 1_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000:.1f}M"
                elif volume >= 1_000:
                    volume_text = f"📈 Vol: {volume/1_000:.1f}K"
                else:
                    volume_text = f"📈 Vol: {volume:.0f}"
                self.stats_24h_volume.setText(volume_text)
            
            # 24h High/Low
            if 'highPrice' in stats and 'lowPrice' in stats:
                high = float(stats['highPrice'])
                low = float(stats['lowPrice'])
                if high >= 1000:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.2f}/${low:,.2f}")
                else:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.4f}/${low:,.4f}")
            
            # Trade Count
            if 'count' in stats:
                trades = int(stats['count'])
                self.stats_trades.setText(f"🔄 Trades: {trades:,}")
                
        except Exception as e:
            print(f"Error updating market statistics: {e}")
    
    def add_enhanced_price_history(self, price, stats):
        """Add price history entry with sophisticated trend analysis"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Calculate trend with multiple indicators
        trend_emoji = "➡️"  # Default neutral
        trend_text = "Stable"
        trend_color = QColor(255, 255, 255)
        
        if self.price_history_table.rowCount() > 0:
            try:
                last_price_item = self.price_history_table.item(0, 1)
                if last_price_item:
                    last_price_text = last_price_item.text().replace('$', '').replace(',', '')
                    last_price = float(last_price_text)
                    
                    price_change = (price - last_price) / last_price * 100
                    
                    if price_change > 0.5:
                        trend_emoji = "📈"
                        trend_text = "Strong Up"
                        trend_color = QColor(0, 255, 136)
                    elif price_change > 0.1:
                        trend_emoji = "⬆️"
                        trend_text = "Up"
                        trend_color = QColor(136, 255, 136)
                    elif price_change < -0.5:
                        trend_emoji = "📉"
                        trend_text = "Strong Down"
                        trend_color = QColor(255, 68, 68)
                    elif price_change < -0.1:
                        trend_emoji = "⬇️"
                        trend_text = "Down"
                        trend_color = QColor(255, 136, 136)
            except (ValueError, AttributeError):
                pass
        
        # Insert new row with comprehensive data
        self.price_history_table.insertRow(0)
        
        self.price_history_table.setItem(0, 0, QTableWidgetItem(current_time))
        
        # Price with appropriate formatting
        if price >= 1000:
            price_item = QTableWidgetItem(f"${price:,.2f}")
        else:
            price_item = QTableWidgetItem(f"${price:,.4f}")
        self.price_history_table.setItem(0, 1, price_item)
        
        # 24h Change percentage
        if 'priceChangePercent' in stats:
            change_pct = float(stats['priceChangePercent'])
            change_item = QTableWidgetItem(f"{change_pct:+.2f}%")
            if change_pct > 0:
                change_item.setForeground(QColor(0, 255, 136))
            elif change_pct < 0:
                change_item.setForeground(QColor(255, 68, 68))
            self.price_history_table.setItem(0, 2, change_item)
        else:
            self.price_history_table.setItem(0, 2, QTableWidgetItem("--"))
        
        # Volume with intelligent formatting
        if 'volume' in stats:
            volume = float(stats['volume'])
            if volume >= 1_000_000:
                volume_text = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                volume_text = f"{volume/1_000:.1f}K"
            else:
                volume_text = f"{volume:.0f}"
            self.price_history_table.setItem(0, 3, QTableWidgetItem(volume_text))
        else:
            self.price_history_table.setItem(0, 3, QTableWidgetItem("--"))
        
        # Trend indicator
        trend_item = QTableWidgetItem(f"{trend_emoji} {trend_text}")
        trend_item.setForeground(trend_color)
        self.price_history_table.setItem(0, 4, trend_item)
        
        # Maintain reasonable table size
        if self.price_history_table.rowCount() > 25:
            self.price_history_table.removeRow(25)
    
    def add_ml_signal(self, signal_data):
        """Add ML signal with comprehensive tracking and analytics"""
        try:
            current_time = signal_data['timestamp'].strftime("%H:%M:%S")
            
            # Add to main signals table
            self.signals_table.insertRow(0)
            
            self.signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            # Enhanced signal display with confidence-based styling
            signal_text = signal_data['signal']
            confidence = signal_data['confidence']
            
            if signal_data.get('is_ml_trained', False):
                display_signal = f"{signal_text} 🤖"
            else:
                display_signal = f"{signal_text} 📋"
            
            signal_item = QTableWidgetItem(display_signal)
            
            # Color coding based on signal type and confidence
            if signal_text == "BUY":
                if confidence >= 80:
                    signal_item.setForeground(QColor(0, 255, 136))  # Bright green
                else:
                    signal_item.setForeground(QColor(136, 255, 136))  # Light green
            elif signal_text == "SELL":
                if confidence >= 80:
                    signal_item.setForeground(QColor(255, 68, 68))  # Bright red
                else:
                    signal_item.setForeground(QColor(255, 136, 136))  # Light red
            else:  # HOLD
                signal_item.setForeground(QColor(255, 255, 102))  # Yellow
            
            self.signals_table.setItem(0, 2, signal_item)
            self.signals_table.setItem(0, 3, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            
            # Confidence with color coding
            confidence_item = QTableWidgetItem(f"{confidence}%")
            if confidence >= 85:
                confidence_item.setForeground(QColor(0, 255, 136))
            elif confidence >= 70:
                confidence_item.setForeground(QColor(255, 255, 102))
            else:
                confidence_item.setForeground(QColor(255, 165, 0))
            
            self.signals_table.setItem(0, 4, confidence_item)
            
            # Source indicator
            source = "🤖 ML" if signal_data.get('is_ml_trained', False) else "📋 Rules"
            self.signals_table.setItem(0, 5, QTableWidgetItem(source))
            
            # Add to ML-specific signals table
            self.ml_signals_table.insertRow(0)
            self.ml_signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.ml_signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            # Clone signal item for ML table
            ml_signal_item = QTableWidgetItem(display_signal)
            ml_signal_item.setForeground(signal_item.foreground())
            self.ml_signals_table.setItem(0, 2, ml_signal_item)
            
            # Clone confidence item
            ml_confidence_item = QTableWidgetItem(f"{confidence}%")
            ml_confidence_item.setForeground(confidence_item.foreground())
            self.ml_signals_table.setItem(0, 3, ml_confidence_item)
            
            self.ml_signals_table.setItem(0, 4, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            self.ml_signals_table.setItem(0, 5, QTableWidgetItem(
                signal_data.get('model_type', 'Unknown').replace('_', ' ').title()
            ))
            self.ml_signals_table.setItem(0, 6, QTableWidgetItem(
                "✅" if signal_data.get('is_ml_trained', False) else "❌"
            ))

            # Auto trade if enabled
            if (self.auto_trade_checkbox.isChecked() and
                signal_text in ("BUY", "SELL") and confidence >= 75):
                self.auto_trade_signal(signal_text)
            
            # Update analytics
            self.update_signal_analytics(signal_data)
            
            # Maintain table sizes
            for table in [self.signals_table, self.ml_signals_table]:
                if table.rowCount() > 20:
                    table.removeRow(20)
                    
        except Exception as e:
            print(f"Error adding ML signal: {e}")
    
    
    # ===== ENHANCED EVENT HANDLERS =====
    
    def on_symbol_changed(self, symbol):
        """Handle symbol change with intelligent validation and restart logic"""
        old_symbol = self.current_symbol
        self.current_symbol = symbol
        
        self.update_connection_status(f"🔄 Symbol changed: {old_symbol} → {symbol}")
        
        if self.enhanced_data_worker and self.enhanced_data_worker.running:
            reply = QMessageBox.question(self, "Symbol Change", 
                f"Changing symbol will restart data collection and ML training.\n\n"
                f"Current progress will be lost. Continue with {symbol}?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_ml_trading()
                # Restart after UI updates
                QTimer.singleShot(2000, self.start_ml_trading)
            else:
                # Revert symbol change
                self.symbol_combo.blockSignals(True)
                self.symbol_combo.setCurrentText(old_symbol)
                self.symbol_combo.blockSignals(False)
                self.current_symbol = old_symbol
    
    def on_model_changed(self, model_text):
        """Handle ML model selection with comprehensive validation"""
        model_map = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Advanced Ensemble": "advanced_ensemble",
            "Council": "council",
            "Fallback Rules": "fallback",
        }
        
        new_model = model_map.get(model_text, "random_forest")
        
        if new_model == self.current_model:
            return
        
        old_model = self.current_model
        self.current_model = new_model
        
        if self.enhanced_data_worker:
            if new_model == "fallback":
                # Switch to fallback mode immediately
                self.enhanced_data_worker.ml_generator.is_trained = False
                self.update_ml_status("🔧 Using Fallback Rules (No training required)")
                self.ml_accuracy_label.setText("Model Accuracy: N/A (Rule-based)")
                self.ml_model_info.setText("Current Model: Fallback Rules")
                self.update_feature_importance_display([])
            else:
                if (not self.enhanced_data_worker.ml_generator.is_trained or 
                    new_model not in self.enhanced_data_worker.ml_generator.models):
                    
                    if self.auto_retrain_checkbox.isChecked():
                        self.update_ml_status(f"🔄 Switching to {model_text}...")
                        self.enhanced_data_worker.set_model_type(new_model)
                    else:
                        reply = QMessageBox.question(self, "Model Change",
                            f"Switching to {model_text} requires training.\n\n"
                            f"Start training now?",
                            QMessageBox.Yes | QMessageBox.No)
                        
                        if reply == QMessageBox.Yes:
                            self.enhanced_data_worker.set_model_type(new_model)
                        else:
                            # Revert model change
                            self.model_combo.blockSignals(True)
                            self.model_combo.setCurrentText(old_model.replace('_', ' ').title())
                            self.model_combo.blockSignals(False)
                            self.current_model = old_model
                            return
                else:
                    # Model already trained, switch immediately
                    self.enhanced_data_worker.set_model_type(new_model)
                    accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
                    self.update_ml_status(f"✅ Switched to {model_text} (Accuracy: {accuracy:.1%})")
        
        self.ml_model_info.setText(f"Current Model: {model_text}")
    
    def start_ml_trading(self):
        """Start comprehensive ML trading system with enhanced validation"""
        # Validate API configuration
        if not self.binance_api.api_key and not self.binance_api.api_secret:
            reply = QMessageBox.question(self, "Demo Mode", 
                "No API credentials configured.\n\n"
                "Continue in demo mode with simulated data?\n"
                "(You can configure real API keys in the API Configuration tab)",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply != QMessageBox.Yes:
                return
        
        # Validate symbol selection
        self.current_symbol = self.symbol_combo.currentText()
        if not self.current_symbol:
            QMessageBox.warning(self, "Invalid Symbol", "Please select a trading symbol.")
            return
        
        try:
            # Create enhanced data worker
            self.enhanced_data_worker = EnhancedDataWorker(self.binance_api, self.current_symbol)
            
            # Connect all signals with comprehensive error handling
            self.enhanced_data_worker.data_updated.connect(self.update_live_data)
            self.enhanced_data_worker.ml_signal_generated.connect(self.add_ml_signal)
            self.enhanced_data_worker.ml_status_updated.connect(self.update_ml_status)
            self.enhanced_data_worker.training_progress.connect(self.update_training_progress)
            
            # Set model type and initialize
            self.enhanced_data_worker.set_model_type(self.current_model)
            
            # Initialize ML system if not using fallback
            if (self.auto_retrain_checkbox.isChecked() and 
                self.current_model != "fallback"):
                self.training_progress.setVisible(True)
                self.enhanced_data_worker.initialize_ml_system()
            
            # Start data processing
            self.enhanced_data_worker.start_updates()
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.symbol_combo.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.auto_retrain_checkbox.setEnabled(False)
            self.is_trading = True
            
            self.update_connection_status(f"🚀 ML Trading active: {self.current_symbol}")
            
            if self.current_model == "fallback":
                self.update_ml_status("🔧 Using Fallback Rules")
            else:
                self.update_ml_status("🔄 Initializing ML system...")
            
        except Exception as e:
            QMessageBox.critical(self, "Startup Error", 
                f"Failed to start ML trading:\n{str(e)}")
            self.update_connection_status(f"❌ Startup failed: {str(e)}")
    
    def stop_ml_trading(self):
        """Stop ML trading with comprehensive cleanup"""
        try:
            if self.enhanced_data_worker:
                self.enhanced_data_worker.stop_updates()
                self.enhanced_data_worker = None
            
            # Reset UI state
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.symbol_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.auto_retrain_checkbox.setEnabled(True)
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)
            self.is_trading = False
            
            self.update_connection_status("⏹️ ML Trading stopped")
            self.update_ml_status("🔄 Idle")
            
        except Exception as e:
            QMessageBox.warning(self, "Stop Error", 
                f"Error during shutdown:\n{str(e)}")
    
    # ===== ENHANCED DATA PROCESSING =====
    
    def update_live_data(self, data):
        """Process live market data with sophisticated trend analysis"""
        try:
            if 'price' in data and 'price' in data['price']:
                current_price = float(data['price']['price'])
                
                # Enhanced price display with intelligent formatting
                if current_price >= 1000:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.2f}"
                else:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.4f}"

                self.price_display.setText(price_text)
                if hasattr(self, 'price_chart'):
                    self.price_chart.add_price(current_price)
                self.add_enhanced_price_history(current_price, data.get('stats', {}))
            
            # Update comprehensive market statistics
            if 'stats' in data and data['stats']:
                self.update_market_statistics(data['stats'])
            
            # Update counter
            if 'update_count' in data:
                self.update_counter_label.setText(f"Updates: {data['update_count']}")
                
        except Exception as e:
            print(f"Error updating live data: {e}")
            self.update_connection_status(f"⚠️ Data processing error: {str(e)}")
    
    def update_market_statistics(self, stats):
        """Update market statistics with intelligent color coding"""
        try:
            # 24h Price Change with dynamic coloring
            if 'priceChangePercent' in stats:
                change_pct = float(stats['priceChangePercent'])
                change_text = f"📊 24h: {change_pct:+.2f}%"
                
                self.stats_24h_change.setText(change_text)
                if change_pct > 2.0:
                    self.stats_24h_change.setStyleSheet("color: #00ff88; font-weight: bold;")
                elif change_pct > 0:
                    self.stats_24h_change.setStyleSheet("color: #88ff88; font-weight: bold;")
                elif change_pct < -2.0:
                    self.stats_24h_change.setStyleSheet("color: #ff4444; font-weight: bold;")
                elif change_pct < 0:
                    self.stats_24h_change.setStyleSheet("color: #ff8888; font-weight: bold;")
                else:
                    self.stats_24h_change.setStyleSheet("color: #ffffff; font-weight: bold;")
            
            # Volume with intelligent formatting
            if 'volume' in stats:
                volume = float(stats['volume'])
                if volume >= 1_000_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000_000:.1f}B"
                elif volume >= 1_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000:.1f}M"
                elif volume >= 1_000:
                    volume_text = f"📈 Vol: {volume/1_000:.1f}K"
                else:
                    volume_text = f"📈 Vol: {volume:.0f}"
                self.stats_24h_volume.setText(volume_text)
            
            # High/Low with proper formatting
            if 'highPrice' in stats and 'lowPrice' in stats:
                high = float(stats['highPrice'])
                low = float(stats['lowPrice'])
                if high >= 1000:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.2f}/${low:,.2f}")
                else:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.4f}/${low:,.4f}")
            
            # Trade Count
            if 'count' in stats:
                trades = int(stats['count'])
                self.stats_trades.setText(f"🔄 Trades: {trades:,}")
                
        except Exception as e:
            print(f"Error updating market statistics: {e}")
    
    def add_enhanced_price_history(self, price, stats):
        """Add price history with sophisticated trend analysis"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Advanced trend analysis
        trend_emoji = "➡️"
        trend_text = "Stable"
        trend_color = QColor(255, 255, 255)
        
        if self.price_history_table.rowCount() > 0:
            try:
                last_price_item = self.price_history_table.item(0, 1)
                if last_price_item:
                    last_price_text = last_price_item.text().replace('$', '').replace(',', '')
                    last_price = float(last_price_text)
                    
                    price_change = (price - last_price) / last_price * 100
                    
                    if price_change > 0.5:
                        trend_emoji = "📈"
                        trend_text = "Strong Up"
                        trend_color = QColor(0, 255, 136)
                    elif price_change > 0.1:
                        trend_emoji = "⬆️"
                        trend_text = "Up"
                        trend_color = QColor(136, 255, 136)
                    elif price_change < -0.5:
                        trend_emoji = "📉"
                        trend_text = "Strong Down"
                        trend_color = QColor(255, 68, 68)
                    elif price_change < -0.1:
                        trend_emoji = "⬇️"
                        trend_text = "Down"
                        trend_color = QColor(255, 136, 136)
            except (ValueError, AttributeError):
                pass
        
        # Insert comprehensive row data
        self.price_history_table.insertRow(0)
        
        self.price_history_table.setItem(0, 0, QTableWidgetItem(current_time))
        
        # Price formatting
        if price >= 1000:
            price_item = QTableWidgetItem(f"${price:,.2f}")
        else:
            price_item = QTableWidgetItem(f"${price:,.4f}")
        self.price_history_table.setItem(0, 1, price_item)
        
        # 24h Change
        if 'priceChangePercent' in stats:
            change_pct = float(stats['priceChangePercent'])
            change_item = QTableWidgetItem(f"{change_pct:+.2f}%")
            if change_pct > 0:
                change_item.setForeground(QColor(0, 255, 136))
            elif change_pct < 0:
                change_item.setForeground(QColor(255, 68, 68))
            self.price_history_table.setItem(0, 2, change_item)
        else:
            self.price_history_table.setItem(0, 2, QTableWidgetItem("--"))
        
        # Volume
        if 'volume' in stats:
            volume = float(stats['volume'])
            if volume >= 1_000_000:
                volume_text = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                volume_text = f"{volume/1_000:.1f}K"
            else:
                volume_text = f"{volume:.0f}"
            self.price_history_table.setItem(0, 3, QTableWidgetItem(volume_text))
        else:
            self.price_history_table.setItem(0, 3, QTableWidgetItem("--"))
        
        # Trend indicator
        trend_item = QTableWidgetItem(f"{trend_emoji} {trend_text}")
        trend_item.setForeground(trend_color)
        self.price_history_table.setItem(0, 4, trend_item)
        
        # Maintain table size
        if self.price_history_table.rowCount() > 25:
            self.price_history_table.removeRow(25)
    
    def add_ml_signal(self, signal_data):
        """Add ML signal with comprehensive tracking and analytics"""
        try:
            current_time = signal_data['timestamp'].strftime("%H:%M:%S")
            
            # Add to main signals table
            self.signals_table.insertRow(0)
            
            self.signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            # Enhanced signal display
            signal_text = signal_data['signal']
            confidence = signal_data['confidence']
            
            if signal_data.get('is_ml_trained', False):
                display_signal = f"{signal_text} 🤖"
            else:
                display_signal = f"{signal_text} 📋"
            
            signal_item = QTableWidgetItem(display_signal)
            
            # Sophisticated color coding
            if signal_text == "BUY":
                if confidence >= 80:
                    signal_item.setForeground(QColor(0, 255, 136))
                else:
                    signal_item.setForeground(QColor(136, 255, 136))
            elif signal_text == "SELL":
                if confidence >= 80:
                    signal_item.setForeground(QColor(255, 68, 68))
                else:
                    signal_item.setForeground(QColor(255, 136, 136))
            else:  # HOLD
                signal_item.setForeground(QColor(255, 255, 102))
            
            self.signals_table.setItem(0, 2, signal_item)
            self.signals_table.setItem(0, 3, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            
            # Confidence with color coding
            confidence_item = QTableWidgetItem(f"{confidence}%")
            if confidence >= 85:
                confidence_item.setForeground(QColor(0, 255, 136))
            elif confidence >= 70:
                confidence_item.setForeground(QColor(255, 255, 102))
            else:
                confidence_item.setForeground(QColor(255, 165, 0))
            
            self.signals_table.setItem(0, 4, confidence_item)
            
            # Source indicator
            source = "🤖 ML" if signal_data.get('is_ml_trained', False) else "📋 Rules"
            self.signals_table.setItem(0, 5, QTableWidgetItem(source))
            
            # Add to ML signals table
            self.ml_signals_table.insertRow(0)
            self.ml_signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.ml_signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            ml_signal_item = QTableWidgetItem(display_signal)
            ml_signal_item.setForeground(signal_item.foreground())
            self.ml_signals_table.setItem(0, 2, ml_signal_item)
            
            ml_confidence_item = QTableWidgetItem(f"{confidence}%")
            ml_confidence_item.setForeground(confidence_item.foreground())
            self.ml_signals_table.setItem(0, 3, ml_confidence_item)
            
            self.ml_signals_table.setItem(0, 4, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            self.ml_signals_table.setItem(0, 5, QTableWidgetItem(
                signal_data.get('model_type', 'Unknown').replace('_', ' ').title()
            ))
            self.ml_signals_table.setItem(0, 6, QTableWidgetItem(
                "✅" if signal_data.get('is_ml_trained', False) else "❌"
            ))
            
            # Update analytics
            self.update_signal_analytics(signal_data)
            
            # Maintain table sizes
            for table in [self.signals_table, self.ml_signals_table]:
                if table.rowCount() > 20:
                    table.removeRow(20)
                    
        except Exception as e:
            print(f"Error adding ML signal: {e}")
    
    def update_signal_analytics(self, signal_data):
        """Update comprehensive signal analytics with time-based performance tracking"""
        try:
            # Add to performance tracking
            self.signal_performance.append(signal_data)
            
            # Keep only last 100 signals
            if len(self.signal_performance) > 100:
                self.signal_performance.pop(0)
            
            # Calculate comprehensive statistics
            total_signals = len(self.signal_performance)
            buy_signals = sum(1 for s in self.signal_performance if s['signal'] == 'BUY')
            sell_signals = sum(1 for s in self.signal_performance if s['signal'] == 'SELL')
            hold_signals = sum(1 for s in self.signal_performance if s['signal'] == 'HOLD')
            
            # Update summary labels with color coding
            self.total_signals_label.setText(f"Total Signals: {total_signals}")
            self.buy_signals_label.setText(f"Buy Signals: {buy_signals}")
            self.sell_signals_label.setText(f"Sell Signals: {sell_signals}")
            self.hold_signals_label.setText(f"Hold Signals: {hold_signals}")
            
            # Dynamic color coding based on signal distribution
            if buy_signals > sell_signals:
                self.buy_signals_label.setStyleSheet("color: #00ff88; font-weight: bold;")
                self.sell_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            elif sell_signals > buy_signals:
                self.sell_signals_label.setStyleSheet("color: #ff4444; font-weight: bold;")
                self.buy_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            else:
                self.buy_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
                self.sell_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            
            self.hold_signals_label.setStyleSheet("color: #ffff66; font-weight: bold;")
            
            # Update performance table with time-based analysis
            self.update_performance_table()
            
        except Exception as e:
            print(f"Error updating signal analytics: {e}")
    
    
    # ===== ENHANCED EVENT HANDLERS =====
    
    def on_symbol_changed(self, symbol):
        """Handle symbol change with intelligent validation and restart logic"""
        old_symbol = self.current_symbol
        self.current_symbol = symbol
        
        self.update_connection_status(f"🔄 Symbol changed: {old_symbol} → {symbol}")
        
        if self.enhanced_data_worker and self.enhanced_data_worker.running:
            reply = QMessageBox.question(self, "Symbol Change", 
                f"Changing symbol will restart data collection and ML training.\n\n"
                f"Current progress will be lost. Continue with {symbol}?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_ml_trading()
                QTimer.singleShot(2000, self.start_ml_trading)
            else:
                # Revert symbol change
                self.symbol_combo.blockSignals(True)
                self.symbol_combo.setCurrentText(old_symbol)
                self.symbol_combo.blockSignals(False)
                self.current_symbol = old_symbol
    
    def on_model_changed(self, model_text):
        """Handle ML model selection with comprehensive validation"""
        model_map = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Advanced Ensemble": "advanced_ensemble",
            "Council": "council",
            "Fallback Rules": "fallback",
        }
        
        new_model = model_map.get(model_text, "random_forest")
        
        if new_model == self.current_model:
            return
        
        old_model = self.current_model
        self.current_model = new_model
        
        if self.enhanced_data_worker:
            if new_model == "fallback":
                self.enhanced_data_worker.ml_generator.is_trained = False
                self.update_ml_status("🔧 Using Fallback Rules (No training required)")
                self.ml_accuracy_label.setText("Model Accuracy: N/A (Rule-based)")
                self.ml_model_info.setText("Current Model: Fallback Rules")
                self.update_feature_importance_display([])
            else:
                if (not self.enhanced_data_worker.ml_generator.is_trained or 
                    new_model not in self.enhanced_data_worker.ml_generator.models):
                    
                    if self.auto_retrain_checkbox.isChecked():
                        self.update_ml_status(f"🔄 Switching to {model_text}...")
                        self.enhanced_data_worker.set_model_type(new_model)
                    else:
                        reply = QMessageBox.question(self, "Model Change",
                            f"Switching to {model_text} requires training.\n\n"
                            f"Start training now?",
                            QMessageBox.Yes | QMessageBox.No)
                        
                        if reply == QMessageBox.Yes:
                            self.enhanced_data_worker.set_model_type(new_model)
                        else:
                            self.model_combo.blockSignals(True)
                            self.model_combo.setCurrentText(old_model.replace('_', ' ').title())
                            self.model_combo.blockSignals(False)
                            self.current_model = old_model
                            return
                else:
                    self.enhanced_data_worker.set_model_type(new_model)
                    accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
                    self.update_ml_status(f"✅ Switched to {model_text} (Accuracy: {accuracy:.1%})")
        
        self.ml_model_info.setText(f"Current Model: {model_text}")
    
    def start_ml_trading(self):
        """Start comprehensive ML trading system with enhanced validation"""
        if not self.binance_api.api_key and not self.binance_api.api_secret:
            reply = QMessageBox.question(self, "Demo Mode", 
                "No API credentials configured.\n\n"
                "Continue in demo mode with simulated data?\n"
                "(Configure real API keys in the API Configuration tab)",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply != QMessageBox.Yes:
                return
        
        self.current_symbol = self.symbol_combo.currentText()
        if not self.current_symbol:
            QMessageBox.warning(self, "Invalid Symbol", "Please select a trading symbol.")
            return
        
        try:
            self.enhanced_data_worker = EnhancedDataWorker(self.binance_api, self.current_symbol)
            
            # Connect all signals
            self.enhanced_data_worker.data_updated.connect(self.update_live_data)
            self.enhanced_data_worker.ml_signal_generated.connect(self.add_ml_signal)
            self.enhanced_data_worker.ml_status_updated.connect(self.update_ml_status)
            self.enhanced_data_worker.training_progress.connect(self.update_training_progress)
            
            self.enhanced_data_worker.set_model_type(self.current_model)
            
            if (self.auto_retrain_checkbox.isChecked() and 
                self.current_model != "fallback"):
                self.training_progress.setVisible(True)
                self.enhanced_data_worker.initialize_ml_system()
            
            self.enhanced_data_worker.start_updates()
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.symbol_combo.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.auto_retrain_checkbox.setEnabled(False)
            self.is_trading = True
            
            self.update_connection_status(f"🚀 ML Trading active: {self.current_symbol}")
            
            if self.current_model == "fallback":
                self.update_ml_status("🔧 Using Fallback Rules")
            else:
                self.update_ml_status("🔄 Initializing ML system...")
            
        except Exception as e:
            QMessageBox.critical(self, "Startup Error", 
                f"Failed to start ML trading:\n{str(e)}")
            self.update_connection_status(f"❌ Startup failed: {str(e)}")
    
    def stop_ml_trading(self):
        """Stop ML trading with comprehensive cleanup"""
        try:
            if self.enhanced_data_worker:
                self.enhanced_data_worker.stop_updates()
                self.enhanced_data_worker = None
            
            # Reset UI state
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.symbol_combo.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.auto_retrain_checkbox.setEnabled(True)
            self.training_progress.setVisible(False)
            self.training_progress.setValue(0)
            self.is_trading = False
            
            self.update_connection_status("⏹️ ML Trading stopped")
            self.update_ml_status("🔄 Idle")
            
        except Exception as e:
            QMessageBox.warning(self, "Stop Error", 
                f"Error during shutdown:\n{str(e)}")
    
    # ===== ENHANCED DATA PROCESSING =====
    
    def update_live_data(self, data):
        """Process live market data with sophisticated analysis"""
        try:
            if 'price' in data and 'price' in data['price']:
                current_price = float(data['price']['price'])
                
                if current_price >= 1000:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.2f}"
                else:
                    price_text = f"💰 {data['symbol']}: ${current_price:,.4f}"

                self.price_display.setText(price_text)
                if hasattr(self, 'price_chart'):
                    self.price_chart.add_price(current_price)
                self.add_enhanced_price_history(current_price, data.get('stats', {}))
            
            if 'stats' in data and data['stats']:
                self.update_market_statistics(data['stats'])
            
            if 'update_count' in data:
                self.update_counter_label.setText(f"Updates: {data['update_count']}")
                
        except Exception as e:
            print(f"Error updating live data: {e}")
            self.update_connection_status(f"⚠️ Data processing error: {str(e)}")
    
    def update_market_statistics(self, stats):
        """Update market statistics with intelligent color coding"""
        try:
            if 'priceChangePercent' in stats:
                change_pct = float(stats['priceChangePercent'])
                change_text = f"📊 24h: {change_pct:+.2f}%"
                
                self.stats_24h_change.setText(change_text)
                if change_pct > 2.0:
                    self.stats_24h_change.setStyleSheet("color: #00ff88; font-weight: bold;")
                elif change_pct > 0:
                    self.stats_24h_change.setStyleSheet("color: #88ff88; font-weight: bold;")
                elif change_pct < -2.0:
                    self.stats_24h_change.setStyleSheet("color: #ff4444; font-weight: bold;")
                elif change_pct < 0:
                    self.stats_24h_change.setStyleSheet("color: #ff8888; font-weight: bold;")
                else:
                    self.stats_24h_change.setStyleSheet("color: #ffffff; font-weight: bold;")
            
            if 'volume' in stats:
                volume = float(stats['volume'])
                if volume >= 1_000_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000_000:.1f}B"
                elif volume >= 1_000_000:
                    volume_text = f"📈 Vol: {volume/1_000_000:.1f}M"
                elif volume >= 1_000:
                    volume_text = f"📈 Vol: {volume/1_000:.1f}K"
                else:
                    volume_text = f"📈 Vol: {volume:.0f}"
                self.stats_24h_volume.setText(volume_text)
            
            if 'highPrice' in stats and 'lowPrice' in stats:
                high = float(stats['highPrice'])
                low = float(stats['lowPrice'])
                if high >= 1000:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.2f}/${low:,.2f}")
                else:
                    self.stats_high_low.setText(f"⬆️⬇️ H/L: ${high:,.4f}/${low:,.4f}")
            
            if 'count' in stats:
                trades = int(stats['count'])
                self.stats_trades.setText(f"🔄 Trades: {trades:,}")
                
        except Exception as e:
            print(f"Error updating market statistics: {e}")
    
    def add_enhanced_price_history(self, price, stats):
        """Add price history with sophisticated trend analysis"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        trend_emoji = "➡️"
        trend_text = "Stable"
        trend_color = QColor(255, 255, 255)
        
        if self.price_history_table.rowCount() > 0:
            try:
                last_price_item = self.price_history_table.item(0, 1)
                if last_price_item:
                    last_price_text = last_price_item.text().replace('$', '').replace(',', '')
                    last_price = float(last_price_text)
                    
                    price_change = (price - last_price) / last_price * 100
                    
                    if price_change > 0.5:
                        trend_emoji = "📈"
                        trend_text = "Strong Up"
                        trend_color = QColor(0, 255, 136)
                    elif price_change > 0.1:
                        trend_emoji = "⬆️"
                        trend_text = "Up"
                        trend_color = QColor(136, 255, 136)
                    elif price_change < -0.5:
                        trend_emoji = "📉"
                        trend_text = "Strong Down"
                        trend_color = QColor(255, 68, 68)
                    elif price_change < -0.1:
                        trend_emoji = "⬇️"
                        trend_text = "Down"
                        trend_color = QColor(255, 136, 136)
            except (ValueError, AttributeError):
                pass
        
        self.price_history_table.insertRow(0)
        
        self.price_history_table.setItem(0, 0, QTableWidgetItem(current_time))
        
        if price >= 1000:
            price_item = QTableWidgetItem(f"${price:,.2f}")
        else:
            price_item = QTableWidgetItem(f"${price:,.4f}")
        self.price_history_table.setItem(0, 1, price_item)
        
        if 'priceChangePercent' in stats:
            change_pct = float(stats['priceChangePercent'])
            change_item = QTableWidgetItem(f"{change_pct:+.2f}%")
            if change_pct > 0:
                change_item.setForeground(QColor(0, 255, 136))
            elif change_pct < 0:
                change_item.setForeground(QColor(255, 68, 68))
            self.price_history_table.setItem(0, 2, change_item)
        else:
            self.price_history_table.setItem(0, 2, QTableWidgetItem("--"))
        
        if 'volume' in stats:
            volume = float(stats['volume'])
            if volume >= 1_000_000:
                volume_text = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                volume_text = f"{volume/1_000:.1f}K"
            else:
                volume_text = f"{volume:.0f}"
            self.price_history_table.setItem(0, 3, QTableWidgetItem(volume_text))
        else:
            self.price_history_table.setItem(0, 3, QTableWidgetItem("--"))
        
        trend_item = QTableWidgetItem(f"{trend_emoji} {trend_text}")
        trend_item.setForeground(trend_color)
        self.price_history_table.setItem(0, 4, trend_item)
        
        if self.price_history_table.rowCount() > 25:
            self.price_history_table.removeRow(25)
    
    def add_ml_signal(self, signal_data):
        """Add ML signal with comprehensive tracking and analytics"""
        try:
            current_time = signal_data['timestamp'].strftime("%H:%M:%S")
            
            self.signals_table.insertRow(0)
            
            self.signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            signal_text = signal_data['signal']
            confidence = signal_data['confidence']
            
            if signal_data.get('is_ml_trained', False):
                display_signal = f"{signal_text} 🤖"
            else:
                display_signal = f"{signal_text} 📋"
            
            signal_item = QTableWidgetItem(display_signal)
            
            if signal_text == "BUY":
                if confidence >= 80:
                    signal_item.setForeground(QColor(0, 255, 136))
                else:
                    signal_item.setForeground(QColor(136, 255, 136))
            elif signal_text == "SELL":
                if confidence >= 80:
                    signal_item.setForeground(QColor(255, 68, 68))
                else:
                    signal_item.setForeground(QColor(255, 136, 136))
            else:
                signal_item.setForeground(QColor(255, 255, 102))
            
            self.signals_table.setItem(0, 2, signal_item)
            self.signals_table.setItem(0, 3, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            
            confidence_item = QTableWidgetItem(f"{confidence}%")
            if confidence >= 85:
                confidence_item.setForeground(QColor(0, 255, 136))
            elif confidence >= 70:
                confidence_item.setForeground(QColor(255, 255, 102))
            else:
                confidence_item.setForeground(QColor(255, 165, 0))
            
            self.signals_table.setItem(0, 4, confidence_item)
            
            source = "🤖 ML" if signal_data.get('is_ml_trained', False) else "📋 Rules"
            self.signals_table.setItem(0, 5, QTableWidgetItem(source))
            
            # Add to ML signals table
            self.ml_signals_table.insertRow(0)
            self.ml_signals_table.setItem(0, 0, QTableWidgetItem(current_time))
            self.ml_signals_table.setItem(0, 1, QTableWidgetItem(signal_data['symbol']))
            
            ml_signal_item = QTableWidgetItem(display_signal)
            ml_signal_item.setForeground(signal_item.foreground())
            self.ml_signals_table.setItem(0, 2, ml_signal_item)
            
            ml_confidence_item = QTableWidgetItem(f"{confidence}%")
            ml_confidence_item.setForeground(confidence_item.foreground())
            self.ml_signals_table.setItem(0, 3, ml_confidence_item)
            
            self.ml_signals_table.setItem(0, 4, QTableWidgetItem(f"${signal_data['price']:,.4f}"))
            self.ml_signals_table.setItem(0, 5, QTableWidgetItem(
                signal_data.get('model_type', 'Unknown').replace('_', ' ').title()
            ))
            self.ml_signals_table.setItem(0, 6, QTableWidgetItem(
                "✅" if signal_data.get('is_ml_trained', False) else "❌"
            ))
            
            self.update_signal_analytics(signal_data)
            
            for table in [self.signals_table, self.ml_signals_table]:
                if table.rowCount() > 20:
                    table.removeRow(20)
                    
        except Exception as e:
            print(f"Error adding ML signal: {e}")
    
    def update_signal_analytics(self, signal_data):
        """Update comprehensive signal analytics"""
        try:
            self.signal_performance.append(signal_data)
            
            if len(self.signal_performance) > 100:
                self.signal_performance.pop(0)
            
            total_signals = len(self.signal_performance)
            buy_signals = sum(1 for s in self.signal_performance if s['signal'] == 'BUY')
            sell_signals = sum(1 for s in self.signal_performance if s['signal'] == 'SELL')
            hold_signals = sum(1 for s in self.signal_performance if s['signal'] == 'HOLD')
            
            self.total_signals_label.setText(f"Total Signals: {total_signals}")
            self.buy_signals_label.setText(f"Buy Signals: {buy_signals}")
            self.sell_signals_label.setText(f"Sell Signals: {sell_signals}")
            self.hold_signals_label.setText(f"Hold Signals: {hold_signals}")
            
            if buy_signals > sell_signals:
                self.buy_signals_label.setStyleSheet("color: #00ff88; font-weight: bold;")
                self.sell_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            elif sell_signals > buy_signals:
                self.sell_signals_label.setStyleSheet("color: #ff4444; font-weight: bold;")
                self.buy_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            else:
                self.buy_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
                self.sell_signals_label.setStyleSheet("color: #ffffff; font-weight: normal;")
            
            self.hold_signals_label.setStyleSheet("color: #ffff66; font-weight: bold;")
            
            self.update_performance_table()
            
        except Exception as e:
            print(f"Error updating signal analytics: {e}")
    
    def update_performance_table(self):
        """Update performance table with time-based analysis"""
        try:
            if not self.signal_performance:
                return
            
            self.performance_table.setRowCount(0)
            
            now = datetime.datetime.now()
            time_periods = [
                ("Last 5 minutes", 5),
                ("Last 15 minutes", 15),
                ("Last 30 minutes", 30),
                ("Last 1 hour", 60),
                ("Last 2 hours", 120)
            ]
            for period_name, minutes in time_periods:
                cutoff_time = now - datetime.timedelta(minutes=minutes)
                period_signals = [
                    s for s in self.signal_performance
                    if s['timestamp'] >= cutoff_time
                ]
                
                if period_signals:
                    signal_count = len(period_signals)
                    avg_confidence = sum(s['confidence'] for s in period_signals) / signal_count
                    
                    # Find best signal (highest confidence)
                    best_signal = max(period_signals, key=lambda x: x['confidence'])
                    best_signal_text = f"{best_signal['signal']} ({best_signal['confidence']}%)"
                    
                    # Determine dominant model
                    models = [s.get('model_type', 'Unknown') for s in period_signals]
                    model_counts = {}
                    for model in models:
                        model_counts[model] = model_counts.get(model, 0) + 1
                    dominant_model = max(model_counts.items(), key=lambda x: x[1])[0]
                    
                    # Add row to table with color coding
                    row_position = self.performance_table.rowCount()
                    self.performance_table.insertRow(row_position)
                    
                    self.performance_table.setItem(row_position, 0, QTableWidgetItem(period_name))
                    self.performance_table.setItem(row_position, 1, QTableWidgetItem(str(signal_count)))
                    
                    # Color code confidence levels
                    conf_item = QTableWidgetItem(f"{avg_confidence:.1f}%")
                    if avg_confidence >= 80:
                        conf_item.setForeground(QColor(0, 255, 136))
                    elif avg_confidence >= 70:
                        conf_item.setForeground(QColor(255, 255, 102))
                    else:
                        conf_item.setForeground(QColor(255, 165, 0))
                    self.performance_table.setItem(row_position, 2, conf_item)
                    
                    self.performance_table.setItem(row_position, 3, QTableWidgetItem(best_signal_text))
                    self.performance_table.setItem(row_position, 4, 
                        QTableWidgetItem(dominant_model.replace('_', ' ').title()))
                
        except Exception as e:
            print(f"Error updating performance table: {e}")
    
    # ===== COMPREHENSIVE ML SYSTEM MANAGEMENT =====
    
    def update_ml_status(self, status):
        """Update ML system status with comprehensive visual feedback"""
        self.ml_status.setText(f"ML: {status}")
        self.ml_detailed_status.setText(f"Status: {status}")
        
        # Intelligent color coding based on status content
        if any(indicator in status for indicator in ["✅", "trained", "success", "loaded"]):
            color_style = "color: #00ff88; font-weight: bold;"
        elif any(indicator in status for indicator in ["❌", "error", "failed"]):
            color_style = "color: #ff4444; font-weight: bold;"
        elif any(indicator in status for indicator in ["🔄", "training", "loading", "fetching"]):
            color_style = "color: #ffff66; font-weight: bold;"
        else:
            color_style = "color: #ffffff; font-weight: bold;"
        
        self.ml_status.setStyleSheet(color_style)
        self.ml_detailed_status.setStyleSheet(color_style)
    
    def update_training_progress(self, message):
        """Update training progress with intelligent progress tracking"""
        self.training_progress.setVisible(True)
        
        # Sophisticated progress calculation based on training stages
        progress_mapping = {
            "fetching": 20,
            "loaded": 40,
            "training": 70,
            "completed": 100,
            "success": 100
        }
        
        progress_value = 0
        for keyword, value in progress_mapping.items():
            if keyword in message.lower():
                progress_value = value
                break
        
        self.training_progress.setValue(progress_value)
        
        # Auto-hide progress bar after completion
        if progress_value == 100:
            QTimer.singleShot(3000, lambda: self.training_progress.setVisible(False))
        
        # Update accuracy and feature importance if training completed
        if (self.enhanced_data_worker and 
            self.enhanced_data_worker.ml_generator.is_trained and 
            progress_value == 100):
            
            accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
            self.ml_accuracy_label.setText(f"Model Accuracy: {accuracy:.1%}")
            
            # Update feature importance display
            feature_importance = self.enhanced_data_worker.ml_generator.get_feature_importance()
            self.update_feature_importance_display(feature_importance)
    
    def update_feature_importance_display(self, feature_importance):
        """Update feature importance table with sophisticated visualization"""
        try:
            self.feature_importance_table.setRowCount(0)
            
            # Display top 10 most important features
            top_features = feature_importance[:10] if feature_importance else []
            
            for i, (feature_name, importance) in enumerate(top_features):
                self.feature_importance_table.insertRow(i)
                
                # Format feature name for better readability
                display_name = feature_name.replace('_', ' ').title()
                self.feature_importance_table.setItem(i, 0, QTableWidgetItem(display_name))
                
                # Color-coded importance values
                importance_item = QTableWidgetItem(f"{importance:.4f}")
                if importance > 0.1:
                    importance_item.setForeground(QColor(0, 255, 136))  # High importance
                elif importance > 0.05:
                    importance_item.setForeground(QColor(255, 255, 102))  # Medium importance
                else:
                    importance_item.setForeground(QColor(255, 165, 0))  # Low importance
                
                self.feature_importance_table.setItem(i, 1, importance_item)
                
        except Exception as e:
            print(f"Error updating feature importance: {e}")
    
    def browse_model_file(self):
        """Browse for model file with comprehensive file filtering"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select ML Model File", 
            "", 
            "Model Files (*.pkl *.joblib);;Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            self.model_file_input.setText(file_path)
    
    def save_ml_model(self):
        """Save current ML model with comprehensive validation and user feedback"""
        if not self.enhanced_data_worker or not self.enhanced_data_worker.ml_generator.is_trained:
            QMessageBox.warning(self, "Save Model", 
                "❌ No trained model available to save.\n\n"
                "Please train a model first by starting the ML trading system.")
            return
        
        # Generate intelligent default filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"{self.current_symbol}_{self.current_model}_{timestamp}.pkl"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save ML Model", 
            default_filename,
            "Model Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            try:
                # Ensure proper file extension
                if not file_path.endswith('.pkl'):
                    file_path += '.pkl'
                
                success = self.enhanced_data_worker.ml_generator.save_model(file_path, self.current_model)
                
                if success:
                    accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
                    QMessageBox.information(self, "Save Model", 
                        f"✅ Model saved successfully!\n\n"
                        f"📁 Location: {file_path}\n"
                        f"🎯 Accuracy: {accuracy:.1%}\n"
                        f"🤖 Model Type: {self.current_model.replace('_', ' ').title()}")
                    self.model_file_input.setText(file_path)
                else:
                    QMessageBox.warning(self, "Save Model", 
                        "❌ Failed to save model.\n\n"
                        "Please check file permissions and try again.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Save Model", 
                    f"❌ Error saving model:\n\n{str(e)}")
    
    def load_ml_model(self):
        """Load pre-trained ML model with comprehensive validation"""
        model_path = self.model_file_input.text().strip()
        
        if not model_path:
            QMessageBox.warning(self, "Load Model", 
                "📁 Please select a model file first.\n\n"
                "Use the Browse button to select a .pkl model file.")
            return
        
        if not self.enhanced_data_worker:
            QMessageBox.warning(self, "Load Model", 
                "🚀 Please start the trading system first.\n\n"
                "The ML system must be initialized before loading a model.")
            return
        
        try:
            success = self.enhanced_data_worker.ml_generator.load_model(model_path)
            
            if success:
                accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
                QMessageBox.information(self, "Load Model", 
                    f"✅ Model loaded successfully!\n\n"
                    f"📁 Source: {model_path}\n"
                    f"🎯 Accuracy: {accuracy:.1%}")
                
                self.update_ml_status(f"✅ Model loaded (Accuracy: {accuracy:.1%})")
                self.ml_accuracy_label.setText(f"Model Accuracy: {accuracy:.1%}")
                
                # Update feature importance display
                feature_importance = self.enhanced_data_worker.ml_generator.get_feature_importance()
                self.update_feature_importance_display(feature_importance)
                
                # Update model info
                self.ml_model_info.setText(f"Current Model: Loaded from file")
                
            else:
                QMessageBox.warning(self, "Load Model", 
                    "❌ Failed to load model.\n\n"
                    "Please verify the file format and try again.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Load Model", 
                f"❌ Error loading model:\n\n{str(e)}")
    
    def retrain_ml_model(self):
        """Manually trigger comprehensive ML model retraining"""
        if not self.enhanced_data_worker:
            QMessageBox.warning(self, "Retrain Model", 
                "🚀 Please start the trading system first.\n\n"
                "The data collection system must be active for retraining.")
            return
        
        if self.current_model == "fallback":
            QMessageBox.information(self, "Retrain Model", 
                "📋 Fallback rules don't require training.\n\n"
                "Switch to Random Forest or XGBoost for ML training.")
            return
        
        reply = QMessageBox.question(self, "Retrain Model", 
            f"🔄 Retrain {self.current_model.replace('_', ' ').title()} Model\n\n"
            f"This will:\n"
            f"• Fetch fresh historical data\n"
            f"• Retrain the model from scratch\n"
            f"• Take several minutes to complete\n\n"
            f"Continue with retraining?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            self.update_ml_status("🔄 Retraining model...")
            
            # Trigger retraining in background thread
            self.enhanced_data_worker.initialize_ml_system()
    
    def export_signal_history(self):
        """Export comprehensive signal history to CSV with metadata"""
        if not self.signal_performance:
            QMessageBox.warning(self, "Export Signals", 
                "📊 No signal data available to export.\n\n"
                "Please run the trading system to generate signals first.")
            return
        
        # Generate intelligent filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"signal_history_{self.current_symbol}_{timestamp}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Signal History", 
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                import csv
                
                # Ensure proper file extension
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'timestamp', 'symbol', 'signal', 'confidence', 'price', 
                        'model_type', 'is_ml_trained', 'features_summary'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header with metadata
                    writer.writeheader()
                    
                    for signal in self.signal_performance:
                        # Create features summary for easier analysis
                        features = signal.get('features', {})
                        features_summary = f"RSI:{features.get('rsi', 0):.1f} MACD:{features.get('macd_histogram', 0):.4f}"
                        
                        row = {
                            'timestamp': signal['timestamp'].isoformat(),
                            'symbol': signal['symbol'],
                            'signal': signal['signal'],
                            'confidence': signal['confidence'],
                            'price': signal['price'],
                            'model_type': signal.get('model_type', 'Unknown'),
                            'is_ml_trained': signal.get('is_ml_trained', False),
                            'features_summary': features_summary
                        }
                        writer.writerow(row)
                
                signal_count = len(self.signal_performance)
                QMessageBox.information(self, "Export Signals", 
                    f"✅ Signal history exported successfully!\n\n"
                    f"📁 Location: {file_path}\n"
                    f"📊 Records: {signal_count} signals\n"
                    f"🕒 Time Range: Recent trading session")
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Signals", 
                    f"❌ Error exporting signals:\n\n{str(e)}")
    
    # ===== ENHANCED API CONFIGURATION MANAGEMENT =====
    
    def save_api_config(self):
        """Save API configuration with enhanced security and validation"""
        api_key = self.api_key_input.text().strip()
        api_secret = self.api_secret_input.text().strip()
        
        if not api_key or not api_secret:
            QMessageBox.warning(self, "API Configuration", 
                "🔐 Please enter both API key and secret.\n\n"
                "Both credentials are required for API access.")
            return
        
        # Enhanced validation
        if len(api_key) < 32 or len(api_secret) < 32:
            reply = QMessageBox.question(self, "API Configuration", 
                "⚠️ API credentials appear to be too short.\n\n"
                "Binance API keys are typically 64 characters long.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        
        # Save configuration
        self.binance_api.api_key = api_key
        self.binance_api.api_secret = api_secret
        self.binance_api.use_testnet = self.testnet_checkbox.isChecked()
        
        self.update_connection_status("✅ API credentials saved - Ready for testing")
        
        # Update config display with security masking
        masked_key = api_key[:8] + "..." + api_key[-8:] if len(api_key) > 16 else "***"
        env_text = "🧪 Testnet" if self.binance_api.use_testnet else "🔴 Live Trading"
        
        self.config_info.setPlainText(
            f"🔧 Configuration Details:\n"
            f"• Key Type: {self.binance_api.config['key_type']}\n"
            f"• Description: {self.binance_api.config['description']}\n"
            f"• API Key: {masked_key}\n"
            f"• Environment: {env_text}\n"
            f"• Status: ✅ Configured\n\n"
            f"{'⚠️ Using testnet for safety!' if self.binance_api.use_testnet else '🚨 LIVE TRADING MODE - BE EXTREMELY CAREFUL!'}"
        )
        
        QMessageBox.information(self, "API Configuration", 
            f"✅ API credentials saved successfully!\n\n"
            f"🌐 Environment: {env_text}\n"
            f"🔗 Ready for connection testing")
    
    def test_api_connection(self):
        """Test API connection with comprehensive validation and feedback"""
        if not self.binance_api.api_key or not self.binance_api.api_secret:
            QMessageBox.warning(self, "Test Connection", 
                "🔐 Please configure API credentials first.\n\n"
                "Go to the API Configuration tab to set up your credentials.")
            return
        
        self.update_connection_status("🔄 Testing API connection...")
        
        try:
            # Test 1: Server connectivity
            server_time = self.binance_api.get_server_time()
            if not server_time:
                self.update_connection_status("❌ Cannot reach Binance servers")
                QMessageBox.critical(self, "Connection Test", 
                    "❌ Server Connection Failed\n\n"
                    "Cannot reach Binance servers.\n"
                    "Please check your internet connection.")
                return
            
            # Test 2: API credentials validation
            account_info = self.binance_api.get_account_info()
            
            if "error" in account_info:
                error_msg = account_info.get('error', 'Unknown error')
                self.update_connection_status(f"❌ API Error: {error_msg}")
                QMessageBox.critical(self, "Connection Test", 
                    f"❌ API Authentication Failed\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Please verify:\n"
                    f"• API key and secret are correct\n"
                    f"• API permissions are properly set\n"
                    f"• Using correct environment (testnet/live)")
            else:
                self.update_connection_status("✅ API connection verified")
                
                # Extract account details for user feedback
                account_type = account_info.get('accountType', 'Unknown')
                can_trade = account_info.get('canTrade', False)
                env_text = "Testnet" if self.binance_api.use_testnet else "Live Trading"
                
                QMessageBox.information(self, "Connection Test", 
                    f"✅ Connection Test Successful!\n\n"
                    f"🏦 Account Type: {account_type}\n"
                    f"📈 Trading Enabled: {'Yes' if can_trade else 'No'}\n"
                    f"🌐 Environment: {env_text}\n\n"
                    f"Your API is properly configured and ready for use!")
                
                # Auto-refresh account info on successful connection
                self.refresh_account_info()
                
        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            self.update_connection_status(f"❌ {error_msg}")
            QMessageBox.critical(self, "Connection Test", 
                f"❌ Connection Test Failed\n\n{error_msg}")
    
    def clear_api_config(self):
        """Clear API configuration with security confirmation"""
        reply = QMessageBox.question(self, "Clear Configuration", 
            "🗑️ Clear API Configuration\n\n"
            "This will permanently remove all API credentials from the application.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Clear all credential data
            self.api_key_input.clear()
            self.api_secret_input.clear()
            self.binance_api.api_key = ""
            self.binance_api.api_secret = ""
            self.testnet_checkbox.setChecked(True)
            
            # Reset configuration display
            self.config_info.setPlainText(
                f"🔧 Configuration Details:\n"
                f"• Key Type: {self.binance_api.config['key_type']}\n"
                f"• Description: {self.binance_api.config['description']}\n"
                f"• Environment: 🧪 Testnet\n"
                f"• Status: ❌ Not configured\n\n"
                f"⚠️ Please enter your API credentials to enable live trading."
            )
            
            # Clear account info
            self.account_info_display.setPlainText(
                "🔐 Account Information\n" + "="*50 + "\n\n"
                "API credentials have been cleared.\n"
                "Please reconfigure to view account details.")
            
            self.update_connection_status("🔄 Configuration cleared - Enter API credentials")
    
    def refresh_account_info(self):
        """Refresh account information with comprehensive display formatting"""
        if not self.binance_api.api_key or not self.binance_api.api_secret:
            self.account_info_display.setPlainText(
                "🔐 Account Information\n" + "="*50 + "\n\n"
                "Please configure API credentials to view account details.\n\n"
                "Available after configuration:\n"
                "• Real-time balance information\n"
                "• Trading permissions status\n"
                "• Account type and limits")
            return
        
        account_info = self.binance_api.get_account_info()
        
        if "error" in account_info:
            self.account_info_display.setPlainText(
                f"❌ Error loading account information:\n\n{account_info['error']}\n\n"
                f"Please check your API credentials and try again.")
        else:
            # Format comprehensive account information
            formatted_info = "💰 ACCOUNT INFORMATION\n" + "="*60 + "\n\n"
            
            # Account basics
            formatted_info += "🏦 ACCOUNT DETAILS:\n" + "-"*30 + "\n"
            formatted_info += f"Account Type: {account_info.get('accountType', 'Unknown')}\n"
            formatted_info += f"Trading Enabled: {'✅ Yes' if account_info.get('canTrade', False) else '❌ No'}\n"
            formatted_info += f"Withdraw Enabled: {'✅ Yes' if account_info.get('canWithdraw', False) else '❌ No'}\n"
            formatted_info += f"Deposit Enabled: {'✅ Yes' if account_info.get('canDeposit', False) else '❌ No'}\n\n"
            
            # Process and display balances
            if 'balances' in account_info:
                formatted_info += "💰 ASSET BALANCES:\n" + "-"*30 + "\n"
                
                significant_balances = []
                for balance in account_info['balances']:
                    free_balance = float(balance['free'])
                    locked_balance = float(balance['locked'])
                    total_balance = free_balance + locked_balance
                    
                    # Only show balances with meaningful amounts
                    if total_balance > 0.00000001:  # Avoid dust amounts
                        significant_balances.append({
                            'asset': balance['asset'],
                            'free': free_balance,
                            'locked': locked_balance,
                            'total': total_balance
                        })
                
                if significant_balances:
                    # Sort by total balance (descending)
                    significant_balances.sort(key=lambda x: x['total'], reverse=True)
                    
                    # Display top 15 balances
                    for balance in significant_balances[:15]:
                        formatted_info += f"  {balance['asset']:>8}: {balance['free']:>15.8f}"
                        if balance['locked'] > 0:
                            formatted_info += f" + {balance['locked']:>12.8f} (locked)"
                        formatted_info += f" = {balance['total']:>15.8f}\n"
                    
                    if len(significant_balances) > 15:
                        formatted_info += f"  ... and {len(significant_balances) - 15} more assets\n"
                    
                    formatted_info += f"\n📊 Total Assets with Balance: {len(significant_balances)}\n"
                else:
                    formatted_info += "  No significant balances found.\n"
            
            # Trading permissions
            formatted_info += f"\n🔐 PERMISSIONS:\n" + "-"*30 + "\n"
            permissions = account_info.get('permissions', [])
            if permissions:
                for permission in permissions:
                    formatted_info += f"  ✅ {permission.upper()}\n"
            else:
                formatted_info += "  No specific permissions listed.\n"
            
            # Environment and timestamp
            env_text = "🧪 Testnet" if self.binance_api.use_testnet else "🔴 Live Trading"
            formatted_info += f"\n🌐 Environment: {env_text}\n"
            formatted_info += f"🕒 Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.account_info_display.setPlainText(formatted_info)
    
    # ===== UTILITY METHODS AND STATUS MANAGEMENT =====
    
    def update_connection_status(self, status):
        """Update connection status with intelligent color coding and feedback"""
        self.connection_status.setText(f"Connection: {status}")
        
        # Sophisticated status analysis for color coding
        positive_indicators = ["success", "verified", "active", "✅", "saved", "ready"]
        negative_indicators = ["error", "failed", "❌", "cannot", "denied"]
        processing_indicators = ["testing", "loading", "🔄", "fetching", "connecting"]
        
        if any(indicator in status.lower() for indicator in positive_indicators):
            self.connection_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        elif any(indicator in status.lower() for indicator in negative_indicators):
            self.connection_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        elif any(indicator in status.lower() for indicator in processing_indicators):
            self.connection_status.setStyleSheet("color: #ffff66; font-weight: bold;")
        else:
            self.connection_status.setStyleSheet("color: #ffffff; font-weight: bold;")
    
    # ===== COMPREHENSIVE ML SYSTEM MANAGEMENT =====
    
    def update_ml_status(self, status):
        """Update ML system status with comprehensive visual feedback"""
        self.ml_status.setText(f"ML: {status}")
        self.ml_detailed_status.setText(f"Status: {status}")
        
        # Intelligent color coding based on status content
        positive_indicators = ["✅", "trained", "success", "loaded", "verified", "ready"]
        negative_indicators = ["❌", "error", "failed", "denied", "cannot"]
        processing_indicators = ["🔄", "training", "loading", "fetching", "initializing"]
        
        if any(indicator in status.lower() for indicator in positive_indicators):
            color_style = "color: #00ff88; font-weight: bold;"
        elif any(indicator in status.lower() for indicator in negative_indicators):
            color_style = "color: #ff4444; font-weight: bold;"
        elif any(indicator in status.lower() for indicator in processing_indicators):
            color_style = "color: #ffff66; font-weight: bold;"
        else:
            color_style = "color: #ffffff; font-weight: bold;"
        
        self.ml_status.setStyleSheet(color_style)
        self.ml_detailed_status.setStyleSheet(color_style)
    
    def update_training_progress(self, message):
        """Update training progress with intelligent progress tracking"""
        self.training_progress.setVisible(True)
        
        # Sophisticated progress calculation based on training stages
        progress_mapping = {
            "fetching historical data": 10,
            "loaded 5m": 25,
            "loaded 15m": 45,
            "loaded 1h": 65,
            "training ml model": 80,
            "model trained": 100,
            "training completed": 100,
            "success": 100
        }
        
        progress_value = 0
        for keyword, value in progress_mapping.items():
            if keyword in message.lower():
                progress_value = value
                break
        
        self.training_progress.setValue(progress_value)
        
        # Auto-hide progress bar after completion
        if progress_value == 100:
            QTimer.singleShot(3000, lambda: self.training_progress.setVisible(False))
        
        # Update accuracy and feature importance if training completed
        if (self.enhanced_data_worker and 
            self.enhanced_data_worker.ml_generator.is_trained and 
            progress_value == 100):
            
            accuracy = self.enhanced_data_worker.ml_generator.training_accuracy
            self.ml_accuracy_label.setText(f"Model Accuracy: {accuracy:.1%}")
            
            # Update feature importance display
            feature_importance = self.enhanced_data_worker.ml_generator.get_feature_importance()
            self.update_feature_importance_display(feature_importance)
    
    def update_feature_importance_display(self, feature_importance):
        """Update feature importance table with sophisticated visualization"""
        try:
            self.feature_importance_table.setRowCount(0)
            
            # Display top 10 most important features
            top_features = feature_importance[:10] if feature_importance else []
            
            for i, (feature_name, importance) in enumerate(top_features):
                self.feature_importance_table.insertRow(i)
                
                # Format feature name for better readability
                display_name = feature_name.replace('_', ' ').title()
                self.feature_importance_table.setItem(i, 0, QTableWidgetItem(display_name))
                
                # Color-coded importance values
                importance_item = QTableWidgetItem(f"{importance:.4f}")
                if importance > 0.1:
                    importance_item.setForeground(QColor(0, 255, 136))  # High importance
                elif importance > 0.05:
                    importance_item.setForeground(QColor(255, 255, 102))  # Medium importance
                else:
                    importance_item.setForeground(QColor(255, 165, 0))  # Low importance
                
                self.feature_importance_table.setItem(i, 1, importance_item)
                
        except Exception as e:
            print(f"Error updating feature importance: {e}")
    
    # ===== ENHANCED APPLICATION LIFECYCLE MANAGEMENT =====
    
    def closeEvent(self, event):
        """Handle application close with comprehensive cleanup and user confirmation"""
        if self.is_trading:
            reply = QMessageBox.question(self, "Exit Application", 
                "🚀 Trading System Active\n\n"
                "The ML trading system is currently running.\n"
                "Closing will stop all data collection and signal generation.\n\n"
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        
        try:
            # Comprehensive cleanup sequence
            print("🔄 Initiating application cleanup...")
            
            if self.enhanced_data_worker:
                print("⏹️ Stopping data worker thread...")
                self.enhanced_data_worker.stop_updates()
                self.enhanced_data_worker = None
            
            # Clear sensitive data from memory
            if hasattr(self, 'binance_api'):
                self.binance_api.api_key = ""
                self.binance_api.api_secret = ""
            
            # Clear signal performance data
            if hasattr(self, 'signal_performance'):
                self.signal_performance.clear()
            
            print("✅ Application cleanup completed successfully")
            event.accept()
            
        except Exception as e:
            print(f"⚠️ Error during application cleanup: {e}")
            event.accept()  # Still allow exit even if cleanup fails

# ===== ENHANCED DATA WORKER WITH MISSING SIGNALS =====

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

                    # Latest candlestick
                    candle_df = self.binance_api.get_historical_klines(self.symbol, "1m", 2)
                    candle_dict = None
                    if "error" not in candle_df and not candle_df.empty:
                        candle_dict = candle_df.iloc[-1].to_dict()
                    
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
                        'candle': candle_dict,
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

# ===== COMPREHENSIVE MAIN APPLICATION ENTRY POINT =====

def main():
    """
    Main application entry point with comprehensive system validation and error handling
    
    This function provides:
    - Complete system requirements validation
    - Intelligent dependency checking
    - User-friendly error messages and guidance
    - Graceful degradation for missing optional components
    - Professional startup sequence with progress feedback
    """
    try:
        print("="*70)
        print("🤖 ADVANCED ML BINANCE TRADING APPLICATION")
        print("="*70)
        print("Version: 2.0.0 - Production Ready")
        print("Features: Real Machine Learning, Live Market Data, Professional UI")
        print("Safety: Comprehensive Error Handling, Testnet Support")
        print("="*70)
        
        # Create QApplication with enhanced configuration
        app = QApplication(sys.argv)
        app.setApplicationName("🤖 Advanced ML Binance Trading Bot")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("ML Trading Solutions")
        app.setOrganizationDomain("mltrading.solutions")
        
        # Set high-quality application style and attributes
        app.setStyle('Fusion')
        
        # Enable high DPI scaling for modern displays
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Comprehensive system requirements validation
        print("🔍 Validating System Requirements...")
        print("-" * 40)
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 7):
            print(f"❌ Python version: {python_version.major}.{python_version.minor} (Requires 3.7+)")
            print("💡 Please upgrade Python to version 3.7 or higher")
            return 1
        else:
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages with detailed feedback
        required_packages = {
            'pandas': 'Data manipulation and analysis',
            'numpy': 'Numerical computing',
            'sklearn': 'Machine learning algorithms',
            'ta': 'Technical analysis indicators',
            'requests': 'HTTP client for API calls',
            'PyQt5': 'GUI framework'
        }
        
        optional_packages = {
            'xgboost': 'Advanced gradient boosting (optional)',
            'joblib': 'Model persistence (optional)'
        }
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package, description in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package:<12} - {description}")
            except ImportError:
                missing_required.append(package)
                print(f"❌ {package:<12} - MISSING - {description}")
        
        # Check optional packages
        for package, description in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package:<12} - {description}")
            except ImportError:
                missing_optional.append(package)
                print(f"⚠️  {package:<12} - OPTIONAL - {description}")
        
        # Handle missing packages
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print("💡 Install with:")
            print(f"   pip install {' '.join(missing_required)}")
            return 1
        
        if missing_optional:
            print(f"\n⚠️  Optional packages not found: {', '.join(missing_optional)}")
            print("💡 For full functionality, install with:")
            print(f"   pip install {' '.join(missing_optional)}")
            print("   (Application will run with reduced features)")
        
        print("\n✅ All critical system requirements satisfied")
        print("🚀 Initializing application components...")
        
        # Create and configure main window
        window = BinanceTradingApp()
        
        # Set application icon if available
        try:
            # You can add an icon file here
            # window.setWindowIcon(QIcon('icon.png'))
            pass
        except:
            pass
        
        # Show window with fade-in effect (optional)
        window.show()
        
        print("✅ Application started successfully!")
        print("\n📋 Quick Start Guide:")
        print("1. 🔐 Configure API credentials in 'API Configuration' tab")
        print("2. 🤖 Select your preferred ML model (Random Forest recommended)")
        print("3. 📈 Choose a trading symbol (BTCUSDT is default)")
        print("4. 🚀 Click 'Start ML Trading' to begin")
        print("5. 📊 Monitor signals in the 'Live Trading' tab")
        print("\n⚠️  Important: Start with testnet for safety!")
        print("="*70)
        
        # Run the application event loop
        return app.exec_()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Missing dependencies. Install with:")
        print("   pip install PyQt5 pandas numpy scikit-learn ta requests")
        print("   pip install xgboost joblib  # Optional but recommended")
        return 1
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print("💡 Please check your Python installation and try again")
        print("📧 Report issues at: https://github.com/your-repo/issues")
        return 1

if __name__ == "__main__":
    """
    Application entry point with comprehensive error handling and user guidance
    
    🎯 Key Features:
    ├── 📊 Real-time Binance market data integration
    ├── 🤖 Advanced machine learning signal generation
    ├── 🔬 Multiple ML models (Random Forest, XGBoost, Rule-based)
    ├── 📈 25+ technical indicators for comprehensive analysis
    ├── 🧠 Intelligent feature engineering and model training
    ├── 🎨 Professional dark-themed UI with real-time updates
    ├── 🛡️ Robust error handling and data validation
    ├── 💾 Export capabilities and model persistence
    ├── 🧪 Testnet support for safe development
    └── 📱 Cross-platform compatibility
    
    🚀 Getting Started:
    1. Install dependencies: pip install PyQt5 pandas numpy scikit-learn ta requests xgboost
    2. Run the application: python reep.py
    3. Configure API credentials (use testnet first!)
    4. Select ML model and trading symbol
    5. Start trading and monitor real-time signals
    
    ⚠️  Safety Notice:
    - Always test with Binance testnet first
    - Never invest more than you can afford to lose
    - This is educational software, not financial advice
    """
    
    try:
        exit_code = main()
        print(f"\n🔄 Application exited with code: {exit_code}")
        
        if exit_code == 0:
            print("✅ Clean shutdown completed")
        else:
            print("⚠️ Application encountered issues during execution")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user (Ctrl+C)")
        print("🔄 Performing emergency cleanup...")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Unexpected fatal error: {e}")
        print("📧 Please report this issue with the full error log")
        sys.exit(1)
