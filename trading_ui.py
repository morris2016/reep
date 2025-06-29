"""
User Interface for the Advanced ML Binance Trading Bot.

This module provides the main application window and all UI components for interacting
with the trading bot. It uses PyQt5 for the GUI.

Key features managed by the UI:
- API Configuration and Connection Testing.
- Live Market Data Display (price, stats, charts).
- ML Model Selection and Management (training, loading, saving).
- Manual and Automated Trading Controls.
- Display of Trading Signals, Price History, and Executed Trades.
- Analytics and Performance Tracking.
- Logging integration for UI events and errors.
"""
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QGroupBox, QFormLayout,
    QTextEdit, QTabWidget, QMessageBox, QFileDialog,
    QProgressBar, QCheckBox, QSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QPolygonF, QIcon, QDoubleValidator, QIntValidator
from collections import deque
import sys
import datetime
from trading_logic import EnhancedBinanceAPI, EnhancedDataWorker, logger as logic_logger
from database import init_db, insert_trade, fetch_trades
import logging
import csv
import os # For icon path
from typing import List, Dict, Tuple, Optional, Any, Deque


ui_logger = logging.getLogger(__name__)
if not ui_logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s")
    try:
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_app.log")
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        ui_logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        print(f"Critical: Could not open UI log file for writing: {e}", file=sys.stderr)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    ui_logger.addHandler(stream_handler)
    ui_logger.setLevel(logging.INFO)


class PriceChartWidget(QWidget):
    """
    A widget to display a live price history as a simple line chart.
    """
    def __init__(self, parent: Optional[QWidget] = None, max_points: int = 150) -> None: # Increased default max_points
        super().__init__(parent)
        self.prices: Deque[float] = deque(maxlen=max_points)
        self.setMinimumHeight(200) # Increased min height
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add_price(self, price: float) -> None:
        if not isinstance(price, (int, float)):
            ui_logger.warning(f"PriceChartWidget received invalid price type: {type(price)}, value: {price}")
            return
        self.prices.append(price)
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#2A2A2A")) # Slightly lighter dark background

        rect = self.rect()
        w: int = rect.width()
        h: int = rect.height()

        if len(self.prices) < 2:
            painter.setPen(QColor(180, 180, 180)) # Lighter text for "no data"
            painter.drawText(rect, Qt.AlignCenter, "Awaiting sufficient market data for chart...")
            return

        try:
            valid_prices = [p for p in self.prices if isinstance(p, (int, float))]
            if len(valid_prices) < 2:
                 painter.setPen(QColor(180,180,180)); painter.drawText(rect, Qt.AlignCenter, "Collecting more data points..."); return

            min_price: float = min(valid_prices)
            max_price: float = max(valid_prices)
            price_span: float = max_price - min_price

            poly = QPolygonF()
            padding: int = 10 # Increased padding
            chart_h: float = h - 2 * padding
            chart_w: float = w - 2 * padding

            if chart_h <= 0 or chart_w <=0 : return # Not enough space to draw

            if price_span == 0: # Flat line
                y_coord: float = padding + chart_h / 2.0
                poly.append(QPointF(padding, y_coord))
                poly.append(QPointF(padding + chart_w, y_coord))
            else:
                x_step: float = chart_w / (len(valid_prices) -1 ) if len(valid_prices) > 1 else chart_w
                for i, price_val in enumerate(valid_prices):
                    x: float = padding + i * x_step
                    y: float = padding + chart_h - (((price_val - min_price) / price_span) * chart_h)
                    poly.append(QPointF(x, y))

            painter.setPen(QPen(QColor(0, 220, 110), 2.5)) # Slightly thicker, vibrant line
            painter.drawPolyline(poly)

            font = painter.font(); font.setPointSize(9); font.setBold(True) # Slightly larger, bold font for price labels
            painter.setFont(font)
            painter.setPen(QColor(200, 200, 220))

            # Ensure text fits and is readable
            max_price_text = f"${max_price:,.2f}"
            min_price_text = f"${min_price:,.2f}"

            # Draw price labels with a small offset from the border and background for readability
            text_padding = 3
            painter.drawText(QPointF(padding + text_padding, padding + painter.fontMetrics().height()), max_price_text)
            painter.drawText(QPointF(padding + text_padding, h - padding - text_padding), min_price_text)

        except Exception as e:
            ui_logger.error(f"Error in PriceChartWidget paintEvent: {e}", exc_info=True)
            painter.setPen(QColor(255, 60, 60)); painter.drawText(rect, Qt.AlignCenter, "Chart Error - See Logs")


class BinanceTradingApp(QMainWindow):
    """
    Main application window for the Binance Trading Bot.
    """
    def __init__(self) -> None:
        super().__init__()
        ui_logger.info("Initializing BinanceTradingApp UI...")
        self.setWindowTitle("🚀 Advanced ML Binance Trading Bot v2.2") # Version up
        self.setGeometry(50, 50, 1750, 1100)
        self.setMinimumSize(1280, 800) # Adjusted min size

        # Attempt to set window icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png") # Define path
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            ui_logger.info(f"Application icon set from {icon_path}")
        else:
            ui_logger.warning(f"Application icon 'app_icon.png' not found at {icon_path}. Using default.")

        self.binance_api: EnhancedBinanceAPI = EnhancedBinanceAPI()
        self.enhanced_data_worker: Optional[EnhancedDataWorker] = None
        self.is_trading: bool = False
        self.current_symbol: str = "BTCUSDT"
        self.current_model: str = "random_forest"
        self.signal_performance: List[Dict[str, Any]] = []

        try: init_db(); ui_logger.info("Database initialized.")
        except Exception as e:
            ui_logger.critical(f"DB Init Failed: {e}", exc_info=True)
            QMessageBox.critical(self, "DB Error", f"DB Init Failed: {e}\nTrade history unavailable.")

        self._setup_ui_elements()
        self.apply_styling()
        self.load_trade_history_from_db()
        self.update_connection_status("⚪ Ready - Configure API.", "neutral")
        ui_logger.info("BinanceTradingApp UI initialized.")
    
    def _setup_ui_elements(self) -> None:
        ui_logger.debug("Setting up UI elements and layouts.")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget(); self.tabs.setIconSize(QSize(22,22))
        self._setup_trading_tab(); self._setup_ml_tab(); self._setup_config_tab()
        self._setup_account_tab(); self._setup_analytics_tab(); self._setup_about_tab()
        self.main_layout.addWidget(self.tabs)
        ui_logger.debug("All UI tabs set up.")

    def _create_status_bar(self) -> QHBoxLayout:
        """Creates the shared status bar HBox."""
        status_layout = QHBoxLayout()
        self.connection_status = QLabel("Conn: --")
        self.ml_status = QLabel("ML: --")
        self.update_counter_label = QLabel("Upd: 0")
        for label in [self.connection_status, self.ml_status, self.update_counter_label]:
            label.setFont(QFont('Arial', 10, QFont.Bold)) # Slightly smaller for status bar
            label.setToolTip(label.text()) # Show full text on hover
        status_layout.addWidget(self.connection_status, 2) # Give more stretch
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.ml_status, 2) # More stretch
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.update_counter_label, 1)
        status_layout.addStretch(0)
        return status_layout

    def _setup_trading_tab(self) -> None:
        self.trading_tab = QWidget()
        layout = QVBoxLayout(self.trading_tab)
        layout.addLayout(self._create_status_bar()) # Add shared status bar
        
        # Controls section
        controls_group = QGroupBox("⚙️ Trading Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox(); self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT"])
        self.symbol_combo.setToolTip("Select trading pair"); self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed); controls_layout.addWidget(self.symbol_combo)
        
        controls_layout.addWidget(QLabel("ML Model:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["Random Forest", "XGBoost", "CouncilEnsemble", "Fallback Rules"])
        self.model_combo.setToolTip("Select ML model or rule-based fallback"); self.model_combo.currentTextChanged.connect(self.on_model_changed); controls_layout.addWidget(self.model_combo)
        
        self.auto_retrain_checkbox = QCheckBox("Auto-retrain Model"); self.auto_retrain_checkbox.setChecked(True)
        self.auto_retrain_checkbox.setToolTip("Automatically retrain selected ML model if not already trained or when symbol changes."); controls_layout.addWidget(self.auto_retrain_checkbox)
        
        self.start_button = QPushButton("🚀 Start"); self.start_button.setToolTip("Start the trading bot and data feed."); self.start_button.clicked.connect(self.start_ml_trading); controls_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("⏹️ Stop"); self.stop_button.setToolTip("Stop the trading bot."); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_ml_trading); controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        layout.addWidget(controls_group)

        # Price Display and Chart section
        price_group = QGroupBox("📈 Live Market Data"); price_layout = QVBoxLayout(price_group)
        self.price_display = QLabel("💰 Price: Waiting..."); self.price_display.setFont(QFont('Arial', 20, QFont.Bold)); self.price_display.setAlignment(Qt.AlignCenter); self.price_display.setMinimumHeight(70)
        self.price_display.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #212121, stop:1 #2c2c2c); color: #00E070; border: 2px solid #383838; border-radius: 8px; padding: 10px;")
        
        stats_layout = QHBoxLayout()
        self.stats_24h_change = QLabel("24h Δ: --"); self.stats_24h_volume = QLabel("Vol: --")
        self.stats_high_low = QLabel("H/L: --"); self.stats_trades = QLabel("Trades: --")
        for stat_label in [self.stats_24h_change, self.stats_24h_volume, self.stats_high_low, self.stats_trades]:
            stat_label.setFont(QFont('Arial', 9)); stats_layout.addWidget(stat_label)
        price_layout.addWidget(self.price_display); price_layout.addLayout(stats_layout)
        self.price_chart = PriceChartWidget(); price_layout.addWidget(self.price_chart)
        layout.addWidget(price_group)

        # Manual and Auto Trading in a QHBoxLayout
        trade_actions_layout = QHBoxLayout()
        manual_group = QGroupBox("🖐️ Manual Trade"); manual_form_layout = QFormLayout(manual_group)
        self.manual_qty = QLineEdit(); self.manual_qty.setPlaceholderText("e.g., 0.01"); self.manual_qty.setValidator(QDoubleValidator(0, 1000000, 8)); manual_form_layout.addRow("Quantity:", self.manual_qty)
        self.manual_price = QLineEdit(); self.manual_price.setPlaceholderText("Required for LIMIT"); self.manual_price.setValidator(QDoubleValidator(0, 10000000, 8)); manual_form_layout.addRow("Price:", self.manual_price)
        self.manual_order_type = QComboBox(); self.manual_order_type.addItems(["MARKET", "LIMIT"]); manual_form_layout.addRow("Order Type:", self.manual_order_type)
        manual_buttons_hbox = QHBoxLayout(); self.manual_buy_button = QPushButton("BUY"); self.manual_sell_button = QPushButton("SELL")
        self.manual_buy_button.clicked.connect(lambda: self.execute_manual_order("BUY")); self.manual_sell_button.clicked.connect(lambda: self.execute_manual_order("SELL"))
        manual_buttons_hbox.addWidget(self.manual_buy_button); manual_buttons_hbox.addWidget(self.manual_sell_button); manual_form_layout.addRow(manual_buttons_hbox)
        trade_actions_layout.addWidget(manual_group)

        auto_group = QGroupBox("🤖 Auto-Trade Settings"); auto_form_layout = QFormLayout(auto_group)
        self.auto_trade_checkbox = QCheckBox("Enable Auto-Trading"); self.auto_trade_checkbox.setToolTip("Automatically execute trades based on ML signals meeting confidence threshold.") ; auto_form_layout.addRow(self.auto_trade_checkbox)
        self.auto_trade_qty = QLineEdit(); self.auto_trade_qty.setPlaceholderText("e.g., 0.005"); self.auto_trade_qty.setValidator(QDoubleValidator(0,1000000,8)); auto_form_layout.addRow("Quantity/Trade:", self.auto_trade_qty)
        self.confidence_spin = QSpinBox(); self.confidence_spin.setRange(50,100); self.confidence_spin.setValue(80); self.confidence_spin.setSuffix("%"); self.confidence_spin.setToolTip("Minimum signal confidence to trigger an auto-trade.") ; auto_form_layout.addRow("Min. Confidence:", self.confidence_spin)
        trade_actions_layout.addWidget(auto_group)
        layout.addLayout(trade_actions_layout)

        self._setup_data_tables(layout)
        self.tabs.addTab(self.trading_tab, "📈 Live Trading")


    # ... (implement other _setup_<tab>_tab methods similarly, ensuring widgets are self attributes for access)
    def _setup_ml_tab(self) -> None:
        self.ml_tab = QWidget(); layout = QVBoxLayout(self.ml_tab)
        status_group = QGroupBox("🤖 ML System Status & Controls"); status_form = QFormLayout(status_group)
        self.ml_detailed_status = QLabel("Status: Idle."); self.ml_detailed_status.setWordWrap(True)
        self.ml_accuracy_label = QLabel("Model Accuracy: N/A")
        self.ml_model_info = QLabel("Active Model: None")
        self.training_progress = QProgressBar(); self.training_progress.setVisible(False); self.training_progress.setTextVisible(True); self.training_progress.setFormat("%p% - %v/%m")
        status_form.addRow("Current Status:", self.ml_detailed_status)
        status_form.addRow("Training Progress:", self.training_progress)
        status_form.addRow("Est. Accuracy:", self.ml_accuracy_label)
        status_form.addRow("Active Model Type:", self.ml_model_info)
        layout.addWidget(status_group)

        model_mgmt_group = QGroupBox("📁 Model Management"); model_mgmt_form = QFormLayout(model_mgmt_group)
        self.model_file_input = QLineEdit(); self.model_file_input.setPlaceholderText("Path to .joblib model file")
        browse_btn = QPushButton("📂 Browse"); browse_btn.clicked.connect(self.browse_model_file)
        file_hbox = QHBoxLayout(); file_hbox.addWidget(self.model_file_input); file_hbox.addWidget(browse_btn)
        model_mgmt_form.addRow("Model File:", file_hbox)
        actions_hbox = QHBoxLayout()
        self.save_model_button = QPushButton("💾 Save Model"); self.save_model_button.clicked.connect(self.save_ml_model); actions_hbox.addWidget(self.save_model_button)
        self.load_model_button = QPushButton("📤 Load Model"); self.load_model_button.clicked.connect(self.load_ml_model); actions_hbox.addWidget(self.load_model_button)
        self.retrain_button = QPushButton("🔄 Retrain Model"); self.retrain_button.clicked.connect(self.retrain_ml_model); actions_hbox.addWidget(self.retrain_button)
        model_mgmt_form.addRow(actions_hbox)
        layout.addWidget(model_mgmt_group)

        bottom_layout = QHBoxLayout()
        feat_imp_group = QGroupBox("🎯 Feature Importance"); feat_imp_vbox = QVBoxLayout(feat_imp_group)
        self.feature_importance_table = QTableWidget(0,2); self.feature_importance_table.setHorizontalHeaderLabels(["Feature", "Importance"]); self.feature_importance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); feat_imp_vbox.addWidget(self.feature_importance_table)
        bottom_layout.addWidget(feat_imp_group, 1)

        ml_signals_group = QGroupBox("📜 ML Signals Log"); ml_signals_vbox = QVBoxLayout(ml_signals_group)
        self.ml_signals_table = QTableWidget(0,7); self.ml_signals_table.setHorizontalHeaderLabels(["Time", "Symbol", "Signal", "Confidence", "Price", "Model", "Trained?"]); self.ml_signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.ml_signals_table.horizontalHeader().setStretchLastSection(True); ml_signals_vbox.addWidget(self.ml_signals_table)
        self.export_signals_button = QPushButton("📊 Export Signal History"); self.export_signals_button.clicked.connect(self.export_signal_history); ml_signals_vbox.addWidget(self.export_signals_button)
        bottom_layout.addWidget(ml_signals_group, 2)
        layout.addLayout(bottom_layout)
        layout.addStretch()
        self.tabs.addTab(self.ml_tab, "🤖 Machine Learning")

    def _setup_config_tab(self) -> None:
        self.config_tab = QWidget(); layout = QVBoxLayout(self.config_tab)
        api_group = QGroupBox("🔐 Binance API Credentials"); api_form = QFormLayout(api_group)
        self.api_key_input = QLineEdit(); self.api_key_input.setPlaceholderText("Enter your Binance API Key"); self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_secret_input = QLineEdit(); self.api_secret_input.setPlaceholderText("Enter your Binance API Secret"); self.api_secret_input.setEchoMode(QLineEdit.Password)
        self.testnet_checkbox = QCheckBox("🔗 Use Binance Testnet"); self.testnet_checkbox.setChecked(True); self.testnet_checkbox.stateChanged.connect(self._update_config_info_display) # Update info on change

        self.config_info = QTextEdit(); self.config_info.setReadOnly(True); self.config_info.setFixedHeight(120)
        self._update_config_info_display() # Initial population

        api_form.addRow("API Key:", self.api_key_input)
        api_form.addRow("API Secret:", self.api_secret_input)
        api_form.addRow(self.testnet_checkbox)
        api_form.addRow("Current Setup:", self.config_info)

        buttons_hbox = QHBoxLayout()
        self.save_config_button = QPushButton("💾 Save Keys"); self.save_config_button.clicked.connect(self.save_api_config); buttons_hbox.addWidget(self.save_config_button)
        self.test_connection_button = QPushButton("📡 Test Connection"); self.test_connection_button.clicked.connect(self.test_api_connection); buttons_hbox.addWidget(self.test_connection_button)
        self.clear_config_button = QPushButton("🗑️ Clear Keys"); self.clear_config_button.clicked.connect(self.clear_api_config); buttons_hbox.addWidget(self.clear_config_button)
        api_form.addRow(buttons_hbox)
        layout.addWidget(api_group); layout.addStretch()
        self.tabs.addTab(self.config_tab, "⚙️ API Config")

    def _setup_account_tab(self) -> None:
        self.account_tab = QWidget(); layout = QVBoxLayout(self.account_tab)
        acc_group = QGroupBox("💰 Account Information & Balances"); acc_vbox = QVBoxLayout(acc_group)
        self.account_info_display = QTextEdit(); self.account_info_display.setReadOnly(True); self.account_info_display.setFont(QFont("Courier New", 9))
        self.account_info_display.setPlainText("Configure API and click 'Refresh' to view account details.")

        refresh_hbox = QHBoxLayout()
        self.refresh_account_button = QPushButton("🔄 Refresh Account Info"); self.refresh_account_button.clicked.connect(self.refresh_account_info)
        refresh_hbox.addWidget(self.refresh_account_button); refresh_hbox.addStretch()
        acc_vbox.addWidget(self.account_info_display); acc_vbox.addLayout(refresh_hbox)
        layout.addWidget(acc_group); layout.addStretch()
        self.tabs.addTab(self.account_tab, "💰 Account")

    def _setup_analytics_tab(self) -> None:
        self.analytics_tab = QWidget(); layout = QVBoxLayout(self.analytics_tab)
        metrics_group = QGroupBox("📈 Signal Performance Overview"); metrics_vbox = QVBoxLayout(metrics_group)
        summary_hbox = QHBoxLayout()
        self.total_signals_label = QLabel("Total Signals: 0"); self.buy_signals_label = QLabel("Buy Signals: 0")
        self.sell_signals_label = QLabel("Sell Signals: 0"); self.hold_signals_label = QLabel("Hold Signals: 0")
        for label in [self.total_signals_label, self.buy_signals_label, self.sell_signals_label, self.hold_signals_label]:
            label.setFont(QFont('Arial', 10, QFont.Bold)); summary_hbox.addWidget(label)
        summary_hbox.addStretch()
        metrics_vbox.addLayout(summary_hbox)
        metrics_vbox.addWidget(QLabel("Performance Breakdown (Recent Signals):"))
        self.performance_table = QTableWidget(0,5); self.performance_table.setHorizontalHeaderLabels(["Time Window", "Signals", "Avg. Confidence", "Strongest Signal", "Dominant Model"])
        self.performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_vbox.addWidget(self.performance_table)
        layout.addWidget(metrics_group); layout.addStretch()
        self.tabs.addTab(self.analytics_tab, "📊 Analytics")

    def _setup_about_tab(self) -> None:
        self.about_tab = QWidget(); layout = QVBoxLayout(self.about_tab)
        about_text = QLabel(
            "<b>Advanced ML Binance Trading Bot v2.2</b><br><br>"
            "Developed by Jules AI, focusing on robust software engineering for trading applications.<br><br>"
            "This tool is for educational and experimental use. Trading involves substantial risk. "
            "No financial advice is provided. Always use testnet first and manage risk appropriately."
        ); about_text.setWordWrap(True); about_text.setAlignment(Qt.AlignTop)
        layout.addWidget(about_text)
        self.copy_code_button = QPushButton("📋 Copy UI Source"); self.copy_code_button.clicked.connect(self.copy_source_code)
        layout.addWidget(self.copy_code_button, 0, Qt.AlignLeft)
        layout.addStretch()
        self.tabs.addTab(self.about_tab, "ℹ️ About")

    # ... (rest of the methods like apply_styling, event handlers, data updaters, etc. will be included)
    # For brevity, only showing the structure and a few key method signatures that would be updated.
    # The full file will be provided in the overwrite.

    def apply_styling(self) -> None:
        """Applies a modern dark stylesheet to the application."""
        # Stylesheet content is extensive, kept similar to original for brevity in this example.
        # Ensure it's loaded or defined here.
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2b2b2b, stop:1 #1e1e1e); color: #ffffff; }
            QTabWidget::pane { border: 1px solid #444; background: #2E2E2E; border-radius: 5px; }
            QTabBar::tab { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #3a3a3a); color: #e0e0e0; padding: 10px 18px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }
            QTabBar::tab:selected { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #007bff, stop:1 #0056b3); border-bottom: 3px solid #00ff88; color: white; } /* Adjusted selected tab */
            QTabBar::tab:hover { background: #555; }
            QLabel { color: #e0e0e0; padding: 2px; }
            QPushButton { background-color: #007bff; color: white; border: 1px solid #0069d9; padding: 8px 15px; font-size: 12px; font-weight: bold; border-radius: 4px; min-width: 90px; }
            QPushButton:hover { background-color: #0069d9; border-color: #005cbf;}
            QPushButton:pressed { background-color: #005cbf; }
            QPushButton:disabled { background-color: #555; color: #aaa; border-color: #444;}
            QComboBox, QLineEdit, QSpinBox { padding: 7px 10px; font-size: 12px; border: 1px solid #555; border-radius: 4px; background-color: #3E3E3E; color: #e0e0e0; }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border: 1px solid #00ff88; }
            QComboBox::drop-down { border: none; background-color: #4E4E4E; width: 18px; border-radius: 3px; }
            QComboBox::down-arrow { image: url(noimg.png); /* Use Qt standard arrow if no image */ border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #e0e0e0; margin: 1px;}
            QTableWidget { gridline-color: #484848; background-color: #2E2E2E; alternate-background-color: #383838; color: #e0e0e0; border: 1px solid #444; border-radius: 5px; selection-background-color: #007bff; selection-color: white;}
            QHeaderView::section { background-color: #3E3E3E; padding: 8px; border: 1px solid #484848; font-weight: bold; color: #00ff88; border-radius: 0px; } /* Removed radius for cleaner look */
            QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 6px; margin-top: 12px; padding-top: 12px; background-color: #333333;}
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: -5px 8px 0px 8px; color: #00ff88; font-size: 13px; font-weight: bold; background-color: #333333; border-radius: 4px;}
            QTextEdit { background-color: #2E2E2E; color: #e0e0e0; border: 1px solid #444; border-radius: 4px; padding: 5px; selection-background-color: #007bff; selection-color: white;}
            QCheckBox { color: #e0e0e0; font-weight: normal; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; border-radius: 3px; background-color: #3E3E3E; }
            QCheckBox::indicator:checked { background-color: #00ff88; border: 1px solid #00cc66; }
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; color: #e0e0e0; font-weight: bold; background-color: #3E3E3E; min-height: 22px; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ff88, stop:1 #00cc66); border-radius: 3px; margin: 1px; }
            QScrollBar:vertical, QScrollBar:horizontal { background-color: #2E2E2E; width: 12px; margin: 0px; border-radius: 6px; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background-color: #555; border-radius: 6px; min-height: 20px; min-width:20px; }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background-color: #666; }
            QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }
            QScrollBar::add-page, QScrollBar::sub-page { background: none; }
        """)
        ui_logger.debug("UI styling applied.")

    def on_symbol_changed(self, symbol: str) -> None:
        old_symbol = self.current_symbol
        self.current_symbol = symbol
        ui_logger.info(f"Symbol changed: {old_symbol} → {symbol}")
        self.update_connection_status(f"🔄 Symbol set to {symbol}. Restart if active.", "pending")
        if self.enhanced_data_worker and self.enhanced_data_worker.isRunning():
            reply = QMessageBox.question(self, "Confirm Symbol Change", f"Change symbol to {symbol} and restart data stream?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_ml_trading(); QTimer.singleShot(100, self.start_ml_trading)
            else:
                self.symbol_combo.blockSignals(True); self.symbol_combo.setCurrentText(old_symbol); self.symbol_combo.blockSignals(False)
                self.current_symbol = old_symbol; self.update_connection_status(f"🔄 Symbol change to {symbol} cancelled.", "neutral")
        elif self.is_trading: self.start_ml_trading() # If was "trading" but worker died

    def on_model_changed(self, model_text: str) -> None:
        model_map = {"Random Forest":"random_forest", "XGBoost":"xgboost", "CouncilEnsemble":"council", "Fallback Rules":"fallback"}
        new_model = model_map.get(model_text, "random_forest")
        if new_model == self.current_model: return
        ui_logger.info(f"Model changed: {self.current_model} → {new_model}")
        self.current_model = new_model; self.ml_model_info.setText(f"Selected: {model_text}")
        if self.enhanced_data_worker and self.enhanced_data_worker.isRunning():
            if new_model == "fallback":
                if self.enhanced_data_worker.ml_generator: self.enhanced_data_worker.ml_generator.is_trained = False
                self.update_ml_status("🔧 Fallback Rules active.", "neutral")
                self.ml_accuracy_label.setText("Accuracy: N/A (Rules)"); self.update_feature_importance_display([])
            else:
                self.enhanced_data_worker.set_model_type(new_model) # This will trigger re-init/train in worker
                self.update_ml_status(f"🔄 Initializing {model_text}...", "pending")
    
    # ... (other methods will be fully fleshed out in the overwrite call) ...

    def closeEvent(self, event: Any) -> None: # QCloseEvent
        ui_logger.info("Close event triggered.")
        if self.is_trading:
            reply = QMessageBox.question(self, "Exit Confirmation", "Trading system active. Exit anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes: event.ignore(); ui_logger.info("Exit cancelled."); return
        ui_logger.info("Proceeding with shutdown."); self.stop_ml_trading() # Ensure worker stops
        if hasattr(self,'binance_api'): self.binance_api.api_key = ""; self.binance_api.api_secret = ""
        ui_logger.info("Cleanup complete. Exiting."); event.accept()

def main() -> int:
    try:
        root_logger = logging.getLogger()
        if not root_logger.hasHandlers():
            log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s") # Added lineno
            log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_app.log")
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(log_formatter)
            root_logger.addHandler(stream_handler)
            root_logger.setLevel(logging.INFO)
            ui_logger.info("Root logger configured by trading_ui.main.")
        else: ui_logger.info("Root logger already configured.")

        ui_logger.info(f"Application starting... Python: {sys.version.split()[0]}")
        print("="*70 + f"\n🤖 ADVANCED ML BINANCE TRADING BOT - v2.2\n" + "="*70)
        
        app = QApplication(sys.argv)
        app.setApplicationName("AdvMLBinanceBot"); app.setApplicationVersion("2.2.0")
        app.setOrganizationName("JulesAI"); app.setStyle('Fusion')
        if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Dependency Check
        core_deps = ['pandas', 'numpy', 'sklearn', 'ta', 'requests', 'joblib']
        missing = [dep for dep in core_deps if not __import__(dep, fromlist=[''])._is_loaded] if hasattr(__import__('pandas'), '_is_loaded') else [dep for dep in core_deps if not __import__(dep)] # Basic check
        if missing:
            err = f"Missing dependencies: {', '.join(missing)}. Install via pip."
            ui_logger.critical(err); QMessageBox.critical(None, "Dependency Error", err); return 1
        ui_logger.info("Core dependencies OK.")

        window = BinanceTradingApp()
        window.show()
        ui_logger.info("Main window shown. Entering Qt event loop.")
        exit_code = app.exec_()
        ui_logger.info(f"Qt event loop finished. Exit code: {exit_code}")
        return exit_code
        
    except Exception as e:
        fatal_err = f"FATAL STARTUP ERROR in main(): {e}"
        print(fatal_err, file=sys.stderr)
        if logging.getLogger().hasHandlers(): logging.critical(fatal_err, exc_info=True)
        else: # Fallback log if full logging failed
            with open("trading_app_CRITICAL.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} - {fatal_err}\n{traceback.format_exc()}\n")
        try:
            temp_app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Fatal Startup Error", f"Critical error on startup: {e}\nCheck logs.")
            if temp_app is not QApplication.instance(): temp_app.exec_()
        except Exception as qm_e: print(f"Could not display fatal error dialog: {qm_e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    def global_exception_hook(exctype: Any, value: Any, traceback_obj: Any) -> None:
        logger_instance = logging.getLogger(__name__) if logging.getLogger(__name__).hasHandlers() else logging.getLogger()
        logger_instance.critical("Unhandled global exception:", exc_info=(exctype, value, traceback_obj))
        # It's often good practice to also show a user-facing message for unhandled exceptions
        # This ensures the user isn't left with a frozen or crashed app without explanation.
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Unhandled Application Error")
        msg_box.setText("An unexpected error occurred. The application might need to close.\n"
                        "Please check the 'trading_app.log' file for details.")
        msg_box.setDetailedText(f"{str(value)}\n\nTraceback:\n{ ''.join(traceback.format_tb(traceback_obj)) }")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_() # Show the message box
        sys.__excepthook__(exctype, value, traceback_obj) # Then call original hook

    sys.excepthook = global_exception_hook
    import traceback # Make sure traceback is imported for the hook

    final_code = 1
    try: final_code = main()
    except SystemExit as se: final_code = se.code if se.code is not None else 0
    except KeyboardInterrupt: ui_logger.info("App interrupted (Ctrl+C)."); final_code = 0
    except Exception as e:
        crit_msg = f"CRITICAL FAILURE IN __main__: {e}"
        print(crit_msg, file=sys.stderr)
        logger_inst = logging.getLogger(__name__) if logging.getLogger(__name__).hasHandlers() else logging.getLogger()
        logger_inst.critical(crit_msg, exc_info=True)
    finally:
        ui_logger.info(f"Application exiting. Final code: {final_code}")
        logging.shutdown()
        sys.exit(final_code)
