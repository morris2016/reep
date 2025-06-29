"""
Trading Application Entry Point.

This script serves as the main entry point for launching the Advanced ML Binance Trading Bot.
It imports the `main` function from the `trading_ui` module and executes it,
handling the application startup and exit.
"""

from trading_ui import main
import sys # For sys.exit

if __name__ == "__main__":
    """
    Main execution block.
    Calls the main function from trading_ui and exits with its return code.
    """
    # It's good practice for main() to return an exit code.
    # SystemExit is raised to ensure a clean exit with that code.
    exit_code: int = main()
    raise SystemExit(exit_code)
