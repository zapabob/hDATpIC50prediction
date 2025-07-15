"""
Main CLI for hDAT pIC50 prediction system.
"""

import sys
from pathlib import Path
from dat_predictor import DATPredictor, DATPredictorGUI


def main():
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        predictor = DATPredictor()
        model_path = Path(predictor.config.MODEL_DIR) / 'dat_transformer_model.pt'
        if model_path.exists():
            predictor.load_model(str(model_path))
        gui = DATPredictorGUI(predictor)
        gui.show()
        sys.exit(app.exec())
    except Exception as e:
        import logging
        logging.error(f"アプリケーション実行エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 