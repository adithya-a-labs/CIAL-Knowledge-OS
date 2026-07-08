from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from cial_knowledge_os.ocr.tesseract_ocr import TesseractOCREngine

engine = TesseractOCREngine(language='eng')
preflight = engine.preflight(enabled=True)
print(json.dumps(preflight, indent=2))
