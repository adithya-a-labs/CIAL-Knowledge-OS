import importlib
import sys


def test_tesseract_ocr_module_imports_without_optional_llm_dependencies(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "src")
    sys.modules.pop("cial_knowledge_os", None)
    sys.modules.pop("cial_knowledge_os.ocr", None)
    sys.modules.pop("cial_knowledge_os.ocr.tesseract_ocr", None)

    module = importlib.import_module("cial_knowledge_os.ocr.tesseract_ocr")
    engine = module.TesseractOCREngine()

    result = engine.preflight(enabled=True)
    assert result["status"] in {"ready", "python_package_missing", "binary_missing", "disabled"}
