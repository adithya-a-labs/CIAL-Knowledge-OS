import json
from pathlib import Path

path = Path('notebooks/045_Multimodal_Test_Corpus_Evaluation.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code' and cell.get('id') == '#VSC-68883127':
        cell['source'] = [
            '# Print the support matrix for the expected multi-format corpus.',
            '',
            'support_rows = []',
            'for definition in FORMAT_REGISTRY:',
            "    ext_values = ', '.join(f'.{ext}' for ext in definition.extensions)",
            "    loader_name = 'ocr' if definition.requires_ocr else 'registry_loader'",
            '    support_rows.append(',
            '        {',
            '            "Format": definition.format_label,',
            '            "Extension": ext_values,',
            '            "Loader": loader_name,',
            '            "OCR Required": bool(definition.requires_ocr),',
            '            "Status": definition.support_status.value,',
            '        }',
            '    )',
            '',
            '# Append a clear unsupported row for any additional file types.',
            'support_rows.append(',
            '    {',
            '        "Format": "Unsupported others",',
            '        "Extension": ".bin, .exe, .zip, .rar, .7z, .mp3, .mp4, .dwg, .sldprt",',
            '        "Loader": "none",',
            '        "OCR Required": False,',
            '        "Status": "skipped_with_warning",',
            '    }',
            ')',
            'support_frame = pd.DataFrame(support_rows)',
            'display(support_frame[["Format", "Extension", "Loader", "OCR Required", "Status"]])',
            '',
            '# Recursively scan the configured corpus and summarize by extension.',
            '',
            'if not TEST_CORPUS_ROOT.exists():',
            '    display(Markdown(f"### Corpus not found\\nExpected test corpus at {TEST_CORPUS_ROOT}. Create the folder and place files there before running the notebook."))',
            'else:',
            '    readiness = scan_file_format_readiness(TEST_CORPUS_ROOT)',
            '    extension_rows = []',
            '    for item in readiness.get("extensions") or []:',
            '        extension_rows.append(',
            '            {',
            '                "extension": item.get("extension", ""),',
            '                "count": item.get("count", 0),',
            '                "category": item.get("category", ""),',
            '                "format_label": item.get("format_label", ""),',
            '                "support_status": item.get("support_status", ""),',
            '                "ingestion_enabled": item.get("ingestion_enabled", False),',
            '                "requires_ocr": item.get("requires_ocr", False),',
            '            }',
            '        )',
            '    extension_frame = pd.DataFrame(extension_rows).sort_values(["count", "extension"], ascending=[False, True])',
            '    display(extension_frame)',
            '    print("\\nSummary")',
            '    print(json.dumps({',
            '        "total_files": readiness.get("total_files", 0),',
            '        "processable_files": readiness.get("processable_files", 0),',
            '        "ocr_files": readiness.get("ocr_files", 0),',
            '        "recognized_future_files": readiness.get("recognized_future_files", 0),',
            '        "unsupported_files": readiness.get("unsupported_files", 0),',
            '    }, indent=2))',
        ]
        break
else:
    raise SystemExit('target cell not found')

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('updated', path)
