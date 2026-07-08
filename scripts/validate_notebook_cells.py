import ast
import json
from pathlib import Path

root = Path('notebooks/045_Multimodal_Test_Corpus_Evaluation.ipynb')
nb = json.loads(root.read_text(encoding='utf-8'))
print('cells', len(nb['cells']))
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    ast.parse(src)
    print('compiled', cell.get('id'))
print('all_python_cells_compile')
