import json
from pathlib import Path
nb=json.loads(Path('src/test_supervised_graph_learning.ipynb').read_text())
namespace={'__name__':'__main__'}
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    print('CELL',i,flush=True)
    exec(compile(''.join(cell['source']),f'notebook_cell_{i}','exec'),namespace)
    if i==1:
        namespace.update(RUN_SINGLE_DEMOS=False,RUN_PARTITION_COMPARISON=False,
            RUN_FAMILIARITY_PILOT=False,RUN_DETECTION=False,RUN_TRAINING_RECALL=False,
            BATCH_IMAGE_LIMIT=1,QUERY_IMAGE_LIMIT=1,
            OUTPUT=Path('results/supervised_graph/generalization_notebook_smoke_20260922'),
            display=lambda *args:None)
print('NOTEBOOK_SMOKE_PASSED',flush=True)
