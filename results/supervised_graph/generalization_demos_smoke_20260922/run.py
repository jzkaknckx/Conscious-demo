import json
from pathlib import Path
nb=json.loads(Path('src/test_supervised_graph_learning.ipynb').read_text());ns={'__name__':'__main__'}
for i in (1,3,5,7,11,13):
 print('CELL',i,flush=True)
 exec(compile(''.join(nb['cells'][i]['source']),f'notebook_cell_{i}','exec'),ns)
 if i==1: ns.update(OUTPUT=Path('results/supervised_graph/generalization_demos_smoke_20260922'),display=lambda *args:None)
ns['demo_learner'].save(ns['OUTPUT']/'memory.pt')
ns['save_json'](ns['OUTPUT']/'demo_results.json',ns['demo_results'])
print('DEMO_SMOKE_PASSED',flush=True)
