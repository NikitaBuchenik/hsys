"""Build figures exclusively from a validated GPU run."""
import hashlib
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_results import validate, SIZES, METHODS

p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
s=validate(a.results);rows=s['rows']
with (a.results/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

labels={'WMMA':'WMMA: FP16 -> FP32','SharedFP32':'Previous: shared FP32','SharedFP16':'Control: shared FP16 -> FP32'}
fig,ax=plt.subplots(figsize=(9,6))
for method in METHODS:
    y=[r[method+'_seconds']*1000 for r in rows]
    ax.plot(SIZES,y,'o-',label=labels[method])
    ax.fill_between(SIZES,[r[method+'_min']*1000 for r in rows],
                    [r[method+'_max']*1000 for r in rows],alpha=.12)
ax.set(xscale='log',yscale='log',xlabel='Matrix size n',ylabel='Kernel time (ms)',
       title='Median kernel time; shading = min/max of 5 runs')
ax.set_xticks(SIZES,labels=[str(n) for n in SIZES]);ax.grid(True,alpha=.3);ax.legend()
fig.tight_layout();fig.savefig(a.results/'complexity.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6))
ax.plot(SIZES,[r['speedup_previous'] for r in rows],'o-',label='Previous FP32 / WMMA')
ax.plot(SIZES,[r['speedup_same_input'] for r in rows],'s--',label='Shared FP16 / WMMA (control)')
ax.axhline(1,color='gray',linestyle=':');ax.set_xscale('log',base=2)
ax.set_xticks(SIZES,labels=[str(n) for n in SIZES]);ax.set(xlabel='Matrix size n',ylabel='Speedup',title='Compute-only speedup')
ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(a.results/'speedup.png',dpi=180);plt.close(fig)
print('Saved measurements.csv, summary.json, complexity.png, speedup.png')

s['chart_sha256']={name:hashlib.sha256((a.results/name).read_bytes()).hexdigest() for name in ['complexity.png','speedup.png']}
(a.results/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
