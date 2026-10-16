import argparse,csv,json,math,re,statistics
from pathlib import Path
import matplotlib.pyplot as plt
SIZES=[16,32,64,128,256,512,1024]
UNITS={'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1.0}
def main():
 p=argparse.ArgumentParser();p.add_argument('json_file',type=Path);a=p.parse_args()
 data=json.loads(a.json_file.read_text(encoding='utf-8-sig'))
 groups={(method,n):[] for method in ('Shmem','Naive') for n in SIZES}
 for item in data.get('benchmarks',[]):
  if item.get('error_occurred'):raise RuntimeError(item.get('error_message','Benchmark failed'))
  if item.get('run_type','iteration')!='iteration':continue
  m=re.match(r'^BM_Cuda(Naive)?/(\d+)/manual_time$',item['name'])
  if not m:continue
  key=('Naive' if m.group(1) else 'Shmem',int(m.group(2)))
  if key in groups:groups[key].append(float(item['real_time'])*UNITS[item['time_unit']])
 if any(len(v)<1 for v in groups.values()):raise RuntimeError('Missing benchmark data: run matrix_benchmark for all sizes first.')
 rows=[]
 for n in SIZES:
  sh,na=statistics.median(groups['Shmem',n]),statistics.median(groups['Naive',n])
  rows.append({'n':n,'shared_memory_seconds':sh,'naive_seconds':na,'speedup_naive_over_shmem':na/sh,
   'shared_repetitions':len(groups['Shmem',n]),'naive_repetitions':len(groups['Naive',n])})
 out=a.json_file.parent
 with (out/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 plt.figure(figsize=(9,6))
 plt.plot(SIZES,[r['naive_seconds']*1e3 for r in rows],marker='o',label='Наивный kernel')
 plt.plot(SIZES,[r['shared_memory_seconds']*1e3 for r in rows],marker='o',label='Shared memory')
 plt.xscale('log',base=2);plt.yscale('log');plt.xticks(SIZES,SIZES);plt.xlabel('Размер n матрицы n×n');plt.ylabel('Время kernel, мс');plt.title('Экспериментальная вычислительная сложность');plt.grid(True,which='both',alpha=.3);plt.legend();plt.tight_layout();plt.savefig(out/'complexity.png',dpi=180);plt.close()
 plt.figure(figsize=(9,6));plt.plot(SIZES,[r['speedup_naive_over_shmem'] for r in rows],marker='o');plt.axhline(1,color='gray',linestyle='--')
 plt.xscale('log',base=2);plt.xticks(SIZES,SIZES);plt.xlabel('Размер n матрицы n×n');plt.ylabel('Ускорение naive / shared-memory');plt.title('Ускорение новой реализации');plt.grid(True,which='both',alpha=.3);plt.tight_layout();plt.savefig(out/'speedup.png',dpi=180);plt.close()
 (out/'summary.json').write_text(json.dumps({'context':data.get('context',{}),'rows':rows,'theoretical_complexity':'Theta(n^3)'},indent=2,ensure_ascii=False),encoding='utf-8')
 print('Generated measurements.csv, summary.json, complexity.png and speedup.png from measured benchmark data.')
if __name__=='__main__':main()
