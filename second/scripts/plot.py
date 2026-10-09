"""Only measured Google Benchmark iterations are accepted. No sample/fallback data."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

SIZES = [16, 32, 64, 128, 256, 512, 1024]
UNITS = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}

def load_measurements(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not data.get('context', {}).get('gpu'):
        raise ValueError('Missing GPU metadata from matrix_benchmark')
    groups = {(method, n): [] for method in ('Cuda', 'Eigen') for n in SIZES}
    for item in data['benchmarks']:
        if item.get('error_occurred'):
            raise ValueError(item.get('error_message', 'Benchmark failed'))
        if item.get('run_type', 'iteration') != 'iteration':
            continue
        match = re.match(r'^BM_(Cuda|Eigen)Matmul/(\d+)/manual_time$', item['name'])
        if not match:
            continue
        key = (match[1], int(match[2]))
        if key not in groups:
            continue
        seconds = float(item['real_time']) * UNITS[item['time_unit']]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f'Invalid measured time: {item}')
        groups[key].append(seconds)
    if any(len(v) < 3 for v in groups.values()):
        raise ValueError('Need at least 3 measured repetitions for all 7 sizes and both methods')
    rows = []
    for n in SIZES:
        c, e = groups['Cuda', n], groups['Eigen', n]
        cm, em = statistics.median(c), statistics.median(e)
        rows.append(dict(n=n, cuda_seconds=cm, eigen_seconds=em, speedup=em/cm,
                         cuda_min=min(c), cuda_max=max(c), eigen_min=min(e), eigen_max=max(e),
                         cuda_repetitions=len(c), eigen_repetitions=len(e)))
    return data['context'], rows

def slopes(rows, field):
    return [math.log(b[field]/a[field])/math.log(b['n']/a['n']) for a,b in zip(rows,rows[1:])]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('json_file', type=Path)
    args=parser.parse_args(); context,rows=load_measurements(args.json_file)
    import plotly.graph_objects as go
    output=args.json_file.parent
    with (output/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={'context':context,'rows':rows,'cuda_slopes':slopes(rows,'cuda_seconds'),
             'eigen_slopes':slopes(rows,'eigen_seconds'),
             'source_sha256':hashlib.sha256(args.json_file.read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fig=go.Figure()
    for name,key,color in [('CUDA (kernel in operator*)','cuda','#166534'),('Eigen::MatrixXf (1 CPU thread)','eigen','#1d4ed8')]:
        fig.add_trace(go.Scatter(x=SIZES,y=[r[key+'_seconds']*1000 for r in rows],name=name,mode='lines+markers',line_color=color,
            error_y=dict(type='data',symmetric=False,array=[(r[key+'_max']-r[key+'_seconds'])*1000 for r in rows],
                         arrayminus=[(r[key+'_seconds']-r[key+'_min'])*1000 for r in rows])))
    fig.update_layout(title='Matrix multiplication: measured time (median, min/max)',template='plotly_white',
                      xaxis_title='Matrix size n',yaxis_title='Time, ms',xaxis_type='log',yaxis_type='log',width=1100,height=650)
    speed=go.Figure(go.Scatter(x=SIZES,y=[r['speedup'] for r in rows],mode='lines+markers',name='Eigen / CUDA'))
    speed.add_hline(y=1,line_dash='dash')
    speed.update_layout(title='Compute-only speedup: Eigen / CUDA',template='plotly_white',xaxis_title='Matrix size n',
                        yaxis_title='Speedup, times',xaxis_type='log',width=1100,height=650)
    for name,chart in [('complexity',fig),('speedup',speed)]:
        chart.write_html(output/(name+'.html'),include_plotlyjs=True)
        try: chart.write_image(output/(name+'.png'),scale=2)
        except Exception as error:
            raise RuntimeError('PNG export needs Chrome. Install Google Chrome, then rerun plot.py. HTML and raw results are preserved.') from error
    print(f'Plots generated exclusively from {args.json_file}')

if __name__=='__main__': main()
