"""Validate actual run files before charts/report. Python 3.9 compatible."""
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

SIZES = [16,32,64,128,256,512,1024]
METHODS = ['WMMA','SharedFP32','SharedFP16']
UNITS = {'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1.0}

def validate(results):
    results = Path(results)
    root = ET.parse(results/'tests.xml').getroot()
    cases = list(root.iter('testcase'))
    if not cases or any(c.find('failure') is not None or c.find('error') is not None or
                        c.find('skipped') is not None or c.get('status')=='notrun' or
                        c.get('result') in ('skipped','suppressed') for c in cases):
        raise ValueError('Tests failed, skipped or missing')
    main = [c for c in cases if 'All216Shapes' in c.get('classname','') and
            c.get('name','').startswith('MatchesEigenHalf/')]
    if len(main)!=216 or len(set(c.get('name') for c in main))!=216:
        raise ValueError('Expected all 216 WMMA shape tests')
    required = {'PartialBlock48x80','NonDyadicInputs','ZeroInnerDimension','EmptyOutput',
                'IncompatibleDimensions','RejectsNonMultipleDimensions',
                'DeepDataCopyAndMove','MatrixSharesAndMoves','Overflow'}
    if not required.issubset({c.get('name') for c in cases}):
        raise ValueError('Required boundary/ownership tests missing')
    raw = (results/'benchmarks.json').read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    groups = {(method,n):[] for method in METHODS for n in SIZES}
    for item in data.get('benchmarks',[]):
        if item.get('error_occurred'): raise ValueError(item.get('error_message','Benchmark error'))
        if item.get('run_type','iteration')!='iteration': continue
        m = re.fullmatch(r'BM_(WMMA|SharedFP32|SharedFP16)/(\d+)/manual_time',item.get('name',''))
        if not m: continue
        key = (m.group(1),int(m.group(2)))
        if key not in groups: continue
        value = float(item['real_time'])*UNITS[item['time_unit']]
        if not math.isfinite(value) or value<=0: raise ValueError('Invalid measured time')
        groups[key].append(value)
    if any(len(v)!=5 for v in groups.values()):
        raise ValueError('Expected exactly 5 repetitions for each of 3 methods and 7 sizes')
    context = data.get('context',{})
    for key in ['gpu','compute_capability','cuda_driver','cuda_runtime']:
        if key not in context: raise ValueError('Missing environment: '+key)
    if context.get('library_build_type','').lower()!='release':
        raise ValueError('Benchmark library must be Release')
    commit=(results/'commit.txt').read_text(encoding='utf-8-sig').strip()
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Missing/invalid commit hash')
    for name in ['nvcc.txt','nvidia-smi.txt','source-status.txt']:
        if not (results/name).is_file(): raise ValueError('Missing '+name)
    rows=[]
    for n in SIZES:
        row={'n':n}
        for method in METHODS:
            samples=groups[method,n]
            row[method+'_seconds']=statistics.median(samples)
            row[method+'_min']=min(samples); row[method+'_max']=max(samples)
        row['speedup_previous']=row['SharedFP32_seconds']/row['WMMA_seconds']
        row['speedup_same_input']=row['SharedFP16_seconds']/row['WMMA_seconds']
        rows.append(row)
    return {'context':context,'rows':rows,'test_count':len(cases),'repetitions':5,
            'commit':commit,'source_status':(results/'source-status.txt').read_text(encoding='utf-8-sig'),
            'benchmark_sha256':hashlib.sha256(raw).hexdigest(),
            'tests_sha256':hashlib.sha256((results/'tests.xml').read_bytes()).hexdigest()}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path); a=p.parse_args()
    s=validate(a.results)
    print('Validated %d tests and 105 measured iterations'%s['test_count'])
