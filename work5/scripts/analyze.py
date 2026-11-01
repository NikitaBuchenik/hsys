"""Validate real measurements before producing any figures. Python 3.9+."""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

SIZES = [8 << exponent for exponent in range(29)]
METHODS = ('Shared', 'Shuffle')
UNITS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}
EXTRA_TESTS = {'Empty', 'MoreThanTwoPasses', 'WarpAndBlockTails', 'ChunkedWithTail',
               'ReusedWorkspace', 'RejectsInvalidArguments', 'MoveAndHostCopies', 'Overflow'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(destination):
    destination = Path(destination)
    root = Path(__file__).resolve().parents[1]
    files = [root / 'CMakeLists.txt', root / 'README.md'] + list(root.glob('*.cmd'))
    for directory in ['core', 'tests', 'benchmarks', 'scripts']:
        files += [p for p in (root / directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix != '.pyc']
    hashes = {}
    for source in files:
        relative = source.relative_to(root).as_posix()
        target = destination / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = digest(target)
    (destination / 'source-manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


def validate(results):
    results = Path(results)
    cases = list(ET.parse(results / 'tests.xml').getroot().iter('testcase'))
    if len(cases) != 24:
        raise ValueError('Expected 24 tests (16 parameter cases plus 8 boundary/ownership tests)')
    if any(c.find('failure') is not None or c.find('error') is not None or
           c.find('skipped') is not None or c.get('status') != 'run' or
           c.get('result') != 'completed' for c in cases):
        raise ValueError('A test failed, was skipped, or did not complete')
    names = {c.get('name') for c in cases if 'AllSizes' in c.get('classname', '')}
    if names != {'MatchesEigenSum/' + str(i) for i in range(16)}:
        raise ValueError('Missing a required size/strategy test')
    if not EXTRA_TESTS.issubset({c.get('name') for c in cases}):
        raise ValueError('Missing edge tests')
    data = json.loads((results / 'benchmarks.json').read_text(encoding='utf-8-sig'))
    ctx = data.get('context', {})
    for key in ['gpu', 'compute_capability', 'cuda_driver', 'cuda_runtime', 'timing', 'input_pattern']:
        if not ctx.get(key):
            raise ValueError('Missing benchmark context: ' + key)
    if ctx.get('library_build_type', '').lower() != 'release':
        raise ValueError('Expected Release benchmarks')
    capacity = int(ctx['chunk_capacity_elements'])
    if not (1 << 18) <= capacity <= (1 << 26) or capacity & (capacity - 1):
        raise ValueError('Invalid chunk capacity')
    groups = {(method, n): [] for method in METHODS for n in SIZES}
    ids = {key: set() for key in groups}
    for item in data.get('benchmarks', []):
        if item.get('error_occurred'):
            raise ValueError(item.get('error_message', 'Benchmark reported an error'))
        if item.get('run_type', 'iteration') != 'iteration':
            continue
        match = re.fullmatch(r'BM_(Shared|Shuffle)/(\d+)/manual_time', item.get('name', ''))
        if not match:
            raise ValueError('Unexpected benchmark: ' + item.get('name', ''))
        key = (match[1], int(match[2]))
        if key not in groups:
            raise ValueError('Unexpected vector size')
        n = key[1]
        chunks = (n + capacity - 1) // capacity
        label = 'streamed_compute_only' if chunks > 1 else 'resident_compute_only'
        if item.get('label') != label or item.get('chunks') != chunks or \
           item.get('chunk_elements') != min(n, capacity) or item.get('input_bytes') != n * 4 or \
           item.get('verified') != 1:
            raise ValueError('Invalid chunk/method metadata')
        seconds = float(item['real_time']) * UNITS[item['time_unit']]
        if not math.isfinite(seconds) or seconds <= 0 or item.get('iterations', 0) < 1:
            raise ValueError('Invalid CUDA time')
        groups[key].append(seconds)
        ids[key].add(item.get('repetition_index'))
    if any(len(samples) != 5 or ids[key] != set(range(5)) for key, samples in groups.items()):
        raise ValueError('Expected 5 repetitions for both methods at all 29 sizes (290 rows)')
    commit = (results / 'commit.txt').read_text(encoding='utf-8-sig').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Invalid Git commit')
    required = ['nvcc.txt', 'nvidia-smi.txt', 'cpu.txt', 'os.txt', 'compilers.txt', 'source-status.txt',
                'source-manifest.json', 'tests.xml', 'benchmarks.json', 'commit.txt']
    for name in required:
        if not (results / name).is_file():
            raise ValueError('Missing run evidence: ' + name)
    manifest = json.loads((results / 'source-manifest.json').read_text(encoding='utf-8'))
    if not {'core/src/reduction.cu', 'tests/vector_tests.cpp', 'benchmarks/vector_benchmark.cpp'}.issubset(manifest):
        raise ValueError('Source snapshot is incomplete')
    for name, expected in manifest.items():
        source = (results / 'source' / name).resolve()
        if (results / 'source').resolve() not in source.parents or digest(source) != expected:
            raise ValueError('Source snapshot changed: ' + name)
    rows = []
    for n in SIZES:
        row = {'n': n, 'chunks': (n + capacity - 1) // capacity,
               'mode': 'streamed' if n > capacity else 'resident'}
        for method in METHODS:
            samples = groups[method, n]
            row[method + '_seconds'] = statistics.median(samples)
            row[method + '_min'] = min(samples)
            row[method + '_max'] = max(samples)
        row['speedup'] = row['Shared_seconds'] / row['Shuffle_seconds']
        # Conservative descriptive range, not a confidence interval.
        row['speedup_low'] = row['Shared_min'] / row['Shuffle_max']
        row['speedup_high'] = row['Shared_max'] / row['Shuffle_min']
        rows.append(row)
    return {'rows': rows, 'context': ctx, 'commit': commit, 'test_count': len(cases),
            'source_status': (results / 'source-status.txt').read_text(encoding='utf-8-sig'),
            'evidence_sha256': {name: digest(results / name) for name in required}}


def plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = Path(results)
    summary = validate(results)
    rows = summary['rows']
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.25})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    for method, color in [('Shared', '#255f9f'), ('Shuffle', '#cf542e')]:
        for mode, style in [('resident', '-'), ('streamed', '--')]:
            subset = [r for r in rows if r['mode'] == mode]
            if not subset:
                continue
            x = [r['n'] for r in subset]
            ax.plot(x, [r[method + '_seconds'] * 1e6 for r in subset], style + 'o',
                    markersize=3, color=color, label=method + ' / ' + mode)
            ax.fill_between(x, [r[method + '_min'] * 1e6 for r in subset],
                            [r[method + '_max'] * 1e6 for r in subset], color=color, alpha=0.12)
    ax.set(xscale='log', yscale='log', xlabel='Vector length n', ylabel='GPU compute time (microseconds)',
           title='Complete reduction: median of 5 runs, shading = min/max')
    ax.legend()
    fig.savefig(results / 'complexity.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    for mode, style in [('resident', '-'), ('streamed', '--')]:
        subset = [r for r in rows if r['mode'] == mode]
        if not subset:
            continue
        x = [r['n'] for r in subset]
        ax.plot(x, [r['speedup'] for r in subset], style + 'o', markersize=3, label=mode)
        ax.fill_between(x, [r['speedup_low'] for r in subset], [r['speedup_high'] for r in subset], alpha=0.12)
    ax.axhline(1, color='black', linewidth=1)
    ax.set(xscale='log', xlabel='Vector length n', ylabel='Shared time / shuffle time',
           title='Shuffle speedup: above 1 means faster; transfers excluded')
    ax.legend()
    fig.savefig(results / 'speedup.png', dpi=180)
    plt.close(fig)
    with (results / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary['chart_sha256'] = {name: digest(results / name) for name in ['complexity.png', 'speedup.png']}
    (results / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PLOTS SUCCESS: ' + str(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--snapshot', type=Path)
    action.add_argument('--validate', type=Path)
    action.add_argument('--plots', type=Path)
    args = parser.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
    elif args.validate:
        validate(args.validate)
        print('VALIDATED: 24 tests, 290 measured rows, all 29 sizes including 2^31')
    else:
        plots(args.plots)
