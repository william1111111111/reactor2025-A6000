"""Read aggregate records and verify published checksums, tables and privacy."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication', type=Path)
    args = parser.parse_args()
    root = args.publication.resolve()
    read = lambda p: json.loads(p.read_text())
    manifest = read(root / 'MANIFEST.json')
    failures, hashed = [], set()
    for entry in manifest['records']:
        path = root / entry['published']
        assert path.is_relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['published_sha256']:
            failures.append('hash: ' + entry['published'])
        hashed.add(entry['published'])
    patterns = {
        'machine_path': r'/(?:home|public|mnt|tmp)/[A-Za-z0-9]',
        'private_key': r'-----BEGIN .*PRIVATE KEY',
        'credential': r'github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}',
        'ipv4': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'raw_sample_path': r'\b(?:speaker|listener)/[^ /`\"]+/[^ /`\"]+\.npy',
        'excluded_experiment': r'mam_reactor_gt_oracle|4906307fc13a4878|bfit_strict_n24',
    }
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'PUBLICATION_CHECKS.json']
    links = 0
    for path in files:
        if path.is_symlink() or path.suffix not in ('.md', '.json'):
            failures.append('unexpected file: ' + str(path.relative_to(root)))
        text = path.read_text()
        for name, pattern in patterns.items():
            if re.search(pattern, text):
                failures.append(name + ': ' + str(path.relative_to(root)))
        if path.suffix == '.json':
            json.loads(text)
        else:
            for href in re.findall(r'\]\(([^)\n]+)\)', text):
                if href.startswith(('http://','https://','#','mailto:')):
                    continue
                links += 1
                target = path.parent / href.split('#')[0]
                if target.resolve() != root / 'PUBLICATION_CHECKS.json' and not target.exists():
                    failures.append('broken link: ' + str(path.relative_to(root)) + ' -> ' + href)

    def node(value, pointer):
        for key in pointer.split('.'):
            value = value[key]
        return value

    def row(text, label):
        for line in text.splitlines():
            if line.startswith('|'):
                cells = [x.strip() for x in line.strip('|').split('|')]
                if cells[0] == label:
                    return cells[1:]
        raise AssertionError('Missing table label: ' + label)

    history = (root / 'historical/archive_20260930/RESULTS_AND_HISTORY.md').read_text()
    old_verification = read(root / 'archive_20260930/VERIFICATION.json')
    rows = 0
    for check in old_verification['table_checks']:
        if 'source' in check:
            relative = check['source'].split('evidence/persistent_coreset/reports/')[1]
            keys = ('FRC', 'exact_FRD', 'FRDiv', 'FRVar')
        else:
            relative = 'react2025_input_dependence_seed1/summary.json'
            keys = ('FRC', 'exact_FRD', 'FRDiv', 'native_paired_mean_CCC')
        values = node(read(root / 'runs' / relative), check['json_path'])
        expected = [f'{values[k]:.6f}' for k in keys]
        if row(history, check['label']) != expected:
            failures.append('archive table: ' + check['label'])
        rows += 1
    latest = read(root / 'exploration/runs/component_ablations_v1/eval/analysis.json')
    latest_text = (root / 'exploration/runs/component_ablations_v1/eval/RESULTS.md').read_text()
    readme = (root / 'README.md').read_text()
    labels = {'kd_basic':'Basic KD', 'kd_assignment':'Whole-clip assignment',
              'kd_dynamic':'Dynamic residual', 'kd_assignment_dynamic':'Assignment + dynamic',
              'kd_multihead':'Shared backbone + ten linear heads'}
    for name, values in latest['metrics'].items():
        expected = [f'{values[k]:.6f}' for k in ('FRC','exact_FRD','FRDiv','FRVar')]
        if row(latest_text, name)[:4] != expected or row(readme, labels[name]) != expected:
            failures.append('latest table: ' + name)
        rows += 2
    for dataset, value in read(root / 'archive_20260930/OPERATING_POINTS.json').items():
        label = 'REACT2025 all-session VAL' if dataset == 'REACT2025' else '转换版 REACT2024 all-session VAL'
        expected = [str(value['metrics']['contexts'])] + [f"{value['metrics'][k]:.6f}" for k in ('FRC','exact_FRD','FRDiv','FRVar')]
        if row(readme, label) != expected:
            failures.append('operating point: ' + dataset)
        rows += 1
    for name in ('student_method_recovery_v1','distillation_methods_v1','component_ablations_v1'):
        status = read(root / 'exploration/runs' / name / 'status.json')
        if status['stage'] != 'complete':
            failures.append('incomplete: ' + name)
    result = {'publication_date': '2026-10-10', 'source_records': len(manifest['records']),
              'unique_exported_files_hash_verified': len(hashed), 'table_rows_checked': rows,
              'relative_links_checked': links, 'privacy_scans': list(patterns),
              'new_training_or_evaluation': False, 'failures': failures, 'passed': not failures,
              'scope': 'Artifact consistency and aggregate publication only; no scientific certification.'}
    (root / 'PUBLICATION_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
