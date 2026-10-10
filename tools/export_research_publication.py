"""Publish aggregate research records, not datasets/checkpoints/raw run dumps.

Run in a separate publication checkout; input trees are read-only. Sources may
contain uncommitted work. A source SHA identifies the input text, not the commit
that trained a model. This deliberately does NOT export private Mam oracle work.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SECRET = re.compile(r'-----BEGIN .*PRIVATE KEY|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}')
EXCLUDED = re.compile(r'mam.reactor.gt.oracle|4906307fc13a4878|bfit_strict_n24', re.I)
PRIVATE_KEYS = {'rows', 'per_context', 'contexts_rows', 'samples', 'sample_ids', 'clip_ids',
                'clip_id', 'sample_id', 'session_id', 'source_ids', 'target_ids', 'mapping',
                'records', 'gt_files', 'source_paths', 'target_paths', 'predictions',
                'gpu_at_start', 'gpu_before_training', 'pid', 'children', 'hostname', 'username',
                'targets', 'train_ids', 'val_ids', 'test_ids', 'source_schedule', 'view_schedule'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'reports', 'exploration', 'archive', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    roots = {name: getattr(args, name).resolve() for name in ('repo','reports','exploration','archive')}
    candidates = {}

    def add(path, relative, category):
        if path.is_file() and not path.is_symlink():
            candidates[path.resolve()] = (relative, category)

    for p in sorted((roots['repo'] / 'coreset_reactor/results').rglob('*.md')):
        rel = p.relative_to(roots['repo'] / 'coreset_reactor/results')
        add(p, 'historical/' + str(rel), 'historical_report')
    for p in sorted((roots['repo'] / 'reports').glob('*.md')):
        add(p, 'historical/early_reports/' + p.name, 'historical_report')
    for p in sorted(roots['reports'].rglob('*.md')):
        rel = p.relative_to(roots['reports'])
        if not any(part in ('source_snapshot', 'source_addenda', 'student_recipe_bank_20260924') for part in rel.parts):
            add(p, 'runs/' + str(rel), 'run_report')
    for p in sorted(roots['exploration'].glob('*.md')):
        add(p, 'exploration/' + p.name, 'exploration_protocol_or_report')
    for name in ('OPERATING_POINTS.json', 'VERIFICATION.json'):
        add(roots['archive'] / name, 'archive_20260930/' + name, 'aggregate_provenance')
    runs = ('student_method_recovery_v1', 'distillation_methods_v1', 'component_ablations_v1',
            'matched_supervision_v1', 'final_analysis_v1')
    for name in runs:
        root = roots['exploration'] / 'runs' / name
        for rel in ('eval/RESULTS.md', 'eval/analysis.json', 'eval/BENCHMARK.json',
                    'summary.json', 'status.json'):
            add(root / rel, 'exploration/runs/' + name + '/' + rel, 'aggregate_exploration')
    names = ('react2025_student_loss_pruning_seed1', 'react2024_student_loss_pruning_seed1',
        'react2025_key_component_ablations_seed1', 'react2025_input_dependence_seed1',
        'mode_reference_student_seed1', 'react2024_mode_reference_distillation_seed1',
        'react2024_reference_count4_student_seed1', 'teacher_convergence_2400_seed1',
        't1_student_distillation_seed1', 't1_student_refinement_seed1',
        'mode_reference_teacher_v2_seed1', 'react2024_reference_count4_seed1')
    for name in names:
        root = roots['reports'] / name
        for rel in ('summary.json', 'experiment.json', 'provenance.json', 'matched_summary.json',
                    'teacher_upgrade_comparison.json', 'frozen_full_config.json',
                    'frozen_full_provenance.json', 'minimal/config.json', 'residual/config.json'):
            add(root / rel, 'runs/' + name + '/' + rel, 'aggregate_config_or_result')
    substitutions = [(str(roots['repo']), '$SOURCE_ROOT'),
                     (str(roots['reports']), '$RUN_ROOT'),
                     (str(roots['archive']), '$ARCHIVE_ROOT'),
                     (str(roots['exploration']), '$EXPLORATION_ROOT'),
                     (str(roots['reports'].parents[1]), '$PROJECT_ROOT')]
    substitutions.sort(key=lambda item: -len(item[0]))

    def clean(text):
        for original, replacement in substitutions:
            text = text.replace(original, replacement)
        text = re.sub(r'/(?:home|public|mnt|data|tmp)/[^\s`"\'<>|\)\]\}，；]+',
                      lambda m: '$LOCAL_PATH/' + Path(m[0]).name, text)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[HOST_REDACTED]', text)
        text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]', text)
        text = re.sub(r'(?i)\b(?:speaker|listener)/[^/\s`]+/[^\s`"<>|)]+\.npy', '[SAMPLE_PATH_REDACTED]', text)
        return text

    omitted = []
    def clean_json(value, pointer=''):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key.lower() in PRIVATE_KEYS:
                    omitted.append(pointer + '/' + key)
                    continue
                safe_key = clean(key)
                if safe_key in result:
                    raise ValueError('Redacted key collision: ' + pointer)
                result[safe_key] = clean_json(item, pointer + '/' + key)
            return result
        if isinstance(value, list):
            if len(value) > 100:
                omitted.append(pointer)
                return {'not_published': 'long array, possible per-sample data', 'length': len(value)}
            return [clean_json(item, pointer + '/' + str(i)) for i, item in enumerate(value)]
        return clean(value) if isinstance(value, str) else value

    # Byte-identical duplicates are indexed as aliases, not copied repeatedly.
    mapping, by_hash, skipped = {}, {}, []
    for source, (rel, category) in candidates.items():
        text = source.read_text()
        if SECRET.search(text):
            raise ValueError('Potential credential detected; refusing export')
        if EXCLUDED.search(text) or EXCLUDED.search(str(source)):
            skipped.append({'category': category, 'reason': 'excluded research scope'})
            continue
        digest = sha(source)
        destination = by_hash.setdefault(digest, rel)
        mapping[source] = destination
    out.mkdir(parents=True)
    entries, written = [], set()
    for source, destination in mapping.items():
        rel, category = candidates[source]
        omitted.clear()
        if destination not in written:
            target = out / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == '.json':
                content = json.dumps(clean_json(json.loads(source.read_text())), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            else:
                def link(match):
                    label, href = match.groups()
                    if href.startswith(('https://', 'http://', '#', 'mailto:')):
                        return match[0]
                    plain = href.split('#')[0]
                    linked = (source.parent / plain).resolve()
                    if linked in mapping:
                        relative = os.path.relpath(out / mapping[linked], target.parent)
                        suffix = '#' + href.split('#', 1)[1] if '#' in href else ''
                        return f'[{label}]({relative}{suffix})'
                    return '`' + label.strip('`') + '` (local reference: `' + clean(href) + '`)'
                content = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', link, source.read_text())
                content = clean(content)
                content = ('> Sanitized historical record. Dates/status are those of the source; '
                           'see the publication README for current scope and completed follow-ups. '
                           'Local path variables are placeholders, not downloadable assets.\n\n' + content)
            target.write_text(content.rstrip() + '\n')
            written.add(destination)
        source_scope = next((name + '/' + str(source.relative_to(root)) for name,root in roots.items()
                             if source.is_relative_to(root)), source.name)
        entries.append({'source': source_scope, 'source_sha256': sha(source), 'published': destination,
                        'published_sha256': sha(out / destination), 'category': category,
                        'json_fields_omitted': [clean(p) for p in omitted]})
    manifest = {'publication_date': '2026-10-10', 'policy': 'aggregate research records only; no new training or full raw archive',
                'records': entries, 'excluded': skipped, 'unique_exported_files': len(written)}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Published record index', '', 'Historical versions and interrupted/negative runs are retained and labeled.', '',
             '| Source record | Published copy |', '|---|---|']
    lines += [f"| `{e['source']}` | [{e['published']}]({e['published']}) |" for e in entries]
    (out / 'INDEX.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'source_records': len(entries), 'unique_files': len(written), 'excluded': len(skipped)}))


if __name__ == '__main__':
    main()
