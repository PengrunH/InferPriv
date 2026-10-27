"""Rebuild the necessity labels with the three models: extract constraints from every memory statement, judge each
statement with a constraint against every task, and aggregate the votes. Writes constraints.json and need_labels.json to
--out (default results/labeling/), never to data/; API calls are stored in --out/calls/ and reused on a rerun. Needs
OPENAI_API_KEY, DEEPSEEK_API_KEY and ANTHROPIC_API_KEY.

Use this for new personas, sensitive statements or tasks. Rebuilding the released data makes about 2,900 calls.

    python -m inferpriv.labeling.run [--out results/labeling]
"""
import argparse, json
from pathlib import Path
from ..lib import ROOT, CONFIG_DIR, DATA_DIR, RESULTS_DIR, read, load_keys
from .models import MODELS
from .extract import memory_statements, extract, constraint_sets, constraints_file
from .judge import label
from .aggregate import aggregate, need_file, summary

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', type=Path, default=CONFIG_DIR / 'benchmark.json', help='personas, attributes and tasks to label')
    ap.add_argument('--out', type=Path, default=RESULTS_DIR / 'labeling')
    ap.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = ap.parse_args()
    cfg = read(args.config)
    items = memory_statements(cfg)
    load_keys(args.env_file)
    calls = args.out / 'calls'
    extracted = {}
    for m in MODELS:
        extracted[m], st = extract(m, items, calls)
        print(f"  extraction, {m}: {sum(bool(v) for v in extracted[m].values())} statements with a constraint ({st['reused']} reused, {st['computed']} new)", flush=True)
    sets = constraint_sets(items, extracted)
    print(f"  constraint sets: {sum(len(v['constraints']) for v in sets.values())} distinct constraints in {sum(bool(v['constraints']) for v in sets.values())} statements", flush=True)
    judged = {}
    for m in MODELS:
        judged[m], st = label(m, sets, cfg['tasks'], calls)
        print(f"  necessity, {m}: {sum(v[0] for v in judged[m].values())} of {len(judged[m])} pairs Required ({st['reused']} reused, {st['computed']} new)", flush=True)
    labels = aggregate(sets, judged, read(DATA_DIR / 'manual_overrides.json'), cfg['tasks'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'constraints.json').write_text(json.dumps(constraints_file(sets), ensure_ascii=False, indent=1) + '\n')
    (args.out / 'need_labels.json').write_text(json.dumps(need_file(labels), ensure_ascii=False, indent=1) + '\n')
    print(summary(labels, cfg))
    print(f'Wrote {args.out}/constraints.json and need_labels.json. Split pairs without an entry in data/manual_overrides.json '
          'took the majority; check them by hand before copying the files into data/.')

if __name__ == '__main__': main()
