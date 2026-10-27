"""Evaluate a model on InferPriv under one generation prompt: the control run (memory without sensitive statements), then the
attribute run, then the scores.

For every (persona, attribute, task) cell the model writes n replies. On protected cells the guesser reads each reply
and lists k guesses. Every reply of the attribute run is judged for completeness against the memory's Required
statements. Each API result is stored with the hash of its exact request under results/<setup>/, so a rerun reuses
everything and only computes what is missing; a stored result made with a different request stops the run.

    python -m inferpriv.evaluate --model config/models/gpt-6.1-sol.json --prompt baseline [--memory statements]
"""
import argparse, concurrent.futures
from pathlib import Path
from . import benchmark, generation, inference, sat, score
from .lib import ROOT, RESULTS_DIR, Store, combo_key, setup_id, jsonl, load_keys, direct_disclosure

def build_cases(cfg):
    profiles = [p for p in benchmark.load_profiles() if p['bio']['name'] in cfg['profiles']]
    statements = benchmark.load_statements(cfg['memory'])
    cases = []
    for person in profiles:
        for attribute in ([None] if cfg['control'] else cfg['attributes']):
            if attribute and not benchmark.eligible(person, attribute): continue
            memory, record_ids = generation.build_memory(person, statements, attribute)
            for t in cfg['tasks']:
                for sample in range(cfg['n']):
                    cases.append({'person': person, 'attribute': attribute, 'task_file': t, 'sample': sample, 'memory': memory, 'record_ids': record_ids})
    return cases

def open_stores(cfg, results):
    run = Path(results) / benchmark.run_folder(cfg)
    return run, {'generation': Store(run / 'generation.json'),
                 'inference': Store(run / f'inference__{setup_id(cfg["guess"], "k" + str(cfg["k"]))}.json'),
                 'sat': Store(run / f'sat__{setup_id(cfg["judge"])}.json')}

def run_case(c, cfg, tasks, stores, need):
    name, a, t = c['person']['bio']['name'], c['attribute'], c['task_file']
    key, email = combo_key(name, a, t), tasks[t]['email_content']
    row = {'persona': name, 'attribute': a, 'task_file': t, 'sample': c['sample']}
    gen = generation.execute(stores['generation'], key, c['sample'], c['memory'], email, cfg['agent'], cfg['prompt'])
    if gen['status'] != 'ok': return {**row, 'status': 'generation failed'}
    reply = row['reply'] = gen['text']
    # Leakage is only measured on protected tasks; elsewhere the attribute may have to be conveyed, so no guesses are made.
    row['protected'] = a is None or benchmark.is_protected(a, t, need)
    if row['protected']:
        if a: row['direct'] = direct_disclosure(reply, a)[1]
        inf = inference.execute(stores['inference'], key, c['sample'], reply, cfg['k'], cfg['guess'])
        row['guesses'] = inf['parsed']['guesses'] if inf['status'] == 'ok' else None
    if not cfg['control'] and not cfg.get('skip_completeness'):   # completeness is scored on attribute runs only
        row['sat'] = {}
        for sid in benchmark.required_statement_ids(c['person'], c['record_ids'], t, need):
            r = sat.execute(stores['sat'], key, c['sample'], sid, email, sat.required_content(need, sid, t), reply, cfg['judge'])
            row['sat'][sid] = r['parsed']['satisfied'] if r['status'] == 'ok' else None
    return {**row, 'status': 'ok'}

def run(cfg, results=RESULTS_DIR):
    need, tasks, cases = benchmark.load_need_labels(), benchmark.load_tasks(cfg['tasks']), build_cases(cfg)
    folder, stores = open_stores(cfg, results)
    print(f'{folder.name}: {len(cases)} replies', flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['workers']) as ex:
            rows = []
            for i, r in enumerate(ex.map(lambda c: run_case(c, cfg, tasks, stores, need), cases), 1):
                rows.append(r)
                if i % 100 == 0 or i == len(cases): print(f'  {i}/{len(cases)} replies done', flush=True)
    finally:
        for st in stores.values(): st.flush()
    print('  ' + '; '.join(stores[s].describe(s) for s in stores))
    jsonl(folder / 'replies.jsonl', rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True, help='model config, e.g. config/models/gpt-6.1-sol.json')
    ap.add_argument('--prompt', default='baseline', choices=benchmark.PROMPTS)
    ap.add_argument('--memory', default='statements', choices=list(benchmark.MEMORY_VARIANTS),
                    help='form of the sensitive statements: statements (default), unnamed, or declaration')
    ap.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    ap.add_argument('--env-file', type=Path, default=ROOT / '.env', help='KEY=value file with API keys (default: .env)')
    ap.add_argument('--skip-completeness', action='store_true', help='no completeness judgments (the paper skips them for the unnamed and declaration variants)')
    args = ap.parse_args()
    load_keys(args.env_file)
    cfg = {**benchmark.load_config(args.model, args.prompt, args.memory), 'skip_completeness': args.skip_completeness}
    for c in ({**cfg, 'control': True}, cfg): run(c, args.results_dir)
    print(score.describe(score.score(cfg, args.results_dir), cfg['n']))

if __name__ == '__main__': main()
