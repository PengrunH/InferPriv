"""Step 2, task necessity: for every statement with a constraint and every task, each model receives the task message and
the statement's whole constraint set in one call (prompts/labeling/task-necessity.txt) and labels it Required (1) or Not
Required (0)."""
import concurrent.futures
from .. import benchmark
from ..lib import PROMPT_DIR, Store, call_with_retries
from .models import MODELS, parse_json

def as_input(cs):
    """One constraint as is; several as a numbered list, introduced so the label covers the whole set."""
    if len(cs) == 1: return {'constraint_description': cs[0]['constraint'], 'verification_rule': cs[0]['verification_rule']}
    head = ('These constraints were extracted independently from the same memory statement and may describe the same requirement. '
            'Label Required if the reply must address any of them.')
    return {'constraint_description': head + ' ' + ' '.join(f'({i}) {c["constraint"]}' for i, c in enumerate(cs, 1)),
            'verification_rule': ' '.join(f'({i}) {c["verification_rule"]}' for i, c in enumerate(cs, 1))}

def validate(p):
    if not isinstance(p, dict) or p.get('restricts') not in {'yes', 'no'}: return ['missing or invalid label']
    return []

def parse(content):
    p = parse_json(content)
    if isinstance(p, dict) and 'label' in p and 'restricts' not in p:
        p['restricts'] = {1: 'yes', 0: 'no', '1': 'yes', '0': 'no'}.get(p['label'], p['label'])
    return p, validate(p)

def request_for(task, constraint, cfg):
    request = {k: v for k, v in cfg.items() if k != 'provider'}
    msg = (f"Incoming message (sent {task.get('scenario_date', 'date not given')}):\n{task['email_content']}\n\n"
           f"User constraint: {constraint['constraint_description']}\nVerification rule: {constraint['verification_rule']}")
    request['messages'] = [{'role': 'system', 'content': (PROMPT_DIR / 'labeling' / 'task-necessity.txt').read_text()}, {'role': 'user', 'content': msg}]
    request['response_format'] = {'type': 'json_object'}
    return request

def label(model, sets, task_files, calls_dir):
    """One model over every (statement with a constraint, task) pair: {(statement id, task): (label, conflict)}, and the call statistics."""
    tasks, cfg = benchmark.load_tasks(task_files), MODELS[model]
    store = Store(calls_dir / f'necessity__{model}.json')
    jobs = [(s, t) for s, v in sets.items() if v['constraints'] for t in task_files]
    def run(job):
        s, t = job
        existing = store.get(s, t)
        r = call_with_retries(cfg['provider'], request_for(tasks[t], as_input(sets[s]['constraints']), cfg), existing, parse)
        store.count(existing, r)
        if r is not existing and r['status'] != 'error': store.put(r, s, t)
        return r
    try:
        with concurrent.futures.ThreadPoolExecutor(16) as ex: res = list(ex.map(run, jobs))
    finally: store.flush()
    bad = [j for j, r in zip(jobs, res) if r['status'] != 'ok']
    if bad: raise SystemExit(f'{model}: {len(bad)} necessity calls not ok, e.g. {bad[:3]}; rerun to retry them')
    return {j: (int(r['parsed']['restricts'] == 'yes'), r['parsed'].get('conflict', '')) for j, r in zip(jobs, res)}, store.stats
