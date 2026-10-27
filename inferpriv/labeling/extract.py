"""Step 1, constraint extraction: each model reads one memory statement (prompts/labeling/constraint-extraction.txt, the
statement in its {user_profile} placeholder) and returns at most one explicit constraint. A statement's constraint set is
the union over the models, names neutralized and exact duplicates merged."""
import concurrent.futures, re
from .. import benchmark
from ..lib import PROMPT_DIR, Store, call_with_retries
from .models import MODELS, MERGE_ORDER, parse_json, neutral

def memory_statements(cfg):
    """[(statement id, kind, text)]: the sensitive statements of the config's attributes, then every memory statement of
    its personas (accommodations included), with the ids used in data/need_labels.json."""
    items = [(s['record_id'], s['target_attribute'], s['information_attribute']['memory_statement'])
             for s in benchmark.load_statements() if s['target_attribute'] in cfg['attributes']]
    items += [(f"profile:{p['bio']['name']}:{k}", 'accommodation' if k.startswith('accommodation_') else 'base', v['memory_statement'])
              for p in benchmark.load_profiles() if p['bio']['name'] in cfg['profiles'] for k, v in p['information_attributes'].items()]
    return items

def parse(content):
    p = parse_json(content)
    return p, [] if isinstance(p, dict) and isinstance(p.get('constraints'), list) else ['no constraints list']

def extract(model, items, calls_dir):
    """One model over every statement: {statement id: [constraint dicts]}, and the call statistics."""
    template = (PROMPT_DIR / 'labeling' / 'constraint-extraction.txt').read_text()
    store, cfg = Store(calls_dir / f'extraction__{model}.json'), MODELS[model]
    def run(item):
        sid, _, text = item
        req = {k: v for k, v in cfg.items() if k != 'provider'}
        req.update(messages=[{'role': 'user', 'content': template.replace('{user_profile}', text)}], response_format={'type': 'json_object'})
        existing = store.get(sid)
        r = call_with_retries(cfg['provider'], req, existing, parse)
        store.count(existing, r)
        if r is not existing and r['status'] != 'error': store.put(r, sid)
        return sid, r
    try:
        with concurrent.futures.ThreadPoolExecutor(8 if cfg['provider'] == 'anthropic' else 16) as ex: res = list(ex.map(run, items))
    finally: store.flush()
    bad = [sid for sid, r in res if r['status'] != 'ok']
    if bad: raise SystemExit(f'{model}: {len(bad)} extraction calls not ok, e.g. {bad[:3]}; rerun to retry them')
    return {sid: r['parsed']['constraints'] for sid, r in res}, store.stats

norm = lambda s: re.sub(r'\W+', ' ', s.lower()).strip()

DESCRIPTION = ('Explicit user constraints extracted from each memory statement by three models (prompts/labeling/'
               'constraint-extraction.txt). A statement\'s constraint set is the union of what the models extract, names '
               'replaced by "the user" and exact duplicates merged; extracted_by lists the models that produced each constraint. '
               'Statements with an empty set are Not Required for every task.')

def constraint_sets(items, extracted):
    """extracted: {model: {statement id: [constraint dicts]}} -> data/constraints.json's 'statements' mapping."""
    names = {w for p in benchmark.load_profiles() for w in p['bio']['name'].split()}
    out = {}
    for sid, kind, text in items:
        found = {}
        for m in MERGE_ORDER:
            for c in extracted[m][sid]:
                constraint, rule = neutral(c.get('constraint', ''), names), neutral(c.get('verification_rule', ''), names)
                found.setdefault(norm(constraint), {'constraint': constraint, 'verification_rule': rule, 'extracted_by': []})['extracted_by'].append(m)
        out[sid] = {'kind': kind, 'statement': text, 'constraints': list(found.values())}
    return out

def constraints_file(sets):
    """The contents of data/constraints.json."""
    return {'description': DESCRIPTION, 'models': list(MODELS), 'statements': sets}
