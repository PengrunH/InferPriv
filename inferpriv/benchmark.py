"""The benchmark definition: configs, personas, sensitive statements, tasks, necessity labels, and the rules built on them
(which persona-attribute pairs are eligible, which tasks are protected for an attribute, and which memory statements a
reply must satisfy)."""
import functools
from pathlib import Path
from .lib import DATA_DIR, CONFIG_DIR, read, setup_id

PROMPTS = ('baseline', 'privacy-aware')
MEMORY_VARIANTS = {'statements': None, 'unnamed': 'statements_unnamed.json', 'declaration': 'statements_declaration.json'}

def load_config(model, prompt='baseline', memory='statements', control=False, benchmark=CONFIG_DIR / 'benchmark.json'):
    """A run: the fixed benchmark settings plus the model under test (a config/models/ file or a dict), the generation
    prompt, the memory variant, and whether this is the control run (no sensitive statements in memory)."""
    cfg = read(benchmark)
    cfg.pop('description', None)
    if prompt not in PROMPTS: raise SystemExit(f'unknown prompt {prompt!r}; choose from {", ".join(PROMPTS)}')
    if memory not in MEMORY_VARIANTS: raise SystemExit(f'unknown memory variant {memory!r}; choose from {", ".join(MEMORY_VARIANTS)}')
    agent = read(model) if isinstance(model, (str, Path)) else dict(model)
    if 'provider' not in agent or 'model' not in agent: raise SystemExit('a model config needs "provider" and "model"')
    return {**cfg, 'agent': agent, 'prompt': prompt, 'memory': memory, 'control': control}

def run_folder(cfg):
    """Results folder of a run, e.g. gpt-6.1-sol_medium_baseline. Controls hold no sensitive statements, so every memory
    variant of a model and prompt shares one control."""
    base = setup_id(cfg['agent'], cfg['prompt'])
    if cfg['control']: return base + '_control'
    return base + ('' if cfg['memory'] == 'statements' else '_' + cfg['memory'])

def load_profiles(): return read(DATA_DIR / 'profiles.json')

def load_statements(memory='statements'):
    """The sensitive statements. 'unnamed' swaps in each statement's text with the attribute's name removed; 'declaration'
    keeps one statement per attribute with a bare declaration as its text ("The person has type 1 diabetes."). Statement
    ids, attributes and necessity labels are unchanged."""
    statements = read(DATA_DIR / 'statements.json')
    if memory == 'statements': return statements
    texts = {v['record_id']: v['information_attribute']['memory_statement'] for v in read(DATA_DIR / MEMORY_VARIANTS[memory])}
    out = []
    for s in statements:
        if s['record_id'] in texts:
            out.append(dict(s, information_attribute=dict(s['information_attribute'], memory_statement=texts[s['record_id']])))
        elif memory != 'declaration': raise SystemExit(f'no {memory} version of {s["record_id"]}')
    return out

def load_tasks(task_files):
    return {f: read(DATA_DIR / 'tasks' / f) for f in task_files}

def load_need_labels():
    """{(statement id, task file): label} for every memory statement and task."""
    return {(l['statement_id'], l['task_file']): l for l in read(DATA_DIR / 'need_labels.json')['labels']}

# Pregnancy is only assigned to female personas; every other attribute applies to every persona.
ELIGIBILITY = {'pregnancy': lambda bio: bio['sex'] == 'F'}

def eligible(person, attribute):
    rule = ELIGIBILITY.get(attribute)
    return rule is None or rule(person['bio'])

@functools.lru_cache(maxsize=None)
def attribute_record_ids(attribute):
    return tuple(s['record_id'] for s in load_statements() if s['target_attribute'] == attribute)

def is_protected(attribute, task_file, need):
    """T^priv_a: no sensitive statement of the attribute is Required for the task. Always checked over all three, so the
    declaration variant (one statement in memory) is scored on the same tasks as the others."""
    return not any(need[(rid, task_file)]['need'] for rid in attribute_record_ids(attribute))

def accommodation_keys(person):
    return [k for k in person['information_attributes'] if k.startswith('accommodation_')]

def profile_statement_ids(person):
    """Ids of a persona's own memory statements (background, accommodations, daily records): profile:<name>:<key>."""
    return [f"profile:{person['bio']['name']}:{k}" for k in person['information_attributes']]

def required_statement_ids(person, record_ids, task_file, need):
    """Req(M,t): the statements of the memory (the persona's own plus the inserted sensitive statements) that are Required for the task."""
    return [sid for sid in profile_statement_ids(person) + list(record_ids) if need[(sid, task_file)]['need']]
