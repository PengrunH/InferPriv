"""Step 3, final labels: one Need(m, t) for every memory statement and task (data/need_labels.json). Statements without a
constraint are Not Required for every task. For the others, pairs where the three models agree keep their label, and
split pairs take the majority unless data/manual_overrides.json sets them by hand. required_content, what the
completeness judge checks a reply against, is the statement's constraints followed by each model's description of the
conflict in this task."""
from collections import Counter
from .. import benchmark
from .models import MODELS

DESCRIPTION = ('Final Need(m,t) for every (memory statement, task) pair; need = 1 means Required. decided_by: no constraint '
               '(no constraint was extracted from the statement, data/constraints.json, so it is Not Required and was not judged), '
               'unanimous (the three models judging task necessity with prompts/labeling/task-necessity.txt agree), majority '
               '(split vote, hand-checked and agreeing with the majority) or manual (split vote set by hand, '
               'data/manual_overrides.json). required_content is what the completeness judge checks a reply against.')

def decide(votes, override):
    if len(set(votes.values())) == 1: return next(iter(votes.values())), 'unanimous'
    if override is not None: return override['need'], 'manual'
    return int(sum(votes.values()) >= 2), 'majority'

def aggregate(sets, judged, overrides, tasks):
    """sets: data/constraints.json 'statements'; judged: {model: {(statement id, task): (label, conflict)}};
    overrides: data/manual_overrides.json. Returns the label list of data/need_labels.json, every statement x task."""
    man = {(o['statement_id'], o['task_file']): o for o in overrides}
    labels = []
    for sid, v in sets.items():
        for t in tasks:
            if not v['constraints']:
                labels.append({'statement_id': sid, 'task_file': t, 'need': 0, 'decided_by': 'no constraint', 'votes': None, 'required_content': []})
                continue
            votes = {m: judged[m][(sid, t)][0] for m in MODELS}
            need, how = decide(votes, man.get((sid, t)))
            conflicts = [f'Conflict in this task: {c}' for m in MODELS if (c := judged[m][(sid, t)][1])]
            labels.append({'statement_id': sid, 'task_file': t, 'need': need, 'decided_by': how, 'votes': votes,
                           'required_content': [c['constraint'] for c in v['constraints']] + conflicts})
    return labels

def need_file(labels):
    """The contents of data/need_labels.json."""
    return {'description': DESCRIPTION, 'models': list(MODELS),
            'counts': dict(Counter(f"{l['decided_by']}:{l['need']}" for l in labels)), 'labels': labels}

def summary(labels, cfg):
    """Counts as reported in the paper, and the attribute x task pairs that are not protected."""
    c = Counter((l['decided_by'], l['need']) for l in labels)
    need = {(l['statement_id'], l['task_file']): l for l in labels}
    unprotected = [(a, t) for a in cfg['attributes'] for t in cfg['tasks'] if not benchmark.is_protected(a, t, need)]
    judged = len(labels) - c[('no constraint', 0)]
    return (f"{len(labels)} statement-task pairs: {c[('no constraint', 0)]} without a constraint (Not Required); of the {judged} judged, "
            f"{c[('unanimous', 0)] + c[('unanimous', 1)]} unanimous ({c[('unanimous', 1)]} Required), "
            f"{c[('majority', 1)]} Required and {c[('majority', 0)]} Not Required by majority, {c[('manual', 0)] + c[('manual', 1)]} set by hand; "
            f"{sum(l['need'] for l in labels)} Required in total. Protected attribute x task pairs: "
            f"{len(cfg['attributes']) * len(cfg['tasks']) - len(unprotected)} of {len(cfg['attributes']) * len(cfg['tasks'])}"
            + (f" (not protected: {', '.join(f'{a} x {t[:-5]}' for a, t in unprotected)})" if unprotected else ''))
