"""Completeness judge, Sat(y, m, t): does reply y convey what Required statement m needs for task t? The judge sees the
task, the reply, and the label's required_content from data/need_labels.json (the statement's constraints and the
necessity judges' description of the conflict in this task)."""
import json, re
from .lib import PROMPT_DIR, call_with_retries

def required_content(need, statement_id, task_file):
    label = need[(statement_id, task_file)]
    return label['required_content'] if label['need'] else []

def request_for(task_text, specs, reply_text, model_config):
    payload = {'task': task_text, 'required_content': specs, 'outgoing_reply': reply_text}
    options = {k: v for k, v in model_config.items() if k != 'provider'}
    options['messages'] = [{'role': 'system', 'content': (PROMPT_DIR / 'completeness' / 'sat-judge.txt').read_text()},
                           {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, indent=1)}]
    options['response_format'] = {'type': 'json_object'}
    return model_config['provider'], options

def validate(parsed):
    """A JSON object with exactly satisfied in {0, 1} and a string evidence."""
    if not isinstance(parsed, dict): return ['not a JSON object']
    issues = []
    if set(parsed) != {'satisfied', 'evidence'}: issues.append('unexpected keys: ' + str(sorted(parsed)))
    if parsed.get('satisfied') not in (0, 1): issues.append('satisfied must be 0 or 1')
    if not isinstance(parsed.get('evidence'), str): issues.append('evidence must be a string')
    return issues

def _norm(s):
    return re.sub(r'[\W_]+', '', s).casefold()

def evidence_verified(parsed, reply_text):
    """True if the quoted evidence appears in the reply, ignoring punctuation and whitespace; empty evidence counts as verified."""
    ev = parsed.get('evidence', '')
    return not ev.strip() or _norm(ev) in _norm(reply_text)

def execute(store, key, sample, statement_id, task_text, specs, reply_text, model_config):
    provider, request = request_for(task_text, specs, reply_text, model_config)
    existing = store.get(key, str(sample), statement_id)
    def parse_and_validate(content):
        parsed = json.loads(content)
        return parsed, validate(parsed)
    result = call_with_retries(provider, request, existing, parse_and_validate)
    if result is not existing and result.get('status') == 'ok':
        result['evidence_verified'] = evidence_verified(result['parsed'], reply_text)
    store.count(existing, result)
    if result is not existing and result['status'] != 'error': store.put(result, key, str(sample), statement_id)
    return result
