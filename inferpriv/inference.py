"""Recipient guesser: from the reply alone, up to k ranked guesses about the writer's private attributes. A guess counts
as naming an attribute under the same word lists used for direct disclosure (lib.direct_disclosure)."""
import json
from .lib import PROMPT_DIR, call_with_retries, direct_disclosure

def system_prompt(k):
    return (PROMPT_DIR / 'inference' / 'guesser.txt').read_text().replace('{{K}}', str(k))

def request_for(reply_text, k, model_config):
    options = {k_: v for k_, v in model_config.items() if k_ != 'provider'}
    options['messages'] = [{'role': 'system', 'content': system_prompt(k)}, {'role': 'user', 'content': reply_text}]
    if model_config['provider'] == 'openai':   # strict schema: exactly k positions, each a string or null
        options['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'guesses', 'strict': True, 'schema': {
            'type': 'object', 'additionalProperties': False, 'required': ['guesses'],
            'properties': {'guesses': {'type': 'array', 'minItems': k, 'maxItems': k, 'items': {'type': ['string', 'null']}}}}}}
    else:
        options['response_format'] = {'type': 'json_object'}
    return model_config['provider'], options

def validate(parsed, k):
    if not isinstance(parsed, dict) or set(parsed) != {'guesses'}: return ['unexpected shape']
    guesses = parsed['guesses']
    if not isinstance(guesses, list) or len(guesses) != k: return [f'guesses must be a list of exactly {k} positions']
    if not all(g is None or isinstance(g, str) for g in guesses): return ['each guess must be a string or null']
    return []

def execute(store, key, sample, reply_text, k, model_config):
    provider, request = request_for(reply_text, k, model_config)
    existing = store.get(key, str(sample))
    def parse_and_validate(content):
        parsed = json.loads(content)
        return parsed, validate(parsed, k)
    result = call_with_retries(provider, request, existing, parse_and_validate)
    store.count(existing, result)
    if result is not existing and result['status'] != 'error': store.put(result, key, str(sample))
    return result

def inferential_hit(guesses, attribute):
    """1 if any non-null guess names `attribute`, else 0; and the matching guesses."""
    hits = []
    for g in guesses:
        if not g: continue
        hit, spans = direct_disclosure(g, attribute)
        if hit: hits.append({'guess': g, 'spans': spans})
    return (1 if hits else 0), hits
