"""The three models that extract constraints and judge task necessity, and the name neutralizer for extracted constraints."""
import json, re

MODELS = {'gpt-6.1-sol': {'provider': 'openai', 'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium', 'max_completion_tokens': 16000},
          'deepseek-flash': {'provider': 'deepseek', 'model': 'deepseek-flash', 'reasoning_effort': 'medium', 'max_tokens': 10000},
          'claude-haiku-5-5': {'provider': 'anthropic', 'model': 'claude-haiku-5-5', 'effort': 'medium', 'thinking': {'type': 'adaptive'}, 'max_tokens': 10000}}
MERGE_ORDER = ['deepseek-flash', 'gpt-6.1-sol', 'claude-haiku-5-5']   # order in which constraints enter a statement's set

def parse_json(content):
    """The first JSON object in a reply. Haiku sometimes adds a note after a ```json block; DeepSeek sometimes wraps the
    answer as {"type": "json_object", "content": {...}}."""
    try: p = json.loads(content)
    except ValueError:
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.S) or re.search(r'(\{.*\})', content, re.S)
        if not m: raise
        p = json.loads(m.group(1))
    if isinstance(p, dict) and p.get('type') == 'json_object':
        p = {k: v for k, v in p.items() if k != 'type'}
        if set(p) == {'content'} and isinstance(p['content'], dict): p = p['content']
    return p

OBJECT_NEXT = r'(?=\s*(?:$|[.,;:!?)]|to\b|and\b|or\b|with\b|from\b|at\b|in\b|on\b|for\b|by\b|as\b|a\b|an\b|the\b|any\b|that\b))'
def neutral(text, names):
    """Constraints are extracted from one persona's memory but shown to judges next to any persona's reply: replace
    persona names and gendered pronouns with 'the user'."""
    alt = '|'.join(sorted(names, key=len, reverse=True))
    text = re.sub(rf"\b(?:{alt})(?:\s+(?:{alt}))?'s\b", "the user's", text)
    text = re.sub(rf'\b(?:{alt})(?:\s+(?:{alt}))?\b', 'the user', text)
    text = re.sub(r'\b(?:his|hers)\b', "the user's", text, flags=re.I)
    text = re.sub(r'\bher\b' + OBJECT_NEXT, 'the user', text, flags=re.I)
    text = re.sub(r'\bher\b', "the user's", text, flags=re.I)
    text = re.sub(r'\b(?:he|she|him|himself|herself)\b', 'the user', text, flags=re.I)
    text = re.sub(r"(?:the user's|the user)\s+(?:the user's|the user)", 'the user', text)
    return re.sub(r'(^|[.!?]\s+)the user', lambda m: m.group(1) + 'The user', text)
