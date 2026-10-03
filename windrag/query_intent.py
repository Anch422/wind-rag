"""Shared, deterministic routing used equally by every retrieval algorithm."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryIntent:
    scope: str
    farms: frozenset
    events: frozenset
    assets: frozenset
    farm_overview: bool = False


def analyze_query(query):
    text = str(query).lower().replace('’', "'")
    farms = set()
    # Only recognize letters in an explicit farm expression, never arbitrary A/B/C tokens.
    for match in re.finditer(r'\b(?:wind[\s-]+)?farms?[\s:-]+([a-z]\b(?:\s*(?:,|and|&|or|/)\s*[a-z]\b)*)', text):
        farms.update(re.findall(r'\b[a-z]\b', match.group(1)))
    events = frozenset(re.findall(r'\bevent(?:\s+id)?[\s:#_-]*(\d+)\b', text))
    assets = frozenset(re.findall(r'\b(?:turbine|asset)(?:\s+(?:asset|id))?[\s:#_-]*(\d+)\b', text))
    repair = bool(re.search(r'\b(repair\w*|fix\w*|replac\w*|maintenan\w*|service\w*|caus\w*|broke\w*|broken|technician\w*|reports?|troubleshoot\w*|remed\w*|resolve\w*)\b|\bwhy\b|\bwhat\s+led\s+to\b', text))
    sensor = bool(re.search(r'\b(sensor\w*|measure\w*|units?)\b', text))
    problem = bool(re.search(r'\b(problem\w*|anomal\w*|issues?|fault\w*|fail\w*|wrong|malfunction\w*|damage\w*|alarm\w*|breakdown\w*|outage\w*|downtime|errors?|trouble)\b', text))
    health = bool(re.search(r'\b(status|health|condition|overview|operating|operation\w*)\b|\bhow\s+(?:is|are)\b|\bwhat(?:\s+is|\x27s)\s+(?:happening|going\s+on)\b', text))
    normal = bool(re.search(r'\b(normal|healthy)\b|\bwithout\s+(?:anomal\w*|issues?|problems?)\b|\bno\s+(?:anomal\w*|issues?|problems?)\b', text))
    if repair and not (sensor and not problem):
        scope = 'technician_reports'
    elif events or assets:
        scope = 'events'
    elif normal:
        # A comparison needs both labels; an explicit request for normal records does not.
        scope = 'events' if re.search(r'\b(?:and|versus|vs|compare)\b', text) and problem else 'normal_events'
    elif problem:
        scope = 'anomaly_events'
    elif health and not sensor:
        scope = 'events'
    else:
        scope = 'all'
    overview = not sensor and not events and not assets and bool(
        len(farms) > 1 or re.search(r'\b(?:all|every|each|both)\b.*\b(?:wind\s*)?farms?\b|\bwind\s*farms\b|\bfarms\b', text))
    return QueryIntent(scope, frozenset(farms), events, assets, overview)


def matches_document(doc, intent):
    if intent.farms and doc.farm.lower() not in {'wind farm ' + farm for farm in intent.farms}:
        return False
    if intent.scope == 'technician_reports' and doc.kind != 'report':
        return False
    if intent.scope in ('events', 'anomaly_events', 'normal_events') and doc.kind != 'event':
        return False
    # Unlabelled event documents remain candidates; the answer must use their text,
    # rather than treating absence of a metadata label as absence of a problem.
    label = re.search(r'Recorded label:\s*(\w+)', doc.text, re.I)
    if intent.scope == 'anomaly_events' and label and label.group(1).lower() != 'anomaly':
        return False
    if intent.scope == 'normal_events' and label and label.group(1).lower() != 'normal':
        return False
    if intent.events and not any(re.search(r'\bEvent\s+(?:ID:\s*)?0*' + re.escape(event) + r'\b', doc.text, re.I) for event in intent.events):
        return False
    if intent.assets and not any(re.search(r'\b(?:Turbine\s+asset|Asset\s+ID:|Turbine)\s*0*' + re.escape(asset) + r'\b', doc.text, re.I) for asset in intent.assets):
        return False
    return True
