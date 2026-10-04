"""Resolve local chat follow-ups from user-authored messages only."""
import re
from .query_intent import analyze_query


def resolve_followup(query, messages):
    current = analyze_query(query)
    # Naming a farm or explicitly requesting all farms starts a fresh scope.
    if current.farms or current.farm_overview:
        return query
    additions = []
    recent_events = recent_assets = None
    for message in reversed(messages):
        if message['role'] != 'You':
            continue
        previous = message['text']
        intent = analyze_query(previous)
        if intent.farm_overview and not intent.farms:
            break
        if recent_events is None and intent.events:
            recent_events = intent.events
        if recent_assets is None and intent.assets:
            recent_assets = intent.assets
        if intent.farms:
            additions.append(('Wind Farm ' if len(intent.farms) == 1 else 'Wind farms ') + ' and '.join(sorted(farm.upper() for farm in intent.farms)))
            # Specific new identifiers replace old identifiers. Sensor questions
            # should not inherit an unrelated event from the preceding discussion.
            if not re.search(r'\bsensor\w*\b', query, re.I):
                if not current.events:
                    additions.extend('Event ' + event for event in sorted(recent_events or ()))
                if not current.assets:
                    additions.extend('Turbine asset ' + asset for asset in sorted(recent_assets or ()))
                if re.search(r'\b(reports?|repair\w*|fix\w*|cause\w*|it|that|this)\b', query, re.I):
                    components = re.findall(r'\b(?:generator bearing|gearbox|converter|transformer|hydraulic\w*|pitch\s+(?:motor|battery)|rotor\s+brake)\b', previous, re.I)
                    additions.extend(dict.fromkeys(components))
            break
    return query + ('\nChat context: ' + '; '.join(additions) if additions else '')
