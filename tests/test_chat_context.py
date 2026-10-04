import unittest
from windrag.chat_context import resolve_followup
from windrag.query_intent import analyze_query


class ChatContextTests(unittest.TestCase):
    def test_reports_followup_keeps_user_farm_not_assistant_farm(self):
        history = [{'role': 'You', 'text': 'What problems exist at Wind Farm A?'},
                   {'role': 'LOUIE', 'text': 'Wind Farm C has a converter failure.'}]
        resolved = resolve_followup('check technician reports', history)
        self.assertEqual(analyze_query(resolved).farms, frozenset({'a'}))
        self.assertEqual(analyze_query(resolved).scope, 'technician_reports')

    def test_new_scope_overrides_and_new_chat_is_empty(self):
        history = [{'role': 'You', 'text': 'Wind Farm A event 40 generator bearing'}]
        self.assertEqual(resolve_followup('Check reports at Wind Farm C', history), 'Check reports at Wind Farm C')
        self.assertEqual(resolve_followup('Check all wind farms', history), 'Check all wind farms')
        self.assertEqual(resolve_followup('Check reports', []), 'Check reports')
        history.append({'role': 'You', 'text': 'Check all wind farms'})
        self.assertEqual(resolve_followup('Check reports', history), 'Check reports')

    def test_identifiers_and_component_followup(self):
        history = [{'role': 'You', 'text': 'What happened in event 40 at Wind Farm A generator bearing?'}]
        resolved = resolve_followup('How to repair it?', history)
        self.assertEqual(analyze_query(resolved).events, frozenset({'40'}))
        self.assertIn('generator bearing', resolved)
        self.assertEqual(analyze_query(resolve_followup('Check reports for event 42', history)).events, frozenset({'42'}))
        self.assertFalse(analyze_query(resolve_followup('What does sensor_0 measure?', history)).events)
        history.append({'role': 'You', 'text': 'Check event 42'})
        self.assertEqual(analyze_query(resolve_followup('Check reports', history)).events, frozenset({'42'}))
