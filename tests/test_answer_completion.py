import copy
import unittest
from unittest.mock import patch
from windrag.engine import Ollama, Settings


class AnswerCompletionTests(unittest.TestCase):
    def test_truncation_regenerates_and_counts_both_requests(self):
        client = Ollama(Settings())
        payloads = []
        responses = iter([
            {'message': {'content': 'An unfinished list'}, 'done_reason': 'length', 'eval_count': 1024, 'prompt_eval_count': 20},
            {'message': {'content': 'Complete answer [C:event:55].'}, 'done_reason': 'stop', 'eval_count': 200, 'prompt_eval_count': 20},
        ])
        def post(endpoint, payload):
            payloads.append(copy.deepcopy(payload))
            return next(responses)
        with patch.object(client, 'post', side_effect=post):
            answer, usage = client.chat('Ground answers.', 'Question')
        self.assertEqual(answer, 'Complete answer [C:event:55].')
        self.assertEqual([p['options']['num_predict'] for p in payloads], [1024, 2048])
        self.assertEqual(payloads[0]['messages'], payloads[1]['messages'])
        self.assertEqual(usage['answer_tokens'], 1224)
        self.assertEqual(usage['prompt_tokens'], 40)
        self.assertEqual(usage['generation_attempts'], 2)

    def test_persistent_truncation_is_failure_not_successful_partial_answer(self):
        client = Ollama(Settings())
        with patch.object(client, 'post', return_value={'message': {'content': 'Partial'}, 'done_reason': 'length'}) as post:
            with self.assertRaisesRegex(ValueError, 'No incomplete answer was recorded'):
                client.chat('System', 'Question')
            self.assertEqual(post.call_count, 2)

    def test_legacy_cutoff_detection_and_reranker_budget(self):
        client = Ollama(Settings())
        with patch.object(client, 'post', side_effect=[
            {'message': {'content': 'Partial'}, 'eval_count': 1024},
            {'message': {'content': 'Complete.'}, 'eval_count': 10},
        ]):
            self.assertEqual(client.chat('System', 'Question')[0], 'Complete.')
        with patch.object(client, 'post', return_value={'message': {'content': '{}'}, 'done_reason': 'length'}) as post:
            with self.assertRaises(ValueError):
                client.chat('System', 'Question', schema={'type': 'object'}, max_tokens=128)
            self.assertEqual(post.call_count, 1)
