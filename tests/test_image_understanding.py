import json
import unittest
from unittest.mock import patch

from catvision import vlm_features


class ImageObservationTests(unittest.TestCase):
    def test_two_question_marks_produce_two_image_grounded_answers(self):
        from catvision.vlm import LocalVLM
        engine=LocalVLM.__new__(LocalVLM)
        engine.feature_settings={'question_max_new_tokens':256}
        engine.adapter='adapter'
        outputs=[('耳尖向前折。',{'seconds':0.2,'input_tokens':100,'output_tokens':8}),
                 ('坐在地毯上。',{'seconds':0.3,'input_tokens':90,'output_tokens':7})]
        with patch.object(engine,'_image_text',side_effect=outputs) as generate:
            answer=engine.answer_question(object(),'耳朵怎样？它是什么姿态？')
        self.assertEqual(generate.call_count,2)
        self.assertNotIn('它是什么姿态',generate.call_args_list[0].args[1])
        self.assertIn('耳尖向前折',answer['answer'])
        self.assertIn('坐在地毯上',answer['answer'])
        self.assertEqual(len(answer['subquestions']),2)
        self.assertAlmostEqual(answer['inference_seconds'],0.5)

    def test_reads_actual_observations_and_does_not_fill_missing_fields(self):
        text = '```json\n' + json.dumps({'appearance':['灰白短毛','双耳直立'],
            'pose':'一只猫趴在画面中央','scene':'身后可见木椅'},ensure_ascii=False) + '\n```'
        result = vlm_features.parse_observation(text)
        self.assertEqual(result['features'], ['灰白短毛','双耳直立'])
        self.assertEqual(result['pose'], '一只猫趴在画面中央')
        self.assertEqual(result['scene'], '身后可见木椅')
        self.assertEqual(result['observation_status'], 'ok')
        partial = vlm_features.parse_observation('{"appearance":["皮肤褶皱明显"]}')
        self.assertEqual(partial['features'], ['皮肤褶皱明显'])
        self.assertEqual(partial['pose'], '')
        self.assertEqual(partial['observation_status'], 'partial')

    def test_wrong_types_and_truncated_output_remain_visible_without_invented_features(self):
        result = vlm_features.parse_observation('{"appearance":"黑猫","pose":9,"scene":[]}')
        self.assertEqual(result['features'], [])
        self.assertEqual(result['observation_status'], 'partial')
        broken = '{"appearance":["白毛"],"pose":"'
        result = vlm_features.parse_observation(broken)
        self.assertEqual(result['observation_status'], 'parse_error')
        self.assertEqual(result['observation_raw_text'], broken)


if __name__ == '__main__':
    unittest.main()
