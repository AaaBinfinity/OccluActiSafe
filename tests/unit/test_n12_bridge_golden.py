"""桥接 Python 不依赖 NumPy 的 N02 黄金消息往返检查。"""
import json
from pathlib import Path
import unittest
from occluactisafe.common.schemas import ObservationMessage, PredictionMessage

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'n02'


class BridgeGoldenTests(unittest.TestCase):
    def test_observation_and_prediction(self):
        for cls, filename in ((ObservationMessage, 'golden_observation.json'),
                              (PredictionMessage, 'golden_prediction.json')):
            with self.subTest(filename=filename):
                source = json.loads((FIXTURES / filename).read_text(encoding='utf-8'))
                message = cls.from_dict(source)
                self.assertEqual(json.loads(message.to_json()), source)
                self.assertEqual(cls.from_json(message.to_json()), message)


if __name__ == '__main__': unittest.main()
