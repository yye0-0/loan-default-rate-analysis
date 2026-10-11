import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verification_gate import require_catboost


class RequestedModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.report = {'data_kind': 'SYNTHETIC_FUNCTIONAL_TEST', 'horizons': {},
                       'optional_models_not_evaluated': []}
        for h in (12, 24):
            self.report['horizons'][str(h)] = {
                phase: {'catboost': {'ipcw_brier': .1}}
                for phase in ('validation', 'test')}
            (self.root / f'catboost_{h}m.joblib').write_bytes(b'test fixture')

    def test_missing_training_cannot_pass(self):
        self.report['optional_models_not_evaluated'] = [
            {'model': 'catboost', 'horizon': 12, 'reason': 'training failed'}]
        with self.assertRaisesRegex(ValueError, 'not evaluated'):
            require_catboost(self.report, self.root)

    def test_missing_horizon_or_nonfinite_score_cannot_pass(self):
        base = json.dumps(self.report)
        del self.report['horizons']['24']['test']['catboost']
        with self.assertRaisesRegex(ValueError, 'Missing CatBoost'):
            require_catboost(self.report, self.root)
        self.report = json.loads(base)
        self.report['horizons']['12']['validation']['catboost']['ipcw_brier'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid CatBoost'):
            require_catboost(self.report, self.root)

    def test_missing_saved_model_cannot_pass(self):
        (self.root / 'catboost_24m.joblib').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing saved'):
            require_catboost(self.report, self.root)

    def test_complete_result_preserves_synthetic_marker(self):
        result = require_catboost(self.report, self.root)
        self.assertEqual(result['horizons'], [12, 24])
        self.assertEqual(result['data_kind'], 'SYNTHETIC_FUNCTIONAL_TEST')
