"""Fail closed when a requested CatBoost experiment was skipped or incomplete."""
import argparse
import json
import math
from pathlib import Path


def require_catboost(report, model_dir):
    skipped = [r for r in report.get('optional_models_not_evaluated', [])
               if r.get('model') == 'catboost']
    if skipped:
        raise ValueError('CatBoost was not evaluated: ' + json.dumps(skipped))
    for horizon in (12, 24):
        result = report['horizons'][str(horizon)]
        for phase in ('validation', 'test'):
            metric = result[phase].get('catboost')
            if metric is None:
                raise ValueError(f'Missing CatBoost {phase} metrics at {horizon} months')
            brier = metric.get('ipcw_brier')
            if not isinstance(brier, (int, float)) or not math.isfinite(brier) or brier < 0:
                raise ValueError(f'Invalid CatBoost Brier score at {horizon} months')
        if not (Path(model_dir) / f'catboost_{horizon}m.joblib').is_file():
            raise ValueError(f'Missing saved CatBoost model at {horizon} months')
    return {'catboost': 'trained_and_evaluated', 'horizons': [12, 24],
            'data_kind': report['data_kind'],
            'meaning': 'Engineering verification; synthetic scores are not business evidence'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk-dir', required=True)
    args = parser.parse_args()
    root = Path(args.risk_dir)
    result = require_catboost(json.loads((root / 'risk_report.json').read_text()), root)
    (root / 'catboost_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
