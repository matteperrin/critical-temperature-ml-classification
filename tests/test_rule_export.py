"""Final rules must come from the fitted evaluation model, never another fit."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import extract_elcs_rules as rules
from src.critical_temperature import run_holdout as h


class RuleExportTests(unittest.TestCase):
    def test_exact_fit_dedup_export_before_prediction_and_ranking(self):
        X = pd.DataFrame({'feature': [1, 1, 2, 3, 4, 5]})
        y = pd.Series([0, 0, 1, 0, 0, 1])
        source = X.assign(critical_temp=[76, 76, 78, 76, 76, 78])
        train, valid = np.array([0, 1, 2, 3]), np.array([4, 5])
        events = []

        class Estimator:
            def fit(self, values, labels):
                events.append('fit')
                np.testing.assert_array_equal(values, X.iloc[[0, 2, 3]])
                return self

            def export_final_rule_population(self, headerNames, className, filename, DCAL):
                events.append('export')
                self.filename = Path(filename)
                self.assertions = (list(headerNames), className, DCAL)
                pd.DataFrame({'Accuracy': [0.1, 0.9], 'Match Count': [8, 4],
                              'Fitness': [0.4, 0.8]}).to_csv(filename, index=False)

            def predict(self, values):
                events.append('predict')
                return np.array([0, 1])

        estimator = Estimator()
        config = h.model_configuration('elcs_dedup', 3, [11, 42, 73])
        with tempfile.TemporaryDirectory() as tmp, patch.object(h, 'make_model', return_value=estimator):
            dest = Path(tmp) / 'rules'
            result, rows = h.evaluate(config, X, y, source, train, valid, 'test', rules_dir=dest)
            self.assertEqual(events, ['fit', 'export', 'predict'])
            self.assertEqual(estimator.assertions, (['feature'], 'above_77k', True))
            self.assertEqual(result['training_rows_raw'], 4)
            self.assertEqual(rows.row_id.tolist(), [4, 5])
            meta = json.loads((dest / 'metadata.json').read_text())
            self.assertEqual(meta['model'], config)
            self.assertEqual(meta['training_row_ids_raw'], [0, 1, 2, 3])
            self.assertEqual(meta['training_row_ids'], [0, 2, 3])
            self.assertEqual(meta['training_rows_raw'], 4)
            self.assertEqual(meta['training_rows'], 3)
            self.assertEqual(pd.read_csv(dest / 'top_rules.csv').Accuracy.tolist(), [0.9, 0.1])
            self.assertNotIn('validationaccuracy', pd.read_csv(dest / 'rules.csv').columns)

    def test_ensemble_members_seed_config_and_top_twenty(self):
        class Member:
            def __init__(self, seed):
                self.seed = seed

            def export_final_rule_population(self, **kwargs):
                pd.DataFrame({'Accuracy': np.arange(25), 'Match Count': 1,
                              'Fitness': self.seed}).to_csv(kwargs['filename'], index=False)

        config = h.model_configuration('ensemble', 2, [73, 11, 42], library_variant='unmodified')
        model = h.Ensemble(config['parameters'])
        model.members = [Member(seed) for seed in [73, 11, 42]]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'rules'
            rules.export_rules(model, config, ['feature'], [0, 1, 2], [0, 1, 2], dest)
            meta = json.loads((dest / 'metadata.json').read_text())
            self.assertEqual(meta['model'], config)
            for seed in [73, 11, 42]:
                top = pd.read_csv(dest / f'member_{seed}' / 'top_rules.csv')
                self.assertEqual(len(top), 20)
                self.assertEqual(top.Accuracy.tolist(), list(range(24, 4, -1)))
                self.assertTrue((top.Fitness == seed).all())
            self.assertEqual([m['parameters']['random_state'] for m in meta['members']], [73, 11, 42])

    def test_final_integration_manifest_inspection_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'synthetic.csv'
            pd.DataFrame({'feature': np.repeat(np.arange(30), 4),
                          'critical_temp': np.tile([76., 76., 78., 79.], 30)}).to_csv(source, index=False)
            # Tiny real upstream fits on synthetic data only, in both isolated runtimes.
            for variant in ('unmodified', 'corrected'):
                for name in h.ELCS_MODELS:
                    with self.subTest(variant=variant, name=name):
                        output = root / f'{variant}_{name}'
                        h.run_cv(name, output, data_path=source, iterations=1,
                                 population_size=5, library_variant=variant)
                        self.assertFalse((output / 'rules').exists())
                        h.run_test(output)
                        meta = rules.inspect_experiment(output)
                        manifest = json.loads((output / 'test_complete.json').read_text())
                        self.assertIn('rules/metadata.json', manifest)
                        X, y, groups, raw = h.load_model_data(source, include_source=True)
                        dev, test, _ = h.create_splits(X, y, groups)
                        self.assertEqual(meta['training_row_ids_raw'], dev.tolist())
                        self.assertTrue(set(meta['training_row_ids']).isdisjoint(test))
                        self.assertEqual(len(meta['members']), 3 if name.startswith('ensemble') else 1)
                        self.assertEqual(meta['model']['library_variant'], variant)
                        file = output / 'rules' / next(iter(meta['files']))
                        file.write_text('changed', encoding='utf-8')
                        with self.assertRaisesRegex(ValueError, 'changed|missing'):
                            rules.inspect_experiment(output)

    def test_export_failure_does_not_publish_rules(self):
        config = h.model_configuration('elcs', 1, [11, 42, 73])
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'rules'
            with self.assertRaises(AttributeError):
                rules.export_rules(object(), config, ['feature'], [0], [0], destination)
            self.assertFalse(destination.exists())

    def test_cli_requires_existing_completed_final_artifacts(self):
        with self.assertRaises(SystemExit):
            rules.main([])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'final|Final'):
                rules.main(['--experiment', tmp])


if __name__ == '__main__':
    unittest.main()
