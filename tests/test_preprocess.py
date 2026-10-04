"""Phase II preprocessing is an offline full-row cleaning path, not model input."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.critical_temperature import preprocess as p


class PreprocessTests(unittest.TestCase):
    def test_full_row_duplicates_invalid_rows_threshold_saved_labels_raw_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.csv'
            output = Path(tmp) / 'processed' / 'clean.csv'
            # Same feature with distinct temperatures must not be collapsed.
            data = pd.DataFrame({'feature': [1, 1, 1, 2, 3, 4, 5, 6, np.inf, -np.inf, 7, 8, np.nan],
                                 'critical_temp': [77, 77, 78, 0, -1, np.nan, np.inf,
                                                   -np.inf, 80, 80, 76, 77.01, 80]})
            data.to_csv(raw, index=False)
            original = raw.read_bytes()
            with patch.object(p, 'RAW_DATA_PATH', raw), patch.object(p, 'PROCESSED_DATA_PATH', output):
                p.preprocess_data()
            self.assertEqual(raw.read_bytes(), original)
            saved = pd.read_csv(output)
            self.assertEqual(saved.feature.tolist(), [1, 1, 7, 8])
            self.assertEqual(saved.critical_temp.tolist(), [77, 78, 76, 77.01])
            self.assertEqual(saved.above_77k.tolist(), [0, 1, 0, 1])
            self.assertTrue(np.isfinite(saved.to_numpy()).all())
            self.assertFalse(saved.duplicated().any())
            self.assertEqual(saved.columns.tolist(), ['feature', 'critical_temp', 'above_77k'])


if __name__ == '__main__':
    unittest.main()
