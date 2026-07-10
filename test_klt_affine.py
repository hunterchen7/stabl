import unittest

from klt_affine import effective_auto_crop_percentile


class AutoCropPercentileTests(unittest.TestCase):
    def test_shrink_forces_all_frame_coverage(self):
        self.assertEqual(effective_auto_crop_percentile(2.0, "shrink"), 0.0)

    def test_other_border_modes_preserve_requested_slack(self):
        self.assertEqual(effective_auto_crop_percentile(1.5, "constant"), 1.5)
        self.assertEqual(effective_auto_crop_percentile(1.5, "replicate"), 1.5)

    def test_invalid_percentiles_are_rejected(self):
        for value in (-0.1, 50.0, 100.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                effective_auto_crop_percentile(value, "constant")


if __name__ == "__main__":
    unittest.main()
