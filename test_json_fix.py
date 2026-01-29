
import unittest
import torch
import json
import numpy as np

class TestJsonCompliance(unittest.TestCase):
    def test_nan_to_num(self):
        # Create a tensor with NaNs and Infs
        output = torch.tensor([
            [1.0, float('nan'), 2.0],
            [float('inf'), 3.0, float('-inf')]
        ])
        
        # This should fail JSON serialization
        with self.assertRaises((ValueError, OverflowError)):
             json.dumps(output.tolist(), allow_nan=False)

        # Apply fix
        fixed_output = torch.nan_to_num(output, nan=0.0, posinf=None, neginf=None)
        
        # Verify values
        # nan -> 0.0
        self.assertEqual(fixed_output[0, 1].item(), 0.0)
        # inf -> large number (unspecified by standard but finite)
        self.assertTrue(torch.isfinite(fixed_output[1, 0]))
        # -inf -> small number
        self.assertTrue(torch.isfinite(fixed_output[1, 2]))
        
        # Verify JSON serialization
        try:
            json_str = json.dumps(fixed_output.tolist(), allow_nan=False)
            print("JSON serialization successful")
        except ValueError as e:
            self.fail(f"JSON serialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
