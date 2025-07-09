import unittest
from src.hybrid_field_mapper import ProductionReadyHybridFieldMapper
import numpy as np

class TestHybridFieldMapper(unittest.TestCase):
    def setUp(self):
        self.harness = ProductionReadyHybridFieldMapper()

    def test_generate_field(self):
        field = self.harness.generate_test_field(
            {'position': [0.5, 0.5], 'length': 0.1},
            {'position': [1.5, 0.5], 'length': 0.1},
            {'conductivity': 5.96e7}, field_size=4
        )
        self.assertEqual(field.shape, (4, 4))

    def test_quantum_correction(self):
        field = np.random.rand(4, 4)
        corrected = self.harness.apply_quantum_correction(field)
        self.assertEqual(corrected.shape, field.shape)

    def test_noise_mitigation(self):
        field = np.random.rand(4, 4)
        denoised = self.harness.apply_noise_mitigation(field)
        self.assertEqual(denoised.shape, field.shape)

if __name__ == '__main__':
    unittest.main()
