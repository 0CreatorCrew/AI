import unittest
from experiment import rules

class NelsonTests(unittest.TestCase):
    def test_rule1(self):
        self.assertTrue(rules([97, 94], 98, 95, 101).r1_low.iloc[-1])
        self.assertFalse(rules([95], 98, 95, 101).r1_low.iloc[-1])
    def test_rule2(self):
        self.assertTrue(rules([97]*9, 98, 95, 101).r2_low.iloc[-1])
        self.assertFalse(rules([97]*8, 98, 95, 101).r2_low.iloc[-1])
    def test_rule3(self):
        self.assertTrue(rules([99,98.9,98.8,98.7,98.6,98.5],98,95,101).r3_low.iloc[-1])
        self.assertFalse(rules([99,98.9,98.8,98.8,98.6,98.5],98,95,101).r3_low.iloc[-1])

if __name__ == '__main__': unittest.main()
