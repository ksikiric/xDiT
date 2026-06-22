import unittest

from xfuser.core.distributed.attention_schedule import GemmPrecisionSchedule


class TestGemmPrecisionScheduleFromString(unittest.TestCase):
    def test_fp8_fp4_tokens(self):
        s = GemmPrecisionSchedule.from_comma_delimited_string("fp4,FP8,fp4")
        self.assertEqual(s.total_steps, 3)
        self.assertEqual(s.use_high_precision_schedule, [False, True, False])

    def test_numeric_and_synonyms(self):
        s = GemmPrecisionSchedule.from_comma_delimited_string("1,0,LOW,HP")
        self.assertEqual(s.use_high_precision_schedule, [True, False, False, True])

    def test_whitespace_tolerated(self):
        s = GemmPrecisionSchedule.from_comma_delimited_string(" fp4 , FP8 ")
        self.assertEqual(s.use_high_precision_schedule, [False, True])

    def test_rejects_unknown_token(self):
        with self.assertRaises(ValueError) as ctx:
            GemmPrecisionSchedule.from_comma_delimited_string("fp4,banana")
        self.assertIn("banana", str(ctx.exception))

    def test_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            GemmPrecisionSchedule.from_comma_delimited_string("")

    def test_is_high_precision_bounds(self):
        s = GemmPrecisionSchedule.from_comma_delimited_string("fp8")
        self.assertTrue(s.is_high_precision(0))
        with self.assertRaises(IndexError):
            s.is_high_precision(1)


if __name__ == "__main__":
    unittest.main()
