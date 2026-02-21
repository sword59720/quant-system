import unittest

import pandas as pd

from scripts.stock_etf.backtest_stock_production_search import compute_window_metrics, is_stable, score_candidate


class TestStockEtfProductionSearch(unittest.TestCase):
    def test_compute_window_metrics(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"]),
                "strategy_nav": [1.01, 1.03, 1.02, 1.06],
                "benchmark_nav_alloc": [1.005, 1.01, 1.008, 1.02],
                "strategy_ret": [0.01, 0.01980198, -0.00970874, 0.03921569],
                "benchmark_ret_alloc": [0.005, 0.00497512, -0.0019802, 0.01190476],
            }
        )
        out = compute_window_metrics(df, None)
        self.assertGreater(out["strategy_annual_return"], out["benchmark_annual_return_alloc"])
        self.assertGreater(out["excess_annual_return_vs_alloc"], 0.0)
        self.assertLess(out["strategy_max_drawdown"], 0.0)

    def test_stable_gate_and_score(self):
        m_good = {
            "full": {
                "excess_annual_return_vs_alloc": 0.02,
                "strategy_max_drawdown": -0.119,
                "benchmark_max_drawdown_alloc": -0.30,
                "strategy_sharpe": 0.9,
                "benchmark_sharpe_alloc": 0.4,
            },
            "oos_2023": {"excess_annual_return_vs_alloc": 0.01},
            "oos_2024": {"excess_annual_return_vs_alloc": 0.03},
            "oos_2025": {"excess_annual_return_vs_alloc": 0.02},
        }
        m_bad = {
            "full": {
                "excess_annual_return_vs_alloc": -0.01,
                "strategy_max_drawdown": -0.20,
                "benchmark_max_drawdown_alloc": -0.30,
                "strategy_sharpe": 0.5,
                "benchmark_sharpe_alloc": 0.4,
            },
            "oos_2023": {"excess_annual_return_vs_alloc": 0.01},
            "oos_2024": {"excess_annual_return_vs_alloc": -0.01},
            "oos_2025": {"excess_annual_return_vs_alloc": 0.02},
        }
        self.assertTrue(is_stable(m_good))
        self.assertFalse(is_stable(m_bad))

        score_good = score_candidate(m_good, trades=40)
        score_bad = score_candidate(m_bad, trades=40)
        self.assertGreater(score_good[0], score_bad[0])
        self.assertGreater(score_good[1], score_bad[1])


if __name__ == "__main__":
    unittest.main()
