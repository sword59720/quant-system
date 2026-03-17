import unittest

import pandas as pd

from scripts.stock_etf.backtest_stock_production_search import compute_window_metrics as prod_search_metrics
from scripts.stock_etf.optimize_stock_drawdown_floor import compute_window_metrics as drawdown_floor_metrics
from scripts.stock_etf.optimize_stock_targets import annualized_return, max_drawdown, sharpe_ratio
from scripts.stock_etf.optimize_stock_targets import compute_window_metrics as target_opt_metrics


class TestStockEtfWindowMetricsRebase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"]),
                "strategy_nav": [1.01, 1.03, 1.02, 1.06],
                "benchmark_nav_alloc": [1.005, 1.01, 1.008, 1.02],
                "strategy_ret": [0.01, 0.01980198, -0.00970874, 0.03921569],
                "benchmark_ret_alloc": [0.005, 0.00497512, -0.0019802, 0.01190476],
            }
        )
        self.window_start = "2026-01-03"
        self.strategy_window_ret = pd.Series([-0.00970874, 0.03921569])
        self.benchmark_window_ret = pd.Series([-0.0019802, 0.01190476])
        self.strategy_nav = pd.concat([pd.Series([1.0]), (1.0 + self.strategy_window_ret).cumprod()], ignore_index=True)
        self.benchmark_nav = pd.concat([pd.Series([1.0]), (1.0 + self.benchmark_window_ret).cumprod()], ignore_index=True)

    def test_target_optimizer_rebases_window_before_scoring(self):
        out = target_opt_metrics(self.df, self.window_start)
        self.assertEqual(out["rows"], 2)
        self.assertAlmostEqual(out["strategy_annual_return"], annualized_return(self.strategy_nav, 252), places=10)
        self.assertAlmostEqual(out["benchmark_annual_return_alloc"], annualized_return(self.benchmark_nav, 252), places=10)
        self.assertAlmostEqual(out["strategy_max_drawdown"], max_drawdown(self.strategy_nav), places=10)
        self.assertAlmostEqual(out["benchmark_max_drawdown_alloc"], max_drawdown(self.benchmark_nav), places=10)
        self.assertAlmostEqual(out["strategy_sharpe"], sharpe_ratio(self.strategy_window_ret, 252), places=10)
        self.assertAlmostEqual(out["benchmark_sharpe_alloc"], sharpe_ratio(self.benchmark_window_ret, 252), places=10)

    def test_related_search_scripts_use_same_rebased_window_math(self):
        prod_out = prod_search_metrics(self.df, self.window_start)
        floor_out = drawdown_floor_metrics(self.df, self.window_start)

        self.assertAlmostEqual(prod_out["strategy_annual_return"], annualized_return(self.strategy_nav, 252), places=10)
        self.assertAlmostEqual(prod_out["benchmark_annual_return_alloc"], annualized_return(self.benchmark_nav, 252), places=10)
        self.assertAlmostEqual(floor_out["strategy_annual_return"], annualized_return(self.strategy_nav, 252), places=10)
        self.assertAlmostEqual(floor_out["strategy_max_drawdown"], max_drawdown(self.strategy_nav), places=10)
        self.assertAlmostEqual(floor_out["strategy_sharpe"], sharpe_ratio(self.strategy_window_ret, 252), places=10)


if __name__ == "__main__":
    unittest.main()
