from quant_mcp.domain.validation import WalkForwardRequest
from quant_mcp.services.walkforward_service import WalkForwardService
from quant_mcp.settings import AppSettings
import pandas as pd


def test_walk_forward_is_chronological(tmp_path):
    settings = AppSettings(data_dir=tmp_path / "data")
    path = settings.data_dir / "features"
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"ret_1": [0.1] * 40, "signal_trend_up": [1] * 40})
    frame.to_pickle(path / "demo_features.parquet")
    service = WalkForwardService(settings)
    result = service.run_walk_forward(WalkForwardRequest(strategy_id="s1", dataset_id="demo", train_bars=20, test_bars=10))
    assert result.folds[0].train_end == result.folds[0].test_start
    assert result.folds[0].train_start == 0
