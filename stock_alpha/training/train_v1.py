from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from stock_alpha.features.v1_daily import build_daily_features
from stock_alpha.labels.triple_barrier import make_triple_barrier_labels
from stock_alpha.labels.ranking_label import make_ranking_labels
from stock_alpha.training.evaluation import prediction_metrics, time_split_dates
from stock_alpha.training.feature_importance import extract_feature_importance
from stock_alpha.models.v1_daily_model import V1DailyAlphaModel
from stock_alpha.models.v2_ranker_model import V2RankerModel
from stock_alpha.features.v2_intraday import build_intraday_features
from stock_alpha.models.v2_intraday_model import V2IntradayScorer
from stock_alpha.storage.cache import DataLake


@dataclass
class TrainResult:
    model_path: Path
    predictions_path: Path
    backend: str
    rows: int


@dataclass
class V1Trainer:
    lake: DataLake
    model_dir: Path | str = Path("models")

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_daily(self, codes: list[str] | None = None) -> pd.DataFrame:
        if codes:
            parts = [self.lake.read_parquet("daily", c) for c in codes]
            non_empty = [p for p in parts if not p.empty]
            df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        else:
            daily_dir = self.lake.root / "daily"
            files = list(daily_dir.glob("*.csv"))
            if not files:
                return pd.DataFrame()
            df = pd.concat([pd.read_csv(p, dtype={"code": str}) for p in files], ignore_index=True)
        # 自动清洗：去空日期 + 去重 + 排序
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if "code" in df.columns:
                df["code"] = df["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
                df = df.drop_duplicates(subset=["code", "date"], keep="last")
                df = df.sort_values(["code", "date"]).reset_index(drop=True)
            else:
                df = df.drop_duplicates(subset=["date"], keep="last")
                df = df.sort_values("date").reset_index(drop=True)
        return df

    def load_minute(self, codes: list[str] | None = None, period: str = "5") -> pd.DataFrame:
        minute_dir = self.lake.root / "minute"
        if not minute_dir.exists():
            return pd.DataFrame()
        if codes:
            files = [minute_dir / f"{str(c).zfill(6)[-6:]}_{period}m.csv" for c in codes]
        else:
            files = list(minute_dir.glob(f"*_{period}m.csv"))
        parts = [pd.read_csv(p, dtype={"code": str}) for p in files if p.exists()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def train(self, codes: list[str] | None = None, model_name: str = "v1_daily_lgb.pkl", train_end: str | None = None, valid_end: str | None = None, include_v2: bool = True, period: str = "5", model_path: str | Path | None = None, label_profit_take: float = 0.03, label_stop_loss: float = 0.02, label_horizon: int = 3, model_type: str = "ranker") -> TrainResult:
        daily = self.load_daily(codes)
        if daily.empty:
            raise RuntimeError("no daily data found; run download-daily first")
        daily["code"] = daily["code"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
        daily["date"] = pd.to_datetime(daily["date"], format="mixed")
        features = build_daily_features(daily)
        split_train_end, split_valid_end = time_split_dates(features, train_end, valid_end)
        train_features = features[features["date"] <= split_train_end]

        if model_type == "ranker":
            # Ranker 模式：使用排序标签 + LGBMRanker
            labels = make_ranking_labels(daily, horizon=label_horizon)
            train_labels = labels[labels["date"] <= split_train_end]
            model = V2RankerModel().fit(train_features, train_labels)
        else:
            # Classifier 模式：沿用原有三分类
            labels = make_triple_barrier_labels(daily, profit_take=label_profit_take, stop_loss=label_stop_loss, horizon=label_horizon)
            train_labels = labels[labels["date"] <= split_train_end]
            model = V1DailyAlphaModel().fit(train_features, train_labels)

        pred = model.predict(features)
        # 评估指标（兑容两种标签格式）
        if model_type != "ranker":
            eval_df = prediction_metrics(pred, labels)
            if not eval_df.empty:
                eval_df["train_end"] = split_train_end
                eval_df["valid_end"] = split_valid_end
                self.lake.write_parquet("evaluation", "v1_metrics", eval_df)
        if include_v2:
            minute = self.load_minute(codes, period=period)
            if not minute.empty:
                try:
                    intraday = build_intraday_features(minute)
                    scores = V2IntradayScorer().score(intraday)
                    pred = V2IntradayScorer().merge_with_v1(pred, scores)
                except Exception as e:
                    print(f"V2 merge skipped: {e}")
        pred_path = self.lake.write_parquet("predictions", "v1_latest", pred)
        model_path = Path(model_path) if model_path else self.model_dir / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {"model": model, "feature_columns": model.feature_columns, "backend": model.backend}
        joblib.dump(bundle, model_path)
        fi = extract_feature_importance(bundle)
        if not fi.empty:
            self.lake.write_parquet("evaluation", "feature_importance", fi)
        return TrainResult(model_path=model_path, predictions_path=pred_path, backend=model.backend, rows=len(pred))

    def predict_latest(self, model_path: str | Path, codes: list[str] | None = None) -> pd.DataFrame:
        daily = self.load_daily(codes)
        if daily.empty:
            raise RuntimeError("no daily data found")
        features = build_daily_features(daily)
        bundle = joblib.load(model_path)
        pred = bundle["model"].predict(features)
        latest = pred.sort_values("date").groupby("code", as_index=False).tail(1)
        self.lake.write_parquet("predictions", "v1_daily_latest", latest)
        return latest.sort_values("final_score", ascending=False)
