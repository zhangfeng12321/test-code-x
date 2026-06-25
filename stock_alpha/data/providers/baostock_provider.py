from __future__ import annotations

import pandas as pd

from .base import MarketDataProvider


def _bs_code(code: str) -> str:
    if code.startswith(("sh.", "sz.")):
        return code
    return ("sh." if code.startswith("6") else "sz.") + code


class BaoStockProvider(MarketDataProvider):
    """BaoStock 免费历史行情 Provider。需要安装 baostock。"""

    def __init__(self):
        try:
            import baostock as bs  # type: ignore
        except ImportError as exc:
            raise RuntimeError("BaoStockProvider requires: pip install baostock") from exc
        self.bs = bs
        self._logged_in = False

    def _login(self) -> None:
        if not self._logged_in:
            rs = self.bs.login()
            if rs.error_code != "0":
                raise RuntimeError(f"baostock login failed: {rs.error_msg}")
            self._logged_in = True

    def close(self) -> None:
        if self._logged_in:
            self.bs.logout()
            self._logged_in = False

    def get_daily_bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        self._login()
        start = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 and "-" not in start else start
        end = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 and "-" not in end else end
        adjustflag = {"hfq": "1", "qfq": "2", "": "3", None: "3"}.get(adjust, "2")
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,isST"
        rs = self.bs.query_history_k_data_plus(_bs_code(code), fields, start_date=start, end_date=end, frequency="d", adjustflag=adjustflag)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if df.empty:
            return df
        df = df.rename(columns={"turn": "turnover_rate", "pctChg": "pct_chg"})
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_chg"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["code"] = code
        return df

    def get_minute_bars(self, code: str, start: str, end: str, period: str = "5") -> pd.DataFrame:
        self._login()
        start = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 and "-" not in start else start
        end = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 and "-" not in end else end
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        rs = self.bs.query_history_k_data_plus(_bs_code(code), fields, start_date=start[:10], end_date=end[:10], frequency=period, adjustflag="3")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if df.empty:
            return df
        # time 形如 20250625093500000
        df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S", errors="coerce")
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["code"] = code
        return df[["code", "datetime", "open", "high", "low", "close", "volume", "amount"]]
