from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from .base import MarketDataProvider


def _ts() -> str:
    """返回当前时间戳，用于日志打印。"""
    return datetime.now().strftime('%H:%M:%S')


class AkShareProvider(MarketDataProvider):
    """AkShare 免费数据源 Provider。需要安装 akshare。"""

    def __init__(self, data_root: str | None = None):
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise RuntimeError("AkShareProvider requires: pip install akshare") from exc
        self.ak = ak
        self.data_root = data_root

    def get_daily_bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        df = self.ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace('-', ''), end_date=end.replace('-', ''), adjust=adjust)
        mapping = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
            "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate", "涨跌幅": "pct_chg",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        return df

    def get_minute_bars(self, code: str, start: str, end: str, period: str = "5") -> pd.DataFrame:
        df = self.ak.stock_zh_a_hist_min_em(symbol=code, start_date=start, end_date=end, period=period, adjust="")
        mapping = {"时间": "datetime", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        df["code"] = code
        return df

    def get_stock_basic(self, as_of: str | None = None) -> pd.DataFrame:
        df = self.ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "code", "name": "name"})
        # 获取精确行业分类（东方财富行业板块）
        industry_map = self._fetch_industry_mapping()
        if not industry_map.empty:
            df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            df = df.merge(industry_map, on="code", how="left")
            df["industry"] = df["industry"].fillna("未知")
        return df

    def _fetch_industry_mapping(self) -> pd.DataFrame:
        """获取股票行业分类映射表。

        优先使用 BaoStock（证监会行业分类，一次性返回全量，不限流），
        失败时回退到东方财富行业板块接口。
        """
        # 方案一：BaoStock 证监会行业分类（推荐，快速且稳定）
        mapping = self._fetch_industry_from_baostock()
        if not mapping.empty:
            return mapping

        # 方案二：东方财富行业板块（fallback，较慢且容易触发限流）
        return self._fetch_industry_from_eastmoney()

    def _fetch_industry_from_baostock(self) -> pd.DataFrame:
        """通过 BaoStock 获取证监会行业分类（一次性返回全量，不限流）。"""
        try:
            import baostock as bs
        except ImportError:
            return pd.DataFrame()

        try:
            lg = bs.login()
            if lg.error_code != '0':
                bs.logout()
                return pd.DataFrame()
            rs = bs.query_stock_industry()
            if rs.error_code != '0':
                bs.logout()
                return pd.DataFrame()
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            bs.logout()
        except Exception:
            try:
                bs.logout()
            except Exception:
                pass
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        import pandas as _pd
        df = _pd.DataFrame(data, columns=rs.fields)
        # 只保留有行业分类的记录
        df = df[df["industry"].astype(str).str.strip() != ""].copy()
        # code 格式: sh.600000 -> 600000
        df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        # 简化行业名：去掉证监会编码前缀（如 "J66货币金融服务" -> "货币金融服务"）
        df["industry"] = df["industry"].astype(str).str.replace(r"^[A-Z]\d{0,4}", "", regex=True)
        mapping = df[["code", "industry"]].drop_duplicates(subset=["code"], keep="first")
        print(f"[{_ts()}]   行业分类完成(BaoStock): 覆盖 {len(mapping)} 只股票")
        return mapping

    def _fetch_industry_from_eastmoney(self) -> pd.DataFrame:
        """通过东方财富行业板块接口构建映射表（fallback，较慢）。"""
        try:
            boards = self.ak.stock_board_industry_name_em()
        except Exception:
            return pd.DataFrame()
        if boards.empty:
            return pd.DataFrame()

        name_col = None
        for col in ["板块名称", "name"]:
            if col in boards.columns:
                name_col = col
                break
        if name_col is None:
            name_col = boards.columns[0]

        total = len(boards)
        records: list[dict] = []
        for idx, industry_name in enumerate(boards[name_col], 1):
            try:
                cons = self.ak.stock_board_industry_cons_em(symbol=industry_name)
            except Exception:
                continue
            if cons.empty:
                continue
            code_col = None
            for col in ["代码", "code"]:
                if col in cons.columns:
                    code_col = col
                    break
            if code_col is None:
                code_col = cons.columns[0]
            for code in cons[code_col]:
                records.append({"code": str(code).zfill(6)[-6:], "industry": industry_name})
            if idx % 10 == 0:
                print(f"[{_ts()}]   行业分类进度: {idx}/{total}")
            time.sleep(0.5)

        if not records:
            return pd.DataFrame()
        mapping = pd.DataFrame(records).drop_duplicates(subset=["code"], keep="first")
        print(f"[{_ts()}]   行业分类完成(东方财富): 共 {total} 个行业, 覆盖 {len(mapping)} 只股票")
        return mapping

    def get_northbound_flow(self, start: str, end: str) -> pd.DataFrame:
        """北向资金每日净流入（沪股通+深股通合计）。
        akshare >= 1.10 已移除 stock_hsgt_north_net_flow_in_em，
        改用 stock_hsgt_hist_em(symbol='北向资金') 获取历史净流入。
        """
        try:
            # 新版接口：stock_hsgt_hist_em
            df = self.ak.stock_hsgt_hist_em(symbol="北向资金")
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        # 列名映射：「日期」→ date，「当日成交净买额」→ north_net_amount
        mapping = {
            "日期": "date",
            "当日成交净买额": "north_net_amount",
            "当日净流入": "north_net_amount",   # 兼容旧列名
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "north_net_amount" not in df.columns:
            # 兜底：取第二列作为净流入
            cols = df.columns.tolist()
            if len(cols) >= 2:
                df = df.rename(columns={cols[1]: "north_net_amount"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["north_net_amount"] = pd.to_numeric(df.get("north_net_amount"), errors="coerce")
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        return df.loc[mask, ["date", "north_net_amount"]].dropna(subset=["date"]).reset_index(drop=True)

    def get_northbound_stock(self, code: str, start: str, end: str) -> pd.DataFrame:
        """个股北向资金持股数据。"""
        try:
            # akshare 个股南向/北向持股
            market = "SH" if code.startswith(("6",)) else "SZ"
            symbol = f"{code}.{market}"
            df = self.ak.stock_hsgt_individual_em(symbol=symbol)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        mapping = {"日期": "date", "持股数量": "north_hold_vol", "持股占比": "north_hold_ratio"}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.NaT
        df["code"] = code
        for c in ["north_hold_vol", "north_hold_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        out_cols = [c for c in ["date", "code", "north_hold_vol", "north_hold_ratio"] if c in df.columns]
        return df.loc[mask, out_cols].dropna(subset=["date"]).reset_index(drop=True)

    def get_financial_indicators(self, code: str) -> pd.DataFrame:
        """获取个股财务指标（季度）：ROE、净利润增速、营收增速、PE、PB等。"""
        try:
            df = self.ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        except Exception:
            try:
                # 备选：东方财富财务摘要
                df = self.ak.stock_financial_analysis_indicator(symbol=code)
            except Exception:
                return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        # 统一列名（不同接口列名不一致，做兼容映射）
        mapping = {
            "报告期": "report_date", "净资产收益率": "roe", "净资产收益率(%)": "roe",
            "净利润同比增长率": "net_profit_growth", "净利润同比增长率(%)": "net_profit_growth",
            "营业收入同比增长率": "revenue_growth", "营业收入同比增长率(%)": "revenue_growth",
            "每股收益": "eps", "基本每股收益(元)": "eps",
            "每股净资产": "bps", "每股净资产(元)": "bps",
            "资产负债率": "debt_ratio", "资产负债率(%)": "debt_ratio",
            "流动比率": "current_ratio",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["code"] = code
        for c in ["roe", "net_profit_growth", "revenue_growth", "eps", "bps", "debt_ratio", "current_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out_cols = [c for c in ["report_date", "code", "roe", "net_profit_growth", "revenue_growth", "eps", "bps", "debt_ratio", "current_ratio"] if c in df.columns]
        return df[out_cols].dropna(subset=["report_date"]).reset_index(drop=True) if out_cols else pd.DataFrame()

    def get_margin_data(self, start: str, end: str) -> pd.DataFrame:
        """获取全市场融资融券每日汇总数据。"""
        try:
            df = self.ak.stock_margin_sse(start_date=start.replace('-', ''), end_date=end.replace('-', ''))
        except Exception:
            try:
                # 备选接口
                df = self.ak.stock_margin_detail_sse(start_date=start.replace('-', ''), end_date=end.replace('-', ''))
            except Exception:
                return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        mapping = {
            "信用交易日期": "date", "日期": "date",
            "融资余额(元)": "margin_balance", "融资余额": "margin_balance",
            "融资买入额(元)": "margin_buy", "融资买入额": "margin_buy",
            "融券余量(股)": "short_volume", "融券余量": "short_volume",
            "融券余额(元)": "short_balance", "融券余额": "short_balance",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["margin_balance", "margin_buy", "short_volume", "short_balance"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out_cols = [c for c in ["date", "margin_balance", "margin_buy", "short_volume", "short_balance"] if c in df.columns]
        return df[out_cols].dropna(subset=["date"]).reset_index(drop=True) if out_cols else pd.DataFrame()

    def get_dragon_tiger_list(self, start: str, end: str) -> pd.DataFrame:
        """获取指定日期范围内的龙虎榜数据。"""
        results = []
        # 优先使用本地交易日历，避免非交易日无效 API 调用
        trade_dates = None
        try:
            from pathlib import Path
            search_roots = []
            if self.data_root:
                search_roots.append(Path(self.data_root))
            search_roots += [Path("data"), Path("data_real_2000_10pct"), Path(".")]
            for root in search_roots:
                cal_path = root / "meta" / "trade_calendar.csv"
                if cal_path.exists():
                    cal = pd.read_csv(cal_path)
                    if "date" in cal.columns and not cal.empty:
                        trade_dates = pd.to_datetime(cal["date"], errors="coerce").dropna()
                        trade_dates = trade_dates[(trade_dates >= pd.to_datetime(start)) & (trade_dates <= pd.to_datetime(end))]
                        break
        except Exception:
            pass
        date_range = trade_dates if trade_dates is not None else pd.date_range(start, end, freq="B")
        for d in date_range:
            date_str = d.strftime("%Y%m%d")
            try:
                df = self.ak.stock_lhb_detail_em(start_date=date_str, end_date=date_str)
            except Exception:
                continue
            if df.empty:
                continue
            results.append(df)
        if not results:
            return pd.DataFrame()
        raw = pd.concat(results, ignore_index=True)
        # 统一列名
        mapping = {
            "上榜日期": "date", "上榜日": "date",   # 新版列名为「上榜日」
            "代码": "code", "名称": "name",
            "上榜原因": "reason",
            "买入额": "buy_amount", "龙虎榜买入额": "buy_amount",   # 新版
            "卖出额": "sell_amount", "龙虎榜卖出额": "sell_amount", # 新版
            "净买额": "net_amount", "龙虎榜净买额": "net_amount",   # 新版
            "买方机构数": "org_buy_count", "卖方机构数": "org_sell_count",
        }
        raw = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        if "code" in raw.columns:
            raw["code"] = raw["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        for c in ["buy_amount", "sell_amount", "net_amount"]:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
        # 机构数合计
        if "org_buy_count" in raw.columns and "org_sell_count" in raw.columns:
            raw["org_count"] = pd.to_numeric(raw["org_buy_count"], errors="coerce").fillna(0) + \
                               pd.to_numeric(raw["org_sell_count"], errors="coerce").fillna(0)
        elif "org_buy_count" in raw.columns:
            raw["org_count"] = pd.to_numeric(raw["org_buy_count"], errors="coerce").fillna(0)
        else:
            raw["org_count"] = 0
        out_cols = [c for c in ["date", "code", "reason", "buy_amount", "sell_amount", "net_amount", "org_count"] if c in raw.columns]
        return raw[out_cols].dropna(subset=["date", "code"]).reset_index(drop=True)
