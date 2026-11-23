"""
Factor + Alpha detector Direct Portfolio Downside Optimization
==================================================
Author: Bohua (with Claude Code & Codex)

Architecture:
  Step 1: Fetch stock returns and the five French factors
  Step 2: 52-week factor regression, retaining central90% of each stock smoothed returns
  Step 3: Rank business-peer alpha differences; global top two receive fixed 10% floors
  Step 4: Minimize raw portfolio downside risk, then minimum-cost/Hold execution
  Step 5: Report signals, factor exposures, downside risk and trade instructions

Usage:
  Edit CONFIG tickers, current holdings and model parameters, then run.
"""

import unicodedata
import io
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import linprog, minimize
from datetime import datetime, timedelta

# ╔════════════════════════════════════════════════════════════╗
# ║  CONFIG — 用户输入在前; 稳定模型默认值在后                 ║
# ╚════════════════════════════════════════════════════════════╝

# ── 日常运行前最常调整的输入 ────────────────────────────────────

# 投资标的
TICKERS = [
    "MU", "VRT", "TSM", "LRCX", "ASML", "GS", "JPM", "MS", "NVT",
    "HSBC", "C", "AMAT", "AMD", "AAPL", "BCS", "BAC", "ANET", "MRVL",
    "AMZN", "NVDA", "GOOGL", "INTC", "TSLA", "BLK",
    "WMT", "BE", "KO", "AVGO", "SHEL", "CVX",
    "ORCL", "MSFT", "BX", "HPQ", "WFC", "BRK-B"
]

# SPY/QQQ 用作 regime 和 market-risk benchmarks；可交易资产池由 TICKERS 定义。

# 当前实际持仓: 每次跑模型前按真实 long-only 仓位更新。
# 输入权重逐项原样使用；未填写的 ticker 当前权重为 0。
# 总和低于 100% 表示隐含现金，高于 100% 表示隐含负现金/融资余额。
# CURRENT_HOLDINGS = None 时，不假设任何当前组合，直接输出无摩擦下行风险目标。
# 拟成交日期T；None使用运行日期。周末/休市不自动假定成交，需指定实际交易日。
PLANNED_EXECUTION_DATE = None
_RUNTIME_INFORMATION_CUTOFF = None  # exclusive: no market observations dated T or later

CURRENT_HOLDINGS = { "MSFT": 0.07,"GOOGL": 0.05, 
                 "WMT": 0.06, "KO": 0.15, "SHEL": 0.10, "BRK-B": 0.15, "MU": 0.05
   
}

# 日常组合设定：一般只在运行前按投资期限、风险偏好或执行规则调整。
DATA_PERIOD_YEARS = 3          # 回溯年数
FACTOR_DATA_DELAY_MONTHS = 2   # 月末因子观测满两个月后可用；缺口沿用较早值。0为不延迟。
TRADING_DAYS_PER_YEAR = 252    # History sample count only; risk and alpha stay daily.
MAX_WEIGHT_SINGLE = 0.15       # 单资产最大 long-only 权重
# 非零股票仓位的硬下限：要么为 0，要么至少为该比例。该离散约束通过
# 预先定义的确定性支持集筛选实现，见优化器中的说明。
MIN_WEIGHT_IF_HELD = 0.05
# Alpha detector（原17A）手动开关：True开启；False关闭选人及其必持下限。
# 关闭时仍保留因子回归，供beta暴露报告使用；其他非零仓位下限仍为5%。
AUTO_PEER_ALPHA_SIGNALS_ENABLED = True
AUTO_ALPHA_FIXED_MIN_HOLD_WEIGHT = 0.05  # Minimum weight for selected Alpha detector assets.
ALPHA_SELECTION_LOOKBACK_WEEKS = 52  # Calendar weeks of fitted, double-smoothed daily rows.
REGRESSION_KEEP_FRACTION = 0.90  # Keep the central90% of each stock smoothed returns; trim matched X/Y rows.
FACTOR_EXPOSURE_CAPS = {}      # 可选覆盖, 例: {"Mkt-RF": 1.2}
# 单股相对理想 execution 的变动不超过 2.5 个百分点时先尝试 Hold。
# 如果 Hold 使执行后股票总权重超过 regime 上限，或破坏其他硬约束，
# 则解除必要的 Hold，并在报告中列出最终需要执行的全部变动。
REBALANCE_THRESHOLD = 0.025    # 2.5 percentage points
TRANSACTION_COST_BPS = 0       # 本次回测：单边交易成本设为 0（零摩擦敏感性情景）

# ── 稳定模型参数（通常无需调整；以下区域中间不插入代码） ───────

# Alpha detector：对最近52周平滑股票收益，按各股票分位数删去两端各5%，再做因子回归。
# 删除对应整行收益与因子，不再次平滑；下行风险和Regime历史样本保持完整。
# 默认最低仓位5%，使用trim90规则。
# 按原始日度alpha减去其他业务peer平均alpha选出每组第一，再取全局前两名。
# 入选资产最低仓位5%；其他非零仓位至少5%；Regime决定组合股票总仓位。
# 原始因子缺失时使用较早的最近可用值。
# 不限制月份；真实值存在时保留真实值，没有较早来源时不使用未来值。
FACTOR_FORWARD_FILL_ENABLED = True

# Whole-session stock-price gaps only; all endpoints must be known by each decision.
WHOLE_DAY_PRICE_LINEAR_ESTIMATION = True
PRICE_SESSION_CALENDAR = "XNYS"  # Current universe consists of US-listed stocks/ADRs.
PRICE_GAP_AUDIT_DIRECTORY = None  # None: UK_model_price_gap_audits beside this script.
_RUNTIME_PRICE_GAP_AUDIT = []
_RUNTIME_PRICE_CALENDARS = {}
_RUNTIME_FACTOR_FILL_AUDIT = []
FACTOR_MODEL_ID = "F-v5-MktRF-SMB-LTRev-RMW-CMA-WLS-peerProxy-directAlpha52W-fixed5-dailyRMS-trim90-Delay2"
AUTO_PEER_MIN_GROUP_SIZE = 3

# 外部业务 peer taxonomy 由预先确定的业务分类定义。
# 只有至少三只业务同质的池内标的才产生自动 peer-relative signal；每个 ticker
# 最多属于一个组。下方 exclusion 是资产池已审核但不适合生成同业相对信号的标的。
AUTO_FIXED_PEER_GROUPS = {
    "Semiconductors": (
        "MU", "TSM", "LRCX", "ASML", "AMAT", "AMD", "NVDA", "MRVL",
        "INTC", "AVGO",
    ),
    "DigitalInfrastructure": ("VRT", "NVT", "ANET"),
    "GlobalBanks": ("JPM", "HSBC", "C", "BCS", "BAC", "WFC"),
    "CapitalMarketsAssetManagers": ("GS", "MS", "BLK", "BX"),
    "MegaCapPlatformSoftware": ("AAPL", "AMZN", "GOOGL", "ORCL", "MSFT"),
}
AUTO_PEER_SIGNAL_EXCLUSIONS = (
    # Only two comparable integrated-energy names are in the pool; the remaining
    # names do not have two sufficiently homogeneous pool peers for a Alpha detector peer signal.
    "SHEL", "CVX", "TSLA", "WMT", "BE", "KO", "HPQ", "BRK-B",
)

# ── 数据参数 ──────────────────────────────────────────────────
FREQUENCY = "weekly_avg_daily" # "monthly", "daily", or "weekly_avg_daily"
# weekly_avg_daily: 先对每日收益计算 5 日移动平均，
#   再对第一层结果计算 5 日移动平均。
FIRST_SMOOTHING_DAYS = 5
SECOND_SMOOTHING_OBS = 5
# 价格缺失超过此阈值时列为不完整历史并告警；有真实收益的资产
# 可用同日业务 peer 收益代理缺失的日收益，不伪造收盘价。
MAX_MISSING_TRADING_DAYS = 5

# ── 模型参数 ──────────────────────────────────────────────────

DOWNSIDE_VARIANCE_RETENTION = 0.85  # Preserve the existing semivariance-quality standard.
# Daily squared-return tolerances retain the previous feasibility and zero-risk gates.
DOWNSIDE_DAILY_VARIANCE_TOLERANCE = 7.936507936507936e-9
DOWNSIDE_DAILY_VARIANCE_FLOOR = 3.968253968253968e-19
DOWNSIDE_DAILY_VARIANCE_TIE = 3.968253968253968e-15
ALPHA_PEER_SPREAD_TOLERANCE = 3.968253968253968e-15

# ── 优化约束 ──────────────────────────────────────────────────
WEIGHT_TARGET = 1.0            # 满仓时总资产权重 100%

# 组合因子暴露上限: 直接进入优化约束, 防止组合过度押注单一因子
USE_FACTOR_EXPOSURE_CAPS = False
DEFAULT_FACTOR_EXPOSURE_CAP = 1.50  # 真实经济风险的基础cap; 普通WLS标准误缓冲在此范围内预留
FACTOR_CAP_SE_Z = 0.0

# ── 因子模型 ──────────────────────────────────────────────────
# 保留的 LT 组：Mkt-RF + SMB + LT_Rev + RMW + CMA。
# 因子输入使用 Kenneth French 的原始因子收益。
FF_FACTORS = ["Mkt-RF", "SMB", "LT_Rev", "RMW", "CMA"]

# 模型使用上述五个原始 French 因子。
FACTORS_TO_USE = FF_FACTORS

# 回归观测整体等权。

# 回归使用 WLS 经典标准误；双层 rolling 平滑可能带来序列相关，
# 其显著性与因子约束缓冲可能偏乐观。
# 平滑收益用于因子回归与Alpha detector信号；下行风险直接使用原始日收益。

# ── 市场状态 / 空仓机制 ────────────────────────────────────────
# 空仓表示优化器保留的一部分风险预算。
# target exposure = 1 - cash buffer，并进入约束 sum(weights)=target exposure。
USE_REGIME_AWARE_OPTIMIZATION = True
REGIME_BENCHMARKS = ["SPY", "QQQ"]
# 最近五个交易日 SPY/QQQ 等权累计收益决定当前单一信号状态；
# 历史非重叠五日状态使用同一门槛估计 Markov 转移。
REGIME_STATES = ["Bullish", "Neutral", "Bearish"]
# Regime 使用SPY/QQQ日线数据, 每5个交易日合成一个状态观测。
REGIME_TRADING_DAYS_PER_PERIOD = 5
# 五日收益 <= -2% 为 Bearish，>= +2% 为 Bullish，其间为 Neutral。
REGIME_FIVE_DAY_THRESHOLD = 0.02
# 状态概率 = 历史 Markov 转移概率 20% + 当前市场信号概率 80%。
REGIME_MARKOV_HISTORY_WEIGHT = 0.20
REGIME_LATEST_SIGNAL_WEIGHT = 0.80

REGIME_CASH_BUFFER = {
    "Bullish": 0.00,
    "Neutral": 0.15,
    "Bearish": 0.40,
}
# Alpha detector必持下限固定为AUTO_ALPHA_FIXED_MIN_HOLD_WEIGHT。
# Alpha detector按固定下限配置入选资产。

MAX_CASH_BUFFER = 0.40
MIN_TARGET_EXPOSURE = 0.60

# --- STEP 1: 数据获取 ---

def save_price_gap_audit(records, execution_date):
    """Persist proxy provenance on runs that actually estimate missing prices."""
    if not records:
        return None
    import json
    from pathlib import Path
    folder = (Path(PRICE_GAP_AUDIT_DIRECTORY).expanduser()
              if PRICE_GAP_AUDIT_DIRECTORY else Path(__file__).resolve().parent / "UK_model_price_gap_audits")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"price_gaps_T{pd.Timestamp(execution_date):%Y%m%d}_{stamp}.json"
    payload = {"execution_date": str(pd.Timestamp(execution_date).date()),
               "calendar": PRICE_SESSION_CALENDAR, "factors": list(FACTORS_TO_USE),
               "prices_are_estimates_not_executable_quotes": True, "records": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def expected_price_sessions(start_date, as_of):
    """Independent US equity session calendar; never create weekend/holiday rows."""
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError("Price-gap checks require exchange_calendars in this Python environment") from exc
    start = pd.Timestamp(start_date).tz_localize(None).normalize()
    end = pd.Timestamp(as_of).tz_localize(None).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    key = (PRICE_SESSION_CALENDAR, start.year, end.year)
    if key not in _RUNTIME_PRICE_CALENDARS:
        _RUNTIME_PRICE_CALENDARS[key] = xcals.get_calendar(
            PRICE_SESSION_CALENDAR, start=f"{start.year - 1}-01-01", end=f"{end.year + 1}-12-31")
    return pd.DatetimeIndex(_RUNTIME_PRICE_CALENDARS[key].sessions_in_range(start, end)).tz_localize(None)


def estimate_whole_day_price_gaps(prices, tickers, as_of, context, expected_start=None):
    """Linear price estimates only on sessions where all model stocks are absent.

    Endpoints are genuine prices available by this invocation's as_of. The
    caller must invoke this separately for every historical signal date.
    Arithmetic interpolation/extrapolation uses trading-session positions.
    These prices estimate model inputs, never executable market quotes.
    """
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
    source = prices.copy().sort_index()
    source.index = pd.DatetimeIndex(source.index).tz_localize(None).normalize()
    if source.index.has_duplicates:
        raise ValueError("Price input dates must be unique")
    source = source.loc[source.index <= cutoff]
    if source.empty or source.dropna(how="all").empty:
        raise ValueError("No actual prices available by the price-gap information cutoff")
    if not WHOLE_DAY_PRICE_LINEAR_ESTIMATION:
        return source
    start = source.index.min() if expected_start is None else pd.Timestamp(expected_start)
    sessions = expected_price_sessions(start, cutoff)
    original = source.reindex(source.index.union(sessions).sort_values())
    original = original.loc[original.index <= cutoff]
    result = original.copy()
    model_names = list(tickers)
    if not set(model_names).issubset(original.columns):
        raise ValueError("Price-gap model stock columns are missing")
    missing = original.reindex(sessions)[model_names].isna().all(axis=1)
    gap_days = sessions[missing.to_numpy()]
    if len(gap_days) == 0:
        return result
    for day in gap_days:
        position = original.index.get_loc(day)
        for ticker in original.columns:
            if pd.notna(original.at[day, ticker]):
                continue  # Preserve every actually observed stock/benchmark price.
            valid = original[ticker].dropna()
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if len(valid) < 2:
                raise ValueError(f"Cannot estimate {ticker} on {day.date()}: fewer than two genuine prices by {cutoff.date()}")
            split = valid.index.searchsorted(day)
            if 0 < split < len(valid):
                points = valid.iloc[split - 1:split + 1]
                method = "linear_interpolation"
            elif split == len(valid):
                points = valid.iloc[-2:]
                method = "trailing_linear_extrapolation"
            else:
                points = valid.iloc[:2]
                method = "leading_linear_extrapolation"
                # A later IPO is an asset-specific missing history, not a
                # market-wide download gap. Leave that to existing peer rules.
                next_non_gap = sessions[(sessions > day) & ~missing.to_numpy()]
                if len(next_non_gap) and points.index[0] > next_non_gap[0]:
                    continue
            left, right = map(pd.Timestamp, points.index)
            if right > cutoff:
                raise RuntimeError("Price estimate used a future decision-time endpoint")
            x0, x1 = original.index.get_indexer([left, right])
            fraction = (position - x0) / (x1 - x0)
            estimate = float(points.iloc[0] + fraction * (points.iloc[1] - points.iloc[0]))
            if not np.isfinite(estimate) or estimate <= 0:
                raise ValueError(f"Linear price estimate is nonpositive/nonfinite for {ticker} on {day.date()}")
            result.at[day, ticker] = estimate
            _RUNTIME_PRICE_GAP_AUDIT.append({
                "context": context, "as_of": str(cutoff.date()), "ticker": ticker,
                "target_date": str(day.date()), "method": method,
                "left_date": str(left.date()), "left_price": float(points.iloc[0]),
                "right_date": str(right.date()), "right_price": float(points.iloc[1]),
                "fraction": float(fraction), "estimated_price": estimate,
                "proxy": True, "executable_quote": False,
            })
    if result.reindex(gap_days)[model_names].isna().all(axis=1).any():
        raise ValueError("Whole-session stock prices remain unavailable after causal linear estimation")
    return result


def classify_ticker_history(prices, tickers, model_start_date, allow_peer_proxy=False):
    """Audit price coverage; a peer-proxy caller may retain incomplete assets."""
    model_prices = prices.loc[prices.index >= model_start_date, tickers]
    reference_ticker = "SPY" if "SPY" in model_prices.columns else max(
        tickers, key=lambda ticker: model_prices[ticker].notna().sum()
    )
    expected_sessions = int(model_prices[reference_ticker].notna().sum())
    if expected_sessions == 0:
        raise ValueError("Cannot determine benchmark trading-day count after the model start date")

    by_ticker = {}
    incomplete_tickers = []
    for ticker in tickers:
        observed_sessions = int(model_prices[ticker].notna().sum())
        missing_sessions = max(0, expected_sessions - observed_sessions)
        is_full_history = missing_sessions <= MAX_MISSING_TRADING_DAYS
        if not is_full_history:
            incomplete_tickers.append(f"{ticker} ({observed_sessions}/{expected_sessions} sessions)")
        by_ticker[ticker] = {
            "observed_sessions": observed_sessions,
            "expected_sessions": expected_sessions,
            "missing_sessions": missing_sessions,
            "is_full_history": is_full_history,
        }

    if incomplete_tickers and not allow_peer_proxy:
        raise ValueError(
            "Incomplete price history after the model start date; remove these "
            "tickers from TICKERS and retry: " + ", ".join(incomplete_tickers)
        )

    return {
        "reference_ticker": reference_ticker,
        "expected_sessions": expected_sessions,
        "by_ticker": by_ticker,
        "incomplete_tickers": incomplete_tickers,
        "active_tickers": list(tickers),
        "model_start_date": model_start_date,
    }

def impute_peer_daily_returns(raw_daily_returns, tickers):
    """Replace missing asset returns with same-date, *observed* business-peer means.

    Peers are read only from the untouched input, so a proxy never uses another
    proxy or any future return. Missing returns without an observed peer remain
    missing; benchmark returns and actual prices are never fabricated.
    """
    source = raw_daily_returns.copy()
    absent = [ticker for ticker in tickers if ticker not in source]
    if absent:
        raise ValueError(f"Missing asset return columns: {absent}")
    source.loc[:, list(tickers)] = source[list(tickers)].replace([np.inf, -np.inf], np.nan)
    completed = source.copy()
    diagnostics = {}
    first_market_return = source.dropna(how="all").index.min()
    groups = {
        ticker: tuple(peer for peer in members if peer != ticker)
        for members in AUTO_FIXED_PEER_GROUPS.values()
        for ticker in members
    }
    for ticker in tickers:
        peers = [peer for peer in groups.get(ticker, ()) if peer in source]
        original = source[ticker]
        missing = original.isna()
        # An asset with no genuine observation must not acquire a wholly
        # synthetic track record and become eligible for trading.
        if original.notna().sum() == 0 or not peers:
            filled = pd.Series(False, index=source.index)
        else:
            proxy = source[peers].mean(axis=1, skipna=True)
            filled = missing & proxy.notna()
            completed.loc[filled, ticker] = proxy.loc[filled]
        first_actual = original.first_valid_index()
        before_first = (filled & (source.index < first_actual)).sum() if first_actual is not None else 0
        diagnostics[ticker] = {
            "peer_group": peers,
            "peer_proxy_days": int(filled.sum()),
            "prehistory_proxy_days": int(before_first),
            "unresolved_days": int(completed.loc[
                completed.index >= first_market_return, ticker].isna().sum())
            if pd.notna(first_market_return) else int(missing.sum()),
            "first_actual_return": (str(pd.Timestamp(first_actual).date())
                                    if first_actual is not None else None),
        }
    return completed, diagnostics

def _transform_return_series(daily_returns, frequency):
    """Transform one asset's observed or disclosed peer-proxy daily returns."""
    clean = daily_returns.dropna()
    if frequency == "monthly":
        return clean.resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
    if frequency == "weekly_avg_daily":
        return _rolling_double_smoothed_average(clean)
    if frequency == "daily":
        return clean
    raise ValueError("FREQUENCY must be monthly, daily, or weekly_avg_daily")



def smoothing_horizon_days():
    """Return trading days covered by each double-rolling smoothed observation."""
    return FIRST_SMOOTHING_DAYS + SECOND_SMOOTHING_OBS - 1

def _rolling_double_smoothed_average(daily_returns):
    """
    Smooth daily returns with two successive rolling averages.
    Layer one averages FIRST_SMOOTHING_DAYS trading days; layer two averages
    SECOND_SMOOTHING_OBS consecutive first-layer observations.
    """
    if daily_returns.empty:
        return daily_returns.iloc[0:0].copy()

    first_days = FIRST_SMOOTHING_DAYS
    second_obs = SECOND_SMOOTHING_OBS
    horizon_days = smoothing_horizon_days()
    if len(daily_returns) < horizon_days:
        return daily_returns.iloc[0:0].copy()

    first_smoothed = daily_returns.rolling(first_days).mean().dropna()
    second_smoothed = first_smoothed.rolling(second_obs).mean().dropna()
    if second_smoothed.empty:
        return second_smoothed

    return second_smoothed.dropna()

def fetch_stock_data(tickers, years, frequency, include_raw_daily=False,
                     include_history_info=False, history_tickers=None):
    """Fetch stock and benchmark returns, with history eligibility for stocks only."""
    end_date = (pd.Timestamp(_RUNTIME_INFORMATION_CUTOFF).to_pydatetime()
                if _RUNTIME_INFORMATION_CUTOFF is not None else datetime.now())
    # 额外 180 天仅供 SPY/QQQ 的 benchmark 与 regime 构造复用；
    # 模型收益仍截回原有 90 天缓冲起点，避免扩大回归样本期。
    model_start_date = end_date - timedelta(days=years * 365 + 90)
    download_start_date = end_date - timedelta(days=years * 365 + 180)

    print("  Downloading price data...")
    prices = yf.download(tickers, start=download_start_date, end=end_date,
                         auto_adjust=True, progress=False)["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.loc[prices.index < pd.Timestamp(end_date), tickers]
    if prices.dropna(how="all").empty:
        raise RuntimeError("No price observations strictly before the execution date")
    eligible_tickers = list(tickers if history_tickers is None else history_tickers)
    actual_prices = prices.copy()
    prices = estimate_whole_day_price_gaps(
        actual_prices, eligible_tickers, pd.Timestamp(end_date) - pd.Timedelta(days=1),
        "current_regression_and_risk", expected_start=download_start_date)
    history_info = classify_ticker_history(prices, eligible_tickers, model_start_date, allow_peer_proxy=True)

    raw_daily_returns = prices.pct_change(fill_method=None)
    raw_daily_returns = raw_daily_returns.loc[raw_daily_returns.index >= model_start_date]
    completed_daily_returns, peer_info = impute_peer_daily_returns(raw_daily_returns, eligible_tickers)
    history_info["peer_imputation"] = peer_info
    proxied = {ticker: row for ticker, row in peer_info.items() if row["peer_proxy_days"]}
    unresolved = {ticker: row["unresolved_days"] for ticker, row in peer_info.items()
                  if row["unresolved_days"]}
    if proxied or history_info["incomplete_tickers"]:
        print("  ⚠ Incomplete asset price history: same-date business-peer "
              "returns used where available; these are proxies, not actual returns.")
        for ticker, row in proxied.items():
            print(f"    {ticker}: proxy={row['peer_proxy_days']} days, "
                  f"before first actual return={row['prehistory_proxy_days']} days, "
                  f"unresolved={row['unresolved_days']} days")
    if unresolved:
        print(f"  ⚠ Remaining missing asset returns: {unresolved}; "
              "calculations use only their available common dates.")
    latest = prices.index.max()
    unpriced_now = [ticker for ticker in eligible_tickers
                    if pd.isna(actual_prices.reindex(prices.index).loc[latest, ticker])]
    history_info["missing_latest_prices"] = unpriced_now
    if unpriced_now:
        print(f"  ⚠ No actual latest close for {unpriced_now} on "
              f"{pd.Timestamp(latest).date()}; any suggested trade in these "
              "assets is not executable until a current price is confirmed.")
    transformed = {
        ticker: _transform_return_series(completed_daily_returns[ticker], frequency)
        for ticker in tickers
    }
    returns = pd.DataFrame(transformed).loc[model_start_date:]

    if include_raw_daily and include_history_info:
        return returns, raw_daily_returns, history_info
    if include_raw_daily:
        return returns, raw_daily_returns
    if include_history_info:
        return returns, history_info
    return returns

def fetch_regime_benchmark_returns(raw_daily_returns):
    """Build five-session benchmark observations for regime classification."""
    benchmark_tickers = sorted(set(REGIME_BENCHMARKS))
    missing = [t for t in benchmark_tickers if t not in raw_daily_returns]
    if missing:
        raise ValueError(f"Raw daily returns are missing regime benchmarks: {missing}")
    daily_returns = raw_daily_returns[benchmark_tickers].dropna(how="any")
    rows, idx = [], []
    period = REGIME_TRADING_DAYS_PER_PERIOD
    starts = list(range(len(daily_returns) - period, -1, -period))[::-1]
    for start in starts:
        block = daily_returns.iloc[start:start + period].dropna(how="all")
        if len(block) < period:
            continue
        rows.append(((1.0 + block[benchmark_tickers]).prod() - 1.0).to_dict())
        idx.append(block.index[-1])
    return pd.DataFrame(rows, index=idx).dropna(how="all")

def _read_ff_csv(url, date_format="%Y%m", date_len=6):
    """Download a ZIP from Kenneth French's site and parse its CSV."""
    import zipfile
    import requests

    # Prefer HTTPS; the official host can return 404 for HTTPS while its
    # identical HTTP ZIP path remains available. Restrict fallback to that host.
    response = requests.get(url, timeout=(10, 30))
    official_prefix = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    used_http_fallback = response.status_code == 404 and url.startswith(official_prefix)
    if used_http_fallback:
        response = requests.get("http://" + url[len("https://"):], timeout=(10, 30))
    response.raise_for_status()
    archive = io.BytesIO(response.content)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Factor download did not return a valid ZIP archive: {response.url}")
    if used_http_fallback:
        print("  Factor download: official HTTP fallback used after HTTPS 404 "
              "(unencrypted transport).")
    with zipfile.ZipFile(archive) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.open(csv_name).read().decode("utf-8")

    lines = raw.split("\n")
    data_start = None
    for i, line in enumerate(lines):
        first = line.strip().split(",")[0].strip()
        if len(first) == date_len and first.isdigit():
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Cannot find dated observations in {url}")

    header_idx = data_start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1

    # 从表头开始读, 遇到非数据行停止
    data_text = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split(",")[0].strip()
        # 表头行或日期数据行
        if not data_text or (len(first) == date_len and first.isdigit()):
            data_text.append(stripped)
        elif data_text and not first.isdigit():
            break  # 遇到年度数据段, 停止

    df = pd.read_csv(io.StringIO("\n".join(data_text)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format=date_format)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def fetch_daily_factor_inputs(start_date, end_date, verbose=True):
    """Download raw daily factor inputs once for signal and risk factor frames."""
    if verbose:
        print("  Loading factor data...")
    ff5_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    lt_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
              "ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip")
    ff5_data = _read_ff_csv(ff5_url, date_format="%Y%m%d", date_len=8)
    lt_data = _read_ff_csv(lt_url, date_format="%Y%m%d", date_len=8)
    if lt_data.shape[1] != 1:
        raise RuntimeError("Unexpected Kenneth French LT_Rev daily factor layout")
    lt_data.columns = ["LT_Rev"]
    daily_factors = ff5_data.join(lt_data, how="inner") / 100.0
    if daily_factors.empty or daily_factors.index.has_duplicates:
        raise RuntimeError("Kenneth French FF5/LT_Rev daily factors failed alignment")
    latest_factor_date = pd.Timestamp(daily_factors.index.max()).normalize()
    if latest_factor_date < pd.Timestamp(end_date).normalize():
        print(f"  ⚠ Factor data delay: French FF5/LT_Rev latest "
              f"{latest_factor_date.date()}; using the latest common date.")

    return daily_factors[FF_FACTORS + ["RF"]]

def factor_information_cutoff(as_of):
    """Latest eligible factor month-end under the configured release lag."""
    date = pd.Timestamp(as_of).normalize()
    lag = FACTOR_DATA_DELAY_MONTHS
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0:
        raise ValueError("FACTOR_DATA_DELAY_MONTHS must be a nonnegative integer")
    if lag == 0:
        return date
    months = pd.period_range(date - pd.DateOffset(months=lag + 5), date, freq="M")
    eligible = [month.to_timestamp(how="end").normalize() for month in months
                if month.to_timestamp(how="end").normalize()
                + pd.DateOffset(months=lag) <= date]
    return eligible[-1]


def fill_missing_factor_inputs(factors, trading_dates, as_of, context):
    """Fill missing raw factors at requested dates, preserving causal source provenance."""
    if factors.index.has_duplicates:
        raise ValueError("Factor input dates must be unique")
    eligible = factors.sort_index().loc[:pd.Timestamp(as_of)].copy()
    if not FACTOR_FORWARD_FILL_ENABLED or trading_dates is None:
        return eligible
    dates = pd.DatetimeIndex(trading_dates).unique().sort_values()
    allowed = dates[dates <= pd.Timestamp(as_of)]
    if allowed.empty:
        return eligible
    index = eligible.index.union(allowed).sort_values()
    result = eligible.reindex(index)
    carried = result.ffill()
    for name in result.columns:
        missing = result.loc[allowed, name].isna()
        for day in allowed[missing.to_numpy()]:
            value = carried.at[day, name]
            if pd.isna(value):
                continue  # No earlier source: never backfill from the future.
            history = eligible.loc[:day, name].dropna()
            source = pd.Timestamp(history.index[-1])
            if source > day or source > pd.Timestamp(as_of):
                raise RuntimeError("Factor proxy source violates its causal boundary")
            result.at[day, name] = float(value)
            _RUNTIME_FACTOR_FILL_AUDIT.append({
                "context": context, "as_of": str(pd.Timestamp(as_of).date()),
                "factor": name, "target_date": str(day.date()),
                "source_date": str(source.date()),
                "source_age_calendar_days": int((day - source).days),
                "value": float(value), "proxy": True,
            })
    return result

def fetch_ff_factors(start_date, end_date, frequency, verbose=True, daily_inputs=None, trading_dates=None):
    """Build the five selected raw Kenneth French factor returns."""
    if frequency == "monthly":
        if verbose:
            print("  Loading factor data...")
        FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                   "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
        LT_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/F-F_LT_Reversal_Factor_CSV.zip")
        ff5_data = _read_ff_csv(FF5_URL)
        lt_data = _read_ff_csv(LT_URL)
    elif frequency in ("daily", "weekly_avg_daily"):
        if daily_inputs is None:
            factors = fetch_daily_factor_inputs(start_date, end_date, verbose=verbose)
        else:
            factors = daily_inputs.copy()
    else:
        raise ValueError("FREQUENCY must be monthly, daily, or weekly_avg_daily")

    if frequency == "monthly":
        if lt_data.shape[1] != 1:
            raise RuntimeError("Unexpected Kenneth French LT_Rev monthly factor layout")
        lt_data.columns = ["LT_Rev"]
        factors = ff5_data.join(lt_data, how="inner") / 100.0

    factors = factors.loc[:factor_information_cutoff(end_date)].copy()
    if factors.empty:
        raise ValueError("No factor observations available under the configured release lag")
    if frequency in ("daily", "weekly_avg_daily"):
        factors = fill_missing_factor_inputs(
            factors, trading_dates, end_date, "regression")
    if frequency == "weekly_avg_daily":
        factors = _rolling_double_smoothed_average(factors)

    # 按日期范围过滤
    factors = factors.loc[start_date:end_date]

    return factors[FF_FACTORS + ["RF"]]

# --- STEP 2: 因子回归 ---

def _fit_asset_factor_regression(series, factors, factor_names):
    """Regress smoothed stock returns on the five factors using WLS."""
    required = list(factor_names) + (["RF"] if "RF" in factors else [])
    aligned = factors[required].join(series.rename("asset"), how="inner").dropna()
    keep = float(REGRESSION_KEEP_FRACTION)
    if not (0.0 < keep <= 1.0):
        raise ValueError("Invalid central regression retention fraction")
    if keep < 1.0:
        tail = (1.0 - keep) / 2.0
        lower, upper = aligned["asset"].quantile([tail, 1.0 - tail]).to_numpy()
        aligned = aligned.loc[aligned["asset"].between(lower, upper, inclusive="both")]
    min_observations = max(3, len(factor_names) + 2)
    if len(aligned) < min_observations:
        raise ValueError("Insufficient usable asset-return observations for WLS regression")

    y = aligned["asset"]
    X_raw = aligned[factor_names]
    X = sm.add_constant(X_raw)
    regression = sm.WLS(y, X, weights=np.ones(len(aligned), dtype=float))
    model = regression.fit()
    return {
        "alpha": float(model.params["const"]),
        "beta": model.params[factor_names].to_numpy(dtype=float),
        "beta_stderr": model.bse[factor_names].to_numpy(dtype=float),
        "alpha_pvalue": float(model.pvalues["const"]),
        "factor_pvalues": model.pvalues[factor_names].to_numpy(dtype=float),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "estimator": "WLS-classical-SE",
    }

def run_factor_regression(stock_returns, factors, factor_names):
    """Run WLS factor regressions, including disclosed peer-proxy return rows."""
    tickers = list(stock_returns.columns)

    def run_pass(active_factors, active_factor_names):
        rows = []
        regression_stats = {}
        for ticker in tickers:
            fitted = _fit_asset_factor_regression(
                stock_returns[ticker], active_factors, active_factor_names,
            )
            rows.append(fitted["beta"])
            regression_stats[ticker] = {
                "alpha": fitted["alpha"],
                "alpha_pvalue": fitted["alpha_pvalue"],
                "r_squared": fitted["r_squared"],
                "adj_r_squared": fitted["adj_r_squared"],
                "factor_betas": dict(zip(active_factor_names, fitted["beta"])),
                "factor_beta_stderr": dict(
                    zip(active_factor_names, fitted["beta_stderr"])),
                "factor_pvalues": dict(
                    zip(active_factor_names, fitted["factor_pvalues"])),
                "n_obs": fitted["n_obs"],
                "estimator": fitted["estimator"],
            }
        return np.asarray(rows, dtype=float), regression_stats

    B, regression_stats = run_pass(factors, factor_names)
    return B, regression_stats, {}, factor_names, factors


# --- STEP 3: Alpha detector 选人 ---





def alpha_selection_inputs(stock_returns, factors, as_of, lookback_weeks=None, skip_weeks=0):
    """Use a full calendar-week window, shifted backward by skip_weeks."""
    weeks = int(ALPHA_SELECTION_LOOKBACK_WEEKS if lookback_weeks is None else lookback_weeks)
    skip = int(skip_weeks)
    if weeks <= 0 or skip < 0:
        raise ValueError("Selection lookback must be positive; skip must be nonnegative")
    end = pd.Timestamp(as_of) - pd.Timedelta(weeks=skip)
    start = end - pd.Timedelta(weeks=weeks)
    stock_dates = (stock_returns.index.to_timestamp(how="end").normalize()
                   if isinstance(stock_returns.index, pd.PeriodIndex) else stock_returns.index)
    factor_dates = (factors.index.to_timestamp(how="end").normalize()
                    if isinstance(factors.index, pd.PeriodIndex) else factors.index)
    selected_stock = stock_returns.loc[(stock_dates > start) & (stock_dates <= end)]
    selected_factors = factors.loc[(factor_dates > start) & (factor_dates <= end)]
    if selected_stock.empty or selected_factors.empty:
        raise ValueError("Selection regression window has no usable observations")
    return selected_stock, selected_factors, {
        "lookback_weeks": weeks, "skip_weeks": skip,
        "start_exclusive": str(start.date()), "end_inclusive": str(end.date()),
    }


def build_17a_signals(tickers, regression_stats, decision_date):
    """Build unannualized regression-intercept scores for business-peer selection."""
    if not AUTO_PEER_ALPHA_SIGNALS_ENABLED:
        return {"enabled": False, "selection_score": {}, "raw_alpha": {}}
    raw = pd.Series({ticker: regression_stats.get(ticker, {}).get("alpha", np.nan)
                     for ticker in tickers}, dtype=float).reindex(tickers)
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Alpha detector requires finite regression intercepts for active assets")
    diagnostics = {
        "enabled": True, "raw_alpha": raw.to_dict(), "selection_score": raw.to_dict(),
        "selection_rule": "raw observation-scale alpha minus other-peer mean; fixed minimum weight",
    }
    peer_members = [ticker for members in AUTO_FIXED_PEER_GROUPS.values() for ticker in members]
    duplicates = sorted({ticker for ticker in peer_members if peer_members.count(ticker) > 1})
    exclusions = list(AUTO_PEER_SIGNAL_EXCLUSIONS)
    unknown = sorted((set(peer_members) | set(exclusions)) - set(TICKERS))
    overlap = sorted(set(peer_members) & set(exclusions))
    unclassified = sorted(set(TICKERS) - set(peer_members) - set(exclusions))
    if duplicates or unknown or overlap or unclassified:
        raise RuntimeError(
            "Invalid Alpha detector peer taxonomy: "
            f"duplicates={duplicates}, unknown={unknown}, overlap={overlap}, "
            f"unclassified={unclassified}")
    return diagnostics




def _regime_benchmark_columns(returns):
    """Prefer SPY/QQQ and otherwise fall back to the first two available columns."""
    if returns is None or returns.empty:
        return []
    cols = [c for c in REGIME_BENCHMARKS if c in returns.columns]
    if not cols:
        cols = list(returns.columns[:2])
    return cols



def _regime_market_daily_returns(raw_benchmark_returns):
    """Return equal-weight daily SPY/QQQ market returns for regime paths."""
    cols = _regime_benchmark_columns(raw_benchmark_returns)
    if not cols:
        return pd.Series(dtype=float)
    return raw_benchmark_returns[cols].dropna(how="any").mean(axis=1)



def _classify_regime_state(market_daily_returns):
    """Classify the latest five-day equal-weight SPY/QQQ return."""
    market = market_daily_returns.dropna()
    if len(market) < REGIME_TRADING_DAYS_PER_PERIOD:
        return "Neutral", 0.0
    five_day_return = float((1.0 + market.iloc[-REGIME_TRADING_DAYS_PER_PERIOD:]).prod() - 1.0)
    state = ("Bearish" if five_day_return <= -REGIME_FIVE_DAY_THRESHOLD
             else "Bullish" if five_day_return >= REGIME_FIVE_DAY_THRESHOLD
             else "Neutral")
    return state, five_day_return

def _market_signal_probabilities(deterministic_state):
    """Give the current signal full probability before the unchanged Markov blend."""
    return pd.Series({state: float(state == deterministic_state) for state in REGIME_STATES})

def _regime_history_available(benchmark_returns):
    """Preserve the former metrics function's data-availability gate."""
    cols = _regime_benchmark_columns(benchmark_returns)
    return bool(cols) and not benchmark_returns[cols].dropna(how="all").mean(axis=1).empty

def _benchmark_regime_series(benchmark_returns, raw_benchmark_returns):
    """Generate the complete historical state sequence on the original dates."""
    if benchmark_returns is None or benchmark_returns.empty:
        return pd.Series(dtype=object)
    if raw_benchmark_returns is None or raw_benchmark_returns.empty:
        return pd.Series(dtype=object)
    states, idx = [], []
    for end_date in benchmark_returns.index:
        if not _regime_history_available(benchmark_returns.loc[:end_date]):
            continue
        market_history = _regime_market_daily_returns(raw_benchmark_returns.loc[:end_date])
        state, _ = _classify_regime_state(market_history)
        states.append(state)
        idx.append(end_date)
    return pd.Series(states, index=idx)

def _transition_probabilities(regime_series):
    """Estimate the Markov transition matrix P(next_state | current_state)."""
    counts = pd.DataFrame(1.0, index=REGIME_STATES, columns=REGIME_STATES)
    if regime_series is not None and len(regime_series) > 1:
        vals = list(regime_series.dropna())
        for prev, nxt in zip(vals[:-1], vals[1:]):
            if prev in counts.index and nxt in counts.columns:
                counts.loc[prev, nxt] += 1.0
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

def _blend_regime_probabilities(prior_probs, signal_probs):
    """Linearly blend historical Markov and current-signal probabilities."""
    prior = prior_probs.reindex(REGIME_STATES).fillna(1.0 / len(REGIME_STATES))
    signal = signal_probs.reindex(REGIME_STATES).fillna(1.0 / len(REGIME_STATES))
    total_weight = (REGIME_MARKOV_HISTORY_WEIGHT + REGIME_LATEST_SIGNAL_WEIGHT)
    if total_weight <= 0:
        raise ValueError("Regime blend weights must sum to a positive value")
    history_weight = REGIME_MARKOV_HISTORY_WEIGHT / total_weight
    latest_weight = REGIME_LATEST_SIGNAL_WEIGHT / total_weight
    blended = history_weight * prior + latest_weight * signal
    total = float(blended.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(1.0 / len(REGIME_STATES), index=REGIME_STATES)
    return blended / total

def _weighted_regime_value(mapping, probs, default):
    """Convert discrete regime parameters into continuous optimization parameters using posterior probabilities."""
    if probs is None or len(probs) == 0:
        return default
    value = 0.0
    for state in REGIME_STATES:
        value += float(probs.get(state, 0.0)) * float(mapping.get(state, default))
    return value

def compute_regime_context(stock_returns, benchmark_returns=None, raw_benchmark_returns=None):
    """
    Return market state and dynamic parameters that enter the optimizer:
      cash buffer / target exposure
    """
    if not USE_REGIME_AWARE_OPTIMIZATION:
        return {
            "state": "Neutral",
            "cash_buffer": 0.0,
            "base_cash_buffer": 0.0,
            "target_exposure": WEIGHT_TARGET,
            "transition_probs": None,
            "prior_probs": None,
            "signal_probs": None,
            "posterior_probs": None,
        }

    source = benchmark_returns if benchmark_returns is not None else stock_returns
    raw_source = (raw_benchmark_returns if raw_benchmark_returns is not None else source)
    regime_series = _benchmark_regime_series(source, raw_source)
    transition_probs = _transition_probabilities(regime_series)
    previous_state = (str(regime_series.iloc[-2]) if len(regime_series) >= 2 else "Neutral")

    if not _regime_history_available(source):
        deterministic_state = "Neutral"
        prior_probs = transition_probs.loc[previous_state]
        signal_probs = pd.Series(1.0 / len(REGIME_STATES), index=REGIME_STATES)
        five_day_return = 0.0
    else:
        market_returns = _regime_market_daily_returns(raw_source)
        deterministic_state, five_day_return = _classify_regime_state(market_returns)
        prior_probs = transition_probs.loc[previous_state]
        signal_probs = _market_signal_probabilities(deterministic_state)

    posterior_probs = _blend_regime_probabilities(prior_probs, signal_probs)
    current_state = str(posterior_probs.idxmax())

    cash_buffer = _weighted_regime_value(REGIME_CASH_BUFFER, posterior_probs, 0.10)
    cash_buffer = float(np.clip(cash_buffer, 0.0, MAX_CASH_BUFFER))
    target_exposure = float(max(MIN_TARGET_EXPOSURE, WEIGHT_TARGET - cash_buffer))

    return {
        "state": current_state,
        "deterministic_state": deterministic_state,
        "cash_buffer": cash_buffer,
        "base_cash_buffer": cash_buffer,
        "target_exposure": target_exposure,
        "transition_probs": transition_probs,
        "prior_probs": prior_probs,
        "signal_probs": signal_probs,
        "posterior_probs": posterior_probs,
        "five_day_market_return": five_day_return,
    }

def get_factor_exposure_cap(factor_name):
    """Return an exposure cap for one factor; None means unrestricted."""
    if not USE_FACTOR_EXPOSURE_CAPS:
        return None
    if factor_name in FACTOR_EXPOSURE_CAPS:
        return FACTOR_EXPOSURE_CAPS[factor_name]
    return DEFAULT_FACTOR_EXPOSURE_CAP

def build_factor_beta_stderr_matrix(tickers, factor_names, regression_stats):
    """Arrange conventional WLS beta standard errors in asset/factor order."""
    values = []
    for ticker in tickers:
        stats = regression_stats.get(ticker, {})
        stderr_by_factor = stats.get("factor_beta_stderr", {})
        row = []
        for factor_name in factor_names:
            stderr = stderr_by_factor.get(factor_name)
            if stderr is None or not np.isfinite(stderr) or stderr < 0:
                raise ValueError(f"{ticker}/{factor_name} is missing a valid WLS beta standard error")
            row.append(float(stderr))
        values.append(row)
    return np.asarray(values, dtype=float)

# --- STEP 5: 组合优化 ---

def _minimum_weight_rule_satisfied(weights, tolerance=1e-7):
    """Check that every nonzero equity allocation is at least the hard floor."""
    minimum = float(MIN_WEIGHT_IF_HELD)
    values = np.asarray(weights, dtype=float)
    if minimum <= tolerance:
        return True
    positive = values > tolerance
    return bool(np.all(values[positive] >= minimum - tolerance))

def mode2_downside_objective(raw_daily_returns, tickers, decision_date=None):
    """Direct causal portfolio downside loss."""
    if decision_date is None:
        raise ValueError("Portfolio downside objective requires a decision date")
    completed, proxy_info = impute_peer_daily_returns(
        raw_daily_returns.loc[:pd.Timestamp(decision_date)], tickers)
    history = completed[list(tickers)]
    history = history.dropna(how="any").tail(int(DATA_PERIOD_YEARS * TRADING_DAYS_PER_YEAR))
    if len(history) < TRADING_DAYS_PER_YEAR:
        raise RuntimeError("Portfolio downside objective needs one complete year of causal returns")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Portfolio downside sample contains non-finite returns")

    def score(weights):
        downside = np.minimum(values @ np.asarray(weights, dtype=float), 0.0)
        return float(np.sqrt(np.mean(downside * downside)))

    def jac(weights):
        downside = np.minimum(values @ np.asarray(weights, dtype=float), 0.0)
        rms = float(np.sqrt(np.mean(downside * downside)))
        if rms == 0.0:
            return np.zeros(values.shape[1], dtype=float)
        return values.T @ downside / (len(values) * rms)

    return score, jac, {
        "risk_definition": "portfolio daily downside RMS, benchmark zero; unannualized",
        "history_start": str(history.index.min().date()),
        "history_end": str(history.index.max().date()),
        "history_days": int(len(history)),
        "peer_imputation": proxy_info,
    }

def minimum_17a_hold_weight(signal_diagnostics):
    """Fixed minimum weight for selected Alpha detector assets."""
    floor = float(AUTO_ALPHA_FIXED_MIN_HOLD_WEIGHT)
    if not np.isfinite(floor) or not (MIN_WEIGHT_IF_HELD <= floor <= MAX_WEIGHT_SINGLE):
        raise ValueError("Invalid fixed Alpha detector minimum holding weight")
    return floor


def mode2_select_17a_floors(tickers, signal_diagnostics):
    """Rank each peer winner by unannualized alpha minus other-peer mean alpha."""
    auto = signal_diagnostics or {}
    alpha = (auto.get("raw_alpha") or {}) if auto.get("enabled", False) else {}
    if not auto.get("enabled", False) or not alpha:
        return {}, []
    positions = {ticker: idx for idx, ticker in enumerate(tickers)}
    candidates = []
    for group, prespecified in AUTO_FIXED_PEER_GROUPS.items():
        members = [ticker for ticker in prespecified if ticker in positions]
        if len(members) < AUTO_PEER_MIN_GROUP_SIZE:
            continue
        if any(ticker not in alpha for ticker in members):
            raise ValueError("Missing raw selection score for positive candidates")
        relative = {ticker: float(alpha[ticker] - np.mean([
            alpha[peer] for peer in members if peer != ticker])) for ticker in members}
        # Use the same peer-relative score in both rounds. On ties, keep member order.
        winner = max(members, key=relative.__getitem__)
        spread = relative[winner]
        if spread > ALPHA_PEER_SPREAD_TOLERANCE:
            candidates.append((group, winner, spread))
    candidates.sort(key=lambda row: (-row[2], row[0]))
    chosen = candidates[:2]
    floor = minimum_17a_hold_weight(signal_diagnostics)
    return ({ticker: floor for _, ticker, _ in chosen},
            [{"group": group, "ticker": ticker, "spread": spread}
             for group, ticker, spread in chosen])

def mode2_factor_inequalities(B, factor_names, B_stderr):
    vectors = []
    if not USE_FACTOR_EXPOSURE_CAPS:
        return vectors
    if B is None or factor_names is None or B_stderr is None:
        raise ValueError("Factor caps require beta and WLS-SE matrices")
    beta = np.asarray(B, dtype=float)
    stderr = np.asarray(B_stderr, dtype=float)
    if beta.shape != stderr.shape or beta.shape[1] != len(factor_names):
        raise ValueError("Factor beta and WLS-SE dimensions differ")
    if not np.isfinite(beta).all() or not np.isfinite(stderr).all():
        raise ValueError("Factor beta or WLS-SE is non-finite")
    for j, factor in enumerate(factor_names):
        cap = get_factor_exposure_cap(factor)
        if cap is None:
            continue
        uncertainty = FACTOR_CAP_SE_Z * np.maximum(stderr[:, j], 0.0)
        vectors.extend([(beta[:, j] + uncertainty, float(cap)), (-beta[:, j] + uncertainty, float(cap))])
    return vectors

def mode2_feasible(weights, tickers, floors, exposure, vectors, score=None, ceiling=None, tolerance=2e-6):
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(tickers),) or not np.isfinite(weights).all()
            or abs(weights.sum() - exposure) > tolerance):
        return False
    if (np.any(weights < -tolerance)
            or np.any(weights > MAX_WEIGHT_SINGLE + tolerance)
            or not _minimum_weight_rule_satisfied(weights, tolerance)):
        return False
    positions = {ticker: idx for idx, ticker in enumerate(tickers)}
    if any(weights[positions[ticker]] < floor - tolerance for ticker, floor in floors.items()):
        return False
    if any(float(weights @ vector) > cap + tolerance for vector, cap in vectors):
        return False
    return (score is None or ceiling is None
            or score(weights) ** 2 <= ceiling ** 2 + DOWNSIDE_DAILY_VARIANCE_TOLERANCE)

def mode2_supports(tickers, floors, reference, exposure):
    required = set(floors)
    positions = {ticker: idx for idx, ticker in enumerate(tickers)}
    order = sorted((ticker for ticker in tickers if ticker not in required),
                   key=lambda ticker: (-float(reference[positions[ticker]]),
                                       positions[ticker]))
    minimum_count = max(len(required), int(np.ceil((exposure - 1e-12) / MAX_WEIGHT_SINGLE)))
    maximum_count = min(len(tickers), int(np.floor((exposure - 1e-12) / MIN_WEIGHT_IF_HELD)))
    for count in range(minimum_count, maximum_count + 1):
        yield required | set(order[:count - len(required)])

def mode2_support_bounds(tickers, support, floors):
    return [(float(floors.get(ticker, MIN_WEIGHT_IF_HELD)),
             float(MAX_WEIGHT_SINGLE)) if ticker in support else (0.0, 0.0)
            for ticker in tickers]

def mode2_feasible_anchor(bounds, exposure, vectors):
    result = linprog(
        c=np.zeros(len(bounds), dtype=float),
        A_ub=np.asarray([vector for vector, _ in vectors]) if vectors else None,
        b_ub=np.asarray([cap for _, cap in vectors]) if vectors else None,
        A_eq=np.ones((1, len(bounds))), b_eq=np.array([exposure]),
        bounds=bounds, method="highs")
    return np.asarray(result.x, dtype=float) if result.success else None

def mode2_optimize_target(raw_daily_returns, tickers, exposure, floors,
                          B, factor_names, B_stderr, decision_date, *,
                          prepared_risk=None, prepared_vectors=None):
    """Use this objective's own continuous optimum to rank 0-or-5 supports."""
    score, jac, risk_info = (prepared_risk if prepared_risk is not None else
                             mode2_downside_objective(raw_daily_returns, tickers, decision_date))
    vectors = (prepared_vectors if prepared_vectors is not None else
               mode2_factor_inequalities(B, factor_names, B_stderr))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - exposure}]
    constraints.extend({"type": "ineq", "fun": lambda w, v=v, cap=cap: cap - w @ v} for v, cap in vectors)
    continuous_bounds = [(float(floors.get(ticker, 0.0)), float(MAX_WEIGHT_SINGLE)) for ticker in tickers]
    anchor = mode2_feasible_anchor(continuous_bounds, exposure, vectors)
    if anchor is None:
        raise RuntimeError("Mandatory Alpha detector bounds are infeasible")
    continuous = minimize(score, anchor, jac=jac, method="SLSQP",
                          bounds=continuous_bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
    if not continuous.success or abs(continuous.x.sum() - exposure) > 1e-6:
        raise RuntimeError("Continuous downside optimum failed")
    ranking = np.asarray(continuous.x, dtype=float)
    best, best_score = None, np.inf
    for support in mode2_supports(tickers, floors, ranking, exposure):
        bounds = mode2_support_bounds(tickers, support, floors)
        start = mode2_feasible_anchor(bounds, exposure, vectors)
        if start is None:
            continue
        result = minimize(score, start, jac=jac, method="SLSQP", bounds=bounds,
                          constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if (result.success and mode2_feasible(result.x, tickers, floors, exposure, vectors)):
            value = score(result.x)
            if value ** 2 < best_score ** 2 - DOWNSIDE_DAILY_VARIANCE_TIE:
                best, best_score = np.asarray(result.x, dtype=float), value
    if best is None:
        raise RuntimeError("No feasible 0-or-5 target support")
    best[np.abs(best) < 1e-10] = 0.0
    return best, risk_info

def mode2_min_cost_execution(raw_daily_returns, tickers, target, current,
                             exposure, floors, B, factor_names, B_stderr,
                             rebalance_threshold, decision_date, *,
                             prepared_risk=None, prepared_vectors=None):
    """Minimize trades subject to the same direct portfolio downside objective."""
    score, _, _ = (prepared_risk if prepared_risk is not None else
                   mode2_downside_objective(raw_daily_returns, tickers, decision_date))
    vectors = (prepared_vectors if prepared_vectors is not None else
               mode2_factor_inequalities(B, factor_names, B_stderr))
    target_score = score(target)
    if target_score ** 2 <= DOWNSIDE_DAILY_VARIANCE_FLOOR:
        raise RuntimeError("Target has non-positive downside risk")
    ceiling = target_score / np.sqrt(float(DOWNSIDE_VARIANCE_RETENTION))
    if (abs(float(current.sum()) - exposure) <= rebalance_threshold
            and mode2_feasible(current, tickers, floors, exposure, vectors,
                               score, ceiling)):
        retention = min(1.0, target_score ** 2 / max(score(current) ** 2, DOWNSIDE_DAILY_VARIANCE_FLOOR))
        return current.copy(), {
            "mode": "no_rebalance", "trade_count": 0,
            "objective_retention": retention,
            "target_retention": float(DOWNSIDE_VARIANCE_RETENTION),
            "objective_label": "Daily_Portfolio_Downside_RMS"}
    n = len(tickers)
    constraints = [
        {"type": "eq", "fun": lambda x: x[:n].sum() - exposure},
        {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - current)},
        {"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - current)},
        {"type": "ineq", "fun": lambda x: ceiling - score(x[:n])},
    ]
    constraints.extend({"type": "ineq", "fun": lambda x, v=v, cap=cap: cap - x[:n] @ v} for v, cap in vectors)
    target_support = {tickers[i] for i, value in enumerate(target) if value > 1e-8}
    candidates = [target_support] + list(mode2_supports(
        tickers, floors, np.maximum(target, current), exposure))
    best, gross_best, seen = None, np.inf, set()
    for support in candidates:
        key = frozenset(support)
        if key in seen:
            continue
        seen.add(key)
        bounds = mode2_support_bounds(tickers, support, floors)
        anchor = mode2_feasible_anchor(bounds, exposure, vectors)
        if anchor is None:
            continue
        for initial in ([target] if support == target_support else [anchor]):
            x0 = np.r_[initial, np.abs(initial - current)]
            result = minimize(lambda x: float(x[n:].sum()), x0,
                              method="SLSQP", bounds=bounds + [(0.0, 1.0)] * n,
                              constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-11})
            if (result.success and mode2_feasible(
                    result.x[:n], tickers, floors, exposure, vectors,
                    score, ceiling)):
                gross = float(np.abs(result.x[:n] - current).sum())
                if gross < gross_best - 1e-10:
                    best, gross_best = np.asarray(result.x[:n], dtype=float), gross
    if best is None:
        raise RuntimeError("No feasible min-cost execution")
    best[np.abs(best) < 1e-10] = 0.0
    return best, {
        "mode": "min_cost",
        "trade_count": int(np.count_nonzero(np.abs(best - current) > 1e-8)),
        "objective_retention": min(1.0, target_score ** 2 / max(score(best) ** 2, DOWNSIDE_DAILY_VARIANCE_FLOOR)),
        "target_retention": float(DOWNSIDE_VARIANCE_RETENTION),
        "objective_label": "Daily_Portfolio_Downside_RMS"}

def mode2_hold_cap_execution(raw_daily_returns, tickers, target, ideal, current,
                             exposure, floors, B, factor_names, B_stderr,
                             decision_date, *, prepared_risk=None, prepared_vectors=None):
    """Keep small execution changes as Hold subject to the regime cap and all hard bounds."""
    vectors = (prepared_vectors if prepared_vectors is not None else
               mode2_factor_inequalities(B, factor_names, B_stderr))
    score, _, _ = (prepared_risk if prepared_risk is not None else
                   mode2_downside_objective(raw_daily_returns, tickers, decision_date))
    target_score = score(target)
    ceiling = target_score / np.sqrt(float(DOWNSIDE_VARIANCE_RETENTION))
    small = np.abs(ideal - current) <= REBALANCE_THRESHOLD + 1e-12
    frozen = set(np.flatnonzero(small))
    implemented = ideal.copy()
    implemented[list(frozen)] = current[list(frozen)]
    released_for_cap = 0
    released_for_other = 0
    while True:
        while implemented.sum() > exposure + 1e-10:
            sellers = [i for i in frozen if ideal[i] < current[i] - 1e-12]
            if not sellers:
                raise RuntimeError("No held sale can restore the regime stock cap")
            i = min(sellers, key=lambda j: (ideal[j] - current[j], j))
            frozen.remove(i)
            implemented[i] = ideal[i]
            released_for_cap += 1
        if mode2_feasible(implemented, tickers, floors,
                          float(implemented.sum()), vectors, score, ceiling):
            break
        if not frozen:
            raise RuntimeError("The full execution passed but Hold repair is infeasible")
        i = min(frozen, key=lambda j: (-abs(ideal[j] - current[j]), j))
        frozen.remove(i)
        implemented[i] = ideal[i]
        released_for_other += 1
    if implemented.sum() > exposure + 1e-8:
        raise RuntimeError("Hold execution exceeds regime stock cap")
    trade_count = int(np.count_nonzero(np.abs(implemented - current) > 1e-8))
    return implemented, {
        "mode": "hold_cap" if trade_count else "no_rebalance",
        "trade_count": trade_count,
        "small_ideal_deltas": int(small.sum()),
        "small_deltas_held": len(frozen),
        "released_for_cap": released_for_cap,
        "released_for_other": released_for_other,
        "objective_retention": min(1.0, target_score ** 2 / max(score(implemented) ** 2, DOWNSIDE_DAILY_VARIANCE_FLOOR)),
        "target_retention": float(DOWNSIDE_VARIANCE_RETENTION),
        "objective_label": "Daily_Portfolio_Downside_RMS"}

def _w(text, width):
    """Right-align text using terminal display width (CJK characters occupy two cells)."""
    dw = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text)
    return " " * max(0, width - dw) + text

def generate_report(tickers, regression_stats, B, factor_names,
                    w_optimal, downside_score, downside_jac,
                    current_weights=None, rebalance_threshold=None,
                    regime_info=None, signal_diagnostics=None,
                    objective_info=None,
                    full_target_weights=None, execution_info=None, mandatory_17a_picks=None):
    """Report selection signals, downside risk, factor exposure and trades."""
    print("\n" + "=" * 70)
    print("  FACTOR + Alpha detector PORTFOLIO DOWNSIDE REPORT")
    print("=" * 70)
    print("\n  1. Alpha detector")
    gate_open = bool((signal_diagnostics or {}).get("enabled", False))
    print(f"  Alpha detector: {'ON' if gate_open else 'OFF'}")
    if gate_open:
        selected = mandatory_17a_picks
        if selected is None:
            _, selected = mode2_select_17a_floors(tickers, signal_diagnostics)
        positions = dict(zip(tickers, map(float, w_optimal)))
        held = [(row["ticker"], positions.get(row["ticker"], 0.0)) for row in selected]
        held = [(ticker, weight) for ticker, weight in held if weight > 1e-8]
        if held:
            print(f"  {'Ticker':>8s}  {'Execution weight':>16s}")
            for ticker, weight in held:
                print(f"  {ticker:>8s}  {weight:>16.2%}")
    if regime_info is not None:
        print("\n  Market Regime / Cash Buffer:")
        print(f"    Forecast state (next 5 trading days): {regime_info['state']}")
        if regime_info.get("deterministic_state") is not None:
            print(f"    Latest 5-trading-day market-signal state: " f"{regime_info['deterministic_state']}")
        print(f"    Cash buffer: {regime_info['cash_buffer']:.1%}")
        row = regime_info.get("posterior_probs")
        if row is None:
            probs = regime_info.get("transition_probs")
            state = regime_info.get("state")
            if probs is not None and state in probs.index:
                row = probs.loc[state]
        if row is not None:
            print("    Next-5-trading-day state probabilities (Markov / recent linear blend): "
                  f"Bullish {row['Bullish']:.0%}, "
                  f"Neutral {row['Neutral']:.0%}, "
                  f"Bearish {row['Bearish']:.0%}")
        print("    Model position sizing (target stock exposure): " f"{regime_info['target_exposure']:.1%}")
    print("    Portfolio objective: direct daily portfolio downside RMS "
          "(zero benchmark)")
    # --- 2. 再平衡信号 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  2. Rebalancing Signal                                  │")
    print("└─────────────────────────────────────────────────────────┘")

    report_current_weights = (np.zeros(len(tickers), dtype=float)
                              if current_weights is None else current_weights)
    has_current_holdings = current_weights is not None
    if rebalance_threshold is None:
        rebalance_threshold = REBALANCE_THRESHOLD
    if full_target_weights is None:
        full_target_weights = w_optimal
    full_target_weights = np.asarray(full_target_weights, dtype=float)
    execution_info = execution_info or {}
    stock_cap = (float(regime_info["target_exposure"])
                 if regime_info is not None else float(WEIGHT_TARGET))
    if float(w_optimal.sum()) > stock_cap + 1e-8:
        raise RuntimeError("Execution stock weight exceeds the regime target exposure cap")

    mode = execution_info.get("mode")
    if mode == "downside_target_no_holdings":
        print("  Displaying the full downside-risk target portfolio")
    elif mode == "min_cost":
        print("  Global min-cost model target "
              f"({execution_info.get('objective_label', 'Daily_Portfolio_Downside_RMS')}): "
              f"involves {execution_info['trade_count']} assets and retains "
              f"{execution_info.get('objective_retention', float('nan')):.1%} of objective quality")
        print()
    elif mode == "hold_cap":
        print("  Execution keeps small changes as Hold when feasible; "
              "the implemented stock weight must stay within the regime cap")
        print(f"  Trades: {execution_info['trade_count']}; "
              f"Holds released for cap: {execution_info['released_for_cap']}; "
              f"for other hard constraints: {execution_info['released_for_other']}; "
              f"objective quality retained: {execution_info['objective_retention']:.1%}")
    elif mode == "objective_fallback":
        print("  Full-universe minimum-cost solver did not converge; using full objective target")
    elif mode == "no_rebalance":
        print("  Current execution weights satisfy hard constraints and do not exceed "
              "the regime stock cap; no trades are required")

    if not has_current_holdings:
        print(f"  {_w('Stock sleeve', 12)}  {_w('Downside target', 12)}")
        print("  " + "-" * 24)
        for i, ticker in enumerate(tickers):
            print(f"  {ticker:>12s}  {w_optimal[i]:>12.1%}")
    else:
        execution_delta = w_optimal - report_current_weights
        need_rebalance = np.abs(execution_delta) > 1e-8
        gross_trade_notional = float(np.abs(execution_delta).sum())
        print(f"  {_w('Ticker', 8)}  {_w('Current', 8)}  "
              f"{_w('Target', 10)}  {_w('Execution', 10)}  "
              f"{_w('Change', 8)}  {'Action':>6s}")
        print("  " + "-" * 66)
        print("  Trade instructions implement every execution-weight change; "
              "stock weight cannot exceed the regime target exposure")
        any_action = False
        for i, ticker in enumerate(tickers):
            delta_weight = execution_delta[i]
            if need_rebalance[i]:
                action = "Buy" if delta_weight > 0 else "Sell"
                any_action = True
            else:
                action = "Hold"
            print(f"  {ticker:>8s}  {report_current_weights[i]:>8.1%}  "
                  f"{full_target_weights[i]:>8.1%}  {w_optimal[i]:>8.1%}  "
                  f"{delta_weight:>+8.3%}  "
                  f"{_w(action, 6)}")

        print(f"  Post-trade stock weight: {w_optimal.sum():.3%}; "
              f"regime cap: {stock_cap:.3%}; cash: {1.0-w_optimal.sum():.3%}")
        if any_action:
            cash_change = abs(float(execution_delta.sum()))
            turnover = (gross_trade_notional + cash_change) / 2
            tc_cost = gross_trade_notional * TRANSACTION_COST_BPS / 10000
            print(f"\n  → Rebalancing recommended: turnover={turnover:.1%}, "
                  f"estimated transaction cost={tc_cost:.2%}")

    # --- 3. 组合风险特征与关键发现 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  3. Portfolio Risk & Findings                           │")
    print("└─────────────────────────────────────────────────────────┘")

    portfolio_downside_rms = downside_score(w_optimal)
    if portfolio_downside_rms > 0.0:
        risk_share = w_optimal * downside_jac(w_optimal) / portfolio_downside_rms
    else:
        risk_share = np.zeros(len(tickers), dtype=float)

    # 因子暴露
    port_factor_betas = w_optimal @ B
    factor_col_width = 11
    print(f"  {'Metric':<15s}" + "".join(f"{name:>{factor_col_width}s}" for name in factor_names))
    print("  " + "-" * (15 + factor_col_width * len(factor_names)))
    print(f"  {'Exposure':<15s}" + "".join(f"{value:>{factor_col_width}.3f}" for value in port_factor_betas))

    print("\n  Execution portfolio downside risk attribution:")
    if objective_info:
        direct = objective_info
        print("  Optimized risk: direct daily portfolio downside RMS, "
              f"{direct['history_days']} daily observations "
              f"({direct['history_start']} to {direct['history_end']})")
    print(f"  Portfolio daily downside RMS: {portfolio_downside_rms:.4%}")
    print("  Signed Euler shares of daily downside RMS; "
          "negative shares offset loss-day risk.")
    print(f"  {_w('Ticker', 8)}  {_w('Execution', 10)}  {_w('RMS share', 12)}")
    print("  " + "-" * 38)
    for idx, ticker in enumerate(tickers):
        if w_optimal[idx] > 1e-8:
            print(f"  {ticker:>8s}  {w_optimal[idx]:>10.1%}  {risk_share[idx]:>+12.1%}")
    print(f"  {'Cash':>8s}  {1.0 - w_optimal.sum():>10.1%}  {0.0:>+12.1%}")
    print(f"  {'Total':>8s}  {1.0:>10.1%}  {risk_share.sum():>+12.1%}")

    max_factor_idx = np.argmax(np.abs(port_factor_betas))
    max_factor = factor_names[max_factor_idx]
    max_beta = port_factor_betas[max_factor_idx]

    factor_labels = {
        "Mkt-RF": "Market risk", "SMB": "Size (small vs big)",
        "RMW": "Profitability quality", "CMA": "Investment style",
        "LT_Rev": "Long-term reversal"
    }

    print("\n  Key findings:")
    print(f"  • Largest factor exposure: {max_factor} "
          f"({factor_labels.get(max_factor, '')}) = {max_beta:+.3f}")

    if gate_open:
        best_alpha_ticker = max(tickers, key=lambda t: abs(regression_stats[t]["alpha"]))
        best_alpha = regression_stats[best_alpha_ticker]["alpha"]
        best_alpha_p = regression_stats[best_alpha_ticker]["alpha_pvalue"]
        sig_str = "significant" if best_alpha_p < 0.05 else "not significant"
        alpha_period = "monthly" if FREQUENCY == "monthly" else "daily"
        print(f"  • Largest {alpha_period} alpha: {best_alpha_ticker} = {best_alpha:+.4%} "
              f"(p={best_alpha_p:.3f}, {sig_str})")

    print("\n" + "=" * 70)

    return {
        "portfolio_downside_rms": portfolio_downside_rms,
        "risk_metric": "daily_portfolio_downside_rms",
        "downside_risk_share": dict(zip(tickers, map(float, risk_share))),
        "weights": dict(zip(tickers, w_optimal)),
        "factor_betas": dict(zip(factor_names, port_factor_betas)),
    }

def align_current_holdings(tickers, holdings, verbose=True):
    """
    Convert CURRENT_HOLDINGS to a ticker-aligned weight array.
    Only non-negative weights are supported; each input is preserved as entered
    and is not normalized by the total.
    """
    if holdings is None:
        if verbose:
            print("\n  (CURRENT_HOLDINGS is unset; displaying frictionless downside target directly)")
        return None

    unknown = sorted(set(holdings) - set(tickers))
    if unknown and verbose:
        print(f"\n  ⚠ CURRENT_HOLDINGS contains assets outside TICKERS; ignored: {unknown}")

    curr_w = np.array([holdings.get(t, 0.0) for t in tickers], dtype=float)
    if np.any(curr_w < 0):
        negative = [f"{ticker}={curr_w[i]:.1%}" for i, ticker in enumerate(tickers) if curr_w[i] < 0]
        raise ValueError(
            "CURRENT_HOLDINGS supports long-only non-negative weights only: "
            + ", ".join(negative)
        )
    current_total = curr_w.sum()

    if current_total < 1e-12:
        if verbose:
            print("\n  (CURRENT_HOLDINGS is empty: the current portfolio is all cash and all asset weights are zero)")
        return curr_w

    if verbose:
        implicit_cash = WEIGHT_TARGET - current_total
        print(f"\n  CURRENT_HOLDINGS total asset weight={current_total:.1%} "
              f"(implied cash={implicit_cash:+.1%})")

    over_limit = curr_w > MAX_WEIGHT_SINGLE + 1e-9
    if over_limit.any() and verbose:
        names = [f"{t}={curr_w[i]:.1%}" for i, t in enumerate(tickers) if over_limit[i]]
        print(f"  ⚠ Current holdings contain individual asset weights above "
              f"MAX_WEIGHT_SINGLE={MAX_WEIGHT_SINGLE:.0%}: "
              f"{', '.join(names)}")

    return curr_w

# --- MAIN ---

def main(execution_date=None):
    global _RUNTIME_FACTOR_FILL_AUDIT, _RUNTIME_INFORMATION_CUTOFF, _RUNTIME_PRICE_GAP_AUDIT
    requested = execution_date if execution_date is not None else PLANNED_EXECUTION_DATE
    planned = pd.Timestamp(datetime.now() if requested is None else requested).normalize()
    _RUNTIME_INFORMATION_CUTOFF = planned
    _RUNTIME_FACTOR_FILL_AUDIT = []
    _RUNTIME_PRICE_GAP_AUDIT = []
    print("=" * 70)
    print("  Factor + Alpha detector Downside Pipeline")
    print(f"  Default: {ALPHA_SELECTION_LOOKBACK_WEEKS}-week selection regression | "
          f"central {REGRESSION_KEEP_FRACTION:.0%} retained | "
          + (f"Alpha detector floor {AUTO_ALPHA_FIXED_MIN_HOLD_WEIGHT:.1%}"
             if AUTO_PEER_ALPHA_SIGNALS_ENABLED else "Alpha detector: OFF"))
    print("=" * 70)

    # --- Step 1: 数据 ---
    try:
        print(f"\n[Step 1] Downloading data...")
        data_tickers = list(dict.fromkeys(TICKERS + REGIME_BENCHMARKS))
        stock_returns, raw_daily_all_returns, history_info = fetch_stock_data(
            data_tickers, DATA_PERIOD_YEARS, FREQUENCY, include_raw_daily=True,
            include_history_info=True, history_tickers=TICKERS)
        model_tickers = history_info["active_tickers"]
        stock_returns = stock_returns[model_tickers]
        raw_daily_returns = raw_daily_all_returns[model_tickers]
        if stock_returns.dropna(how="all").empty:
            raise ValueError("No data returned")
        start_date = stock_returns.dropna(how="all").index.min()
        end_date = stock_returns.dropna(how="all").index.max()
        if pd.Timestamp(end_date) >= planned or raw_daily_returns.index.max() >= planned:
            raise RuntimeError("Signal inputs contain execution-day or later prices")
        print(f"  Information cutoff: {pd.Timestamp(end_date).date()} close; "
              f"requested execution: {planned.date()} close")
        if planned.dayofweek >= 5:
            print("  Requested execution date is a weekend: this is a preparation report; "
                  "set PLANNED_EXECUTION_DATE to an actual trading session and rerun.")
        factor_input_start = history_info["model_start_date"]
        factor_input_end = raw_daily_returns.index.max()
        daily_factor_inputs = fetch_daily_factor_inputs(factor_input_start, factor_input_end)
        factor_cutoff = factor_information_cutoff(end_date)
        daily_factor_inputs = daily_factor_inputs.loc[:factor_cutoff].copy()
        if daily_factor_inputs.empty:
            raise ValueError("No factor observations available under the configured release lag")
        print(f"  Factor timing: Delay {FACTOR_DATA_DELAY_MONTHS} months; "
              f"latest eligible observation {factor_cutoff.date()}; "
              f"latest actual observation {daily_factor_inputs.index.max().date()}")
        if FREQUENCY in ("daily", "weekly_avg_daily"):
            factors = fetch_ff_factors(
                start_date, end_date, FREQUENCY,
                verbose=False, daily_inputs=daily_factor_inputs,
                trading_dates=raw_daily_returns.index)
        else:
            factors = fetch_ff_factors(start_date, end_date, FREQUENCY)
        regime_benchmark_returns = fetch_regime_benchmark_returns(raw_daily_all_returns)

        if FREQUENCY == "monthly":
            stock_returns.index = stock_returns.index.to_period("M")
            if not isinstance(factors.index, pd.PeriodIndex):
                factors.index = factors.index.to_period("M")

    except Exception as e:
        print(f"\n  ✗ Data retrieval failed: {e}")
        print("  → Check the network connection and price data, then retry")
        sys.exit(1)

    available_factors = [f for f in FACTORS_TO_USE if f in factors.columns]
    missing = set(FACTORS_TO_USE) - set(available_factors)
    if missing:
        raise RuntimeError(f"Required LT-group signal factors are missing: {sorted(missing)}")

    raw_regime_benchmark_returns = raw_daily_all_returns[REGIME_BENCHMARKS].dropna(how="any")
    regime_info = compute_regime_context(
        stock_returns, regime_benchmark_returns, raw_regime_benchmark_returns)

    # --- Step 2: 因子回归 ---
    print("\n[Step2] Running factor regression...")
    selection_stock, selection_factors, selection_info = alpha_selection_inputs(
        stock_returns, factors, end_date)
    B, regression_stats, _, available_factors, selection_factors = \
        run_factor_regression(selection_stock, selection_factors, available_factors)
    B_stderr = None

    # --- Step 3: Alpha detector 选人信号 ---
    if AUTO_PEER_ALPHA_SIGNALS_ENABLED:
        print("\n[Step 3] Selecting business-peer alpha leaders...")
    signal_diagnostics = build_17a_signals(
        model_tickers, regression_stats, end_date)

    # --- Step 4: 优化 ---
    print("\n[Step 4] Running direct daily portfolio downside RMS optimization...")

    curr_w = align_current_holdings(model_tickers, CURRENT_HOLDINGS, verbose=False)
    mode2_floors, mode2_chosen = mode2_select_17a_floors(
        model_tickers, signal_diagnostics)
    # Same-run inputs: prepare/check once, reuse locally; never cache across runs.
    prepared_risk = mode2_downside_objective(raw_daily_returns, model_tickers, end_date)
    prepared_vectors = mode2_factor_inequalities(B, available_factors, B_stderr)
    score, jac, objective_info = prepared_risk

    # 完整下行风险目标作为执行方案的共同基准。
    w_full_target, _ = mode2_optimize_target(
        raw_daily_returns, model_tickers, regime_info["target_exposure"],
        mode2_floors, B, available_factors, B_stderr, end_date,
        prepared_risk=prepared_risk, prepared_vectors=prepared_vectors)
    if curr_w is None:
        w_optimal = w_full_target.copy()
        execution_info = {
            "mode": "downside_target_no_holdings",
            "trade_count": 0,
            "objective_retention": 1.0,
            "target_retention": DOWNSIDE_VARIANCE_RETENTION,
            "objective_label": "Daily_Portfolio_Downside_RMS",
        }
    else:
        w_ideal_execution, _ = mode2_min_cost_execution(
            raw_daily_returns, model_tickers, w_full_target, curr_w,
            regime_info["target_exposure"], mode2_floors, B,
            available_factors, B_stderr, REBALANCE_THRESHOLD, end_date,
            prepared_risk=prepared_risk, prepared_vectors=prepared_vectors)
        w_optimal, execution_info = mode2_hold_cap_execution(
            raw_daily_returns, model_tickers, w_full_target,
            w_ideal_execution, curr_w, regime_info["target_exposure"],
            mode2_floors, B, available_factors, B_stderr, end_date,
            prepared_risk=prepared_risk, prepared_vectors=prepared_vectors)

    # --- Step 5: 报告 ---
    results = generate_report(
        model_tickers, regression_stats, B, available_factors,
        w_optimal, score, jac,
        current_weights=curr_w,
        regime_info=regime_info,
        signal_diagnostics=signal_diagnostics,
        objective_info=objective_info,
        full_target_weights=w_full_target,
        execution_info=execution_info,
        mandatory_17a_picks=mode2_chosen,
    )
    results["information_timing"] = {
        "requested_execution_date": str(planned.date()),
        "latest_price_information": str(pd.Timestamp(end_date).date()),
        "latest_actual_factor_date": str(daily_factor_inputs.index.max().date()),
        "factor_delay_months": FACTOR_DATA_DELAY_MONTHS,
        "factor_information_cutoff": str(factor_cutoff.date()),
        "factor_release_rule": "month-end observation plus configured calendar months; then causal older-value fill",
        "price_rule": "strictly before T; latest available T-1 session close",
        "execution_rule": "plan for T close; this script does not execute trades",
        "first_return_rule": "next trading session after T; not realized in this live report",
        "trading_session_verified": False,
    }
    results["price_gap_audit"] = list(_RUNTIME_PRICE_GAP_AUDIT)
    results["price_gap_policy"] = {
        "enabled": WHOLE_DAY_PRICE_LINEAR_ESTIMATION,
        "calendar": PRICE_SESSION_CALENDAR,
        "scope": "sessions where every model stock price is missing",
        "method": "interior arithmetic price interpolation; endpoints linear extrapolation from two genuine prices",
        "causal_rule": "both endpoints known by each current or historical signal cutoff",
        "executable_quote": False,
    }
    results["price_gap_audit_file"] = save_price_gap_audit(_RUNTIME_PRICE_GAP_AUDIT, planned)
    if _RUNTIME_PRICE_GAP_AUDIT:
        print(f"  Estimated stock/benchmark prices: {len(_RUNTIME_PRICE_GAP_AUDIT)} records; "
              "proxy inputs only, not observed returns or executable quotes. "
              "Linear estimation can smooth or distort downside risk.")
        print(f"  Price estimate provenance saved: {results['price_gap_audit_file']}")
    results["factor_fill_audit"] = list(_RUNTIME_FACTOR_FILL_AUDIT)
    results["factor_fill_policy"] = {
        "enabled": FACTOR_FORWARD_FILL_ENABLED,
        "method": "nearest earlier available raw factor; no future backfill",
        "scope": "regression factor inputs",
    }
    if _RUNTIME_FACTOR_FILL_AUDIT:
        unique_dates = {row["target_date"] for row in _RUNTIME_FACTOR_FILL_AUDIT}
        max_age = max(row["source_age_calendar_days"] for row in _RUNTIME_FACTOR_FILL_AUDIT)
        print(f"  Factor proxies: {len(unique_dates)} distinct dates; "
              f"maximum source age {max_age} calendar days. "
              "These are proxy returns, not observed factor returns.")
    results["model_id"] = FACTOR_MODEL_ID
    results["selection_window"] = {**selection_info, "regression_retained_fraction": REGRESSION_KEEP_FRACTION}
    results["signal_diagnostics"] = signal_diagnostics
    results["mandatory_17a_picks"] = mode2_chosen
    results["mandatory_17a_floor"] = {
        "driver": "fixed_weight",
        "configured_weight": AUTO_ALPHA_FIXED_MIN_HOLD_WEIGHT,
        "selection_enabled": signal_diagnostics.get("enabled", False),
        "minimum_weight_when_selected": next(iter(mode2_floors.values()), None),
        "selection_regression_weeks": ALPHA_SELECTION_LOOKBACK_WEEKS,
    }
    results["execution_info"] = execution_info
    results["portfolio_objective_info"] = objective_info
    results["data_quality"] = {
        "price_history": history_info["by_ticker"],
        "incomplete_price_history": history_info["incomplete_tickers"],
        "latest_prices_missing": history_info["missing_latest_prices"],
        "peer_return_imputation": history_info["peer_imputation"],
    }

    return results

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("This production model has no objective flags")
    results = main()
