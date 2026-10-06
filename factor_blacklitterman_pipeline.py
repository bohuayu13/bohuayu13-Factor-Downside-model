"""
Factor Model + Black-Litterman Portfolio Optimization Pipeline
==============================================================
Author: Bohua (with Claude Code & Codex)

Architecture:
  Step 0: Fetch live market-cap data
  Step 1: Fetch stock returns + Fama-French factor data
  Step 2: Factor regression + residual diagnostics
  Step 3: Ledoit-Wolf-shrunk raw prior covariance and downside asset-risk covariance
  Step 4: Black-Litterman with mixed views (asset + factor level)
  Step 5: Editable A/B portfolio optimization → optimal weights
  Step 6: Report generation
  Step 7: Optimization stability and view-confidence diagnostics
  Step 8: Historical walk-forward benchmark comparison

Usage:
  Edit the CONFIG section below — tickers, views, parameters — then run.
"""

import argparse
import hashlib
import unicodedata
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from scipy.optimize import linprog, minimize
from datetime import datetime, timedelta
import time
import warnings
import zipfile


# ╔════════════════════════════════════════════════════════════╗
# ║  CONFIG — 用户输入在前; 稳定模型默认值在后                 ║
# ╚════════════════════════════════════════════════════════════╝

# ── 日常运行前最常调整的输入 ────────────────────────────────────

# 投资标的
TICKERS = [
    "MU", "VRT", "TSM", "LRCX", "ASML", "GS", "JPM", "MS", "NVT",
    "HSBC", "C", "AMAT", "AMD", "AAPL", "BCS", "BAC", "ANET", "MRVL",
    "AMZN", "NVDA", "GOOGL", "TER", "LITE", "INTC", "TSLA", "BLK",
    "WMT", "FDX", "BE", "SBUX", "WDC", "KO", "PEP", "AVGO", "SHEL", "CVX", "XOM",
    "ORCL", "MSFT", "BX",
]

# SPY/QQQ are regime and market-risk benchmarks only.  They are not
# investable assets in this stock-only model and are kept in REGIME_BENCHMARKS.

# 当前实际持仓: 每次跑模型前按真实 long-only 仓位更新。
# 输入权重逐项原样使用；未填写的 ticker 当前权重为 0。
# 总和低于 100% 表示隐含现金，高于 100% 表示隐含负现金/融资余额。
# CURRENT_HOLDINGS = None 时，不假设任何当前组合，直接输出无摩擦 BL Optimal。
CURRENT_HOLDINGS = {
    "MU": 0.14,
    "HSBC": 0.15,
    "AMZN": 0.14,
    "MSFT": 0.120,
    "WMT": 0.1
    
}

# 资产观点: (类型, 参数, 年化超额收益/价差, 手动置信度0-1)
# 绝对观点: ("absolute", "NVDA", 0.20, 0.7)
# 相对观点: ("relative", ("NVDA", "HSBC"), 0.08, 0.6)
ASSET_VIEWS = [


]

# 因子观点: 收益统一为年化超额收益/年化价差。
# 绝对观点: ("absolute", "HML", -0.03, 0.4)
# 相对观点: ("relative", ("SMB", "HML"), 0.05, 0.5)
FACTOR_VIEWS = [

]

# 日常组合设定：一般只在运行前按投资期限、风险偏好或执行规则调整。
DATA_PERIOD_YEARS = 3          # 回溯年数; 保留低EW权重的历史尾部, 稳定WLS/HAC beta估计
RISK_FREE_RATE = 0.04          # 用于 Sharpe、回测超额收益与 SPY 市场超额收益
MAX_WEIGHT_SINGLE = 0.15       # 单资产最大 long-only 权重
# 非零股票仓位的硬下限：要么为 0，要么至少为该比例。该离散约束通过
# 预先定义的确定性支持集筛选实现，见优化器中的说明。
MIN_WEIGHT_IF_HELD = 0.05
FACTOR_EXPOSURE_CAPS = {}      # 可选覆盖, 例: {"Mkt-RF": 1.2}
# 执行层的 no-rebalance 检查与单资产交易建议共用的最小偏离阈值。
REBALANCE_THRESHOLD = 0.05     # 5%
TRANSACTION_COST_BPS = 25      # 单边交易成本（基点，1bp = 0.01%）

# ── 手动选择优化目标（每次运行前可填写） ───────────────────────
# 一般填 "A"：Maximum Sharpe，作为主模型默认。
# 如需主动进入防守模式，手动改填 "B"：Minimum Downside Semivariance。
# B 只改变股票权重的优化 objective；它不会额外提高现金比例，也不会改动
# posterior state、BL prior π、downside-risk matrix 或交易成本规则。
# B 的静态回测实现波动/回撤更低，但 Sharpe/Sortino 低于 A；请将 B 用作
# 人工风险偏好覆盖，不要把它视为已经验证的自动收益信号。
MANUAL_OBJECTIVE_MODE = "A"   # 只填 "A" 或 "B"

# ── 稳定模型参数（通常无需调整；以下区域中间不插入代码） ───────

# 17A 自动 peer-relative alpha views。正式参数来自完成的 1Y/2Y/3Y
# walk-forward 与敏感性检验：10% alpha cap、20% fixed confidence。
# 自动校准在每一次正式运行时完全在内存中重建；不会读取、写入或留下
# calibration sidecar。每个历史信号只使用当时已有的数据，5 日结果必须
# 在当前决策日前严格成熟；若不能建立有效校准则 fail closed。
AUTO_PEER_ALPHA_VIEWS_ENABLED = True
FACTOR_MODEL_ID = "F-no-pca-v1"
AUTO_ALPHA_FORECAST_HORIZON_DAYS = 5
AUTO_ALPHA_MIN_CALIBRATION_DATES = 26
# 两年：最近 104 个非重叠、已成熟的 5-trading-day 历史预测。
AUTO_ALPHA_CALIBRATION_MAX_DATES = 104
AUTO_ALPHA_CALIBRATION_PRIOR_DATES = 26
AUTO_ALPHA_MAX_ANNUAL_TILT = 0.10
AUTO_ALPHA_WINSOR_LOWER = 0.05
AUTO_ALPHA_WINSOR_UPPER = 0.95
AUTO_ALPHA_SIGNAL_Z_CLIP = 3.0
AUTO_PEER_VIEW_CONFIDENCE = 0.20
AUTO_PEER_MIN_GROUP_SIZE = 3
AUTO_PEER_VIEWS_PER_GROUP = 1

# 仅限本次 Python 运行的内存状态。它由 main() 在建立 views 之前重建，
# 因而不存在跨运行沿用、过期或写入磁盘的 calibration state。
_RUNTIME_AUTO_ALPHA_CALIBRATION = None

# 外部业务 peer taxonomy 预先固定，不从历史收益、相关性或回测结果反推。
# 每个 ticker 最多属于一个组；活跃成员不足3只时该组不产生自动view。
AUTO_FIXED_PEER_GROUPS = {
    "Semiconductors": (
        "MU", "TSM", "LRCX", "ASML", "AMAT", "AMD", "NVDA", "MRVL",
        "TER", "INTC", "AVGO",
    ),
    "DigitalInfrastructure": ("VRT", "NVT", "ANET", "LITE", "WDC"),
    "MoneyCenterBanks": ("JPM", "HSBC", "C", "BCS", "BAC"),
    "CapitalMarketsAssetManagers": ("GS", "MS", "BLK", "BX"),
    "MegaCapPlatformSoftware": ("AAPL", "AMZN", "GOOGL", "ORCL", "MSFT"),
    "IntegratedEnergy": ("SHEL", "CVX", "XOM"),
}

# ── 数据参数 ──────────────────────────────────────────────────
FREQUENCY = "weekly_avg_daily" # "monthly", "daily", or "weekly_avg_daily"
# weekly_avg_daily: 先算每日收益, 用rolling 5日做第一次weak-law平滑;
#   再对连续5个第一次平滑结果做第二次rolling weak-law平滑。
#   小EW从第一次峰值附近自动向前延长, 直到周期内首日权重不低于周期外最近日;
#   half-life落在权重真正开始下降的那一天。
FIRST_SMOOTHING_DAYS = 5
SECOND_SMOOTHING_OBS = 5
RECENT_TAIL_EW_POWER = 1.0     # 1.0=普通小EW, 2.0=平方小EW(更偏最新)
# Every asset must have sufficiently complete price history across the model
# window. Assets with more than this number of missing trading days stop the
# run and are reported for removal from TICKERS.
MAX_MISSING_TRADING_DAYS = 5
AUTO_RISK_GROUP_CORRELATION = 0.60  # 自动共同风险组的最低 rho(total risk)

# ── 模型参数 ──────────────────────────────────────────────────

# 观点置信度自动校准: 最终置信度 = 模型统计质量+ 人工输入的置信度
MODEL_CONFIDENCE_WEIGHT = 0.70
MANUAL_CONFIDENCE_WEIGHT = 0.30
# ── 优化目标（可手动切换） ─────────────────────────────────────
# A = ``max_sharpe``：在当前 BL expected-return prior 与 downside-risk matrix 下，
#     最大化 annual expected excess return / downside semideviation。
# B = ``minimum_downside_semivariance``：固定 state-implied stock exposure 后，
#     仅最小化 Ledoit-Wolf-shrunk downside semivariance；它不使用 BL return
#     ranking 来决定股票间权重。B 不是“全现金”模式：cash 仍由 3-state mapping 决定。
#
# 默认从上方用户参数区读取；每次运行还可用 ``--objective B`` 临时覆盖，或用
# ``--choose-objective`` 在开始下载数据前输入 A/B。命令行覆盖不回写此参数。
PORTFOLIO_OBJECTIVE_MODE = MANUAL_OBJECTIVE_MODE


# B 的执行层不能沿用 A 的 Sharpe retention。此值要求执行组合的 downside
# semivariance 至少保留 full B target 的该比例：variance(executed) <=
# variance(full_target) / B_DOWNSIDE_VARIANCE_RETENTION。0.85 相当于最多
# 高 17.6% downside semivariance（约高 8.5% downside semideviation），以减少交易。
B_DOWNSIDE_VARIANCE_RETENTION = 0.85

# ── 优化约束 ──────────────────────────────────────────────────
WEIGHT_TARGET = 1.0            # 满仓时总资产权重 100%
N_OPTIM_RESTARTS = 5           # 最多使用的确定性SLSQP起点数量
RISK_CONTRIBUTION_TOP_N = 8    # 报告中显示风险贡献最高的资产

# 组合因子暴露上限: 直接进入优化约束, 防止组合过度押注单一因子
USE_FACTOR_EXPOSURE_CAPS = True
DEFAULT_FACTOR_EXPOSURE_CAP = 1.50  # 真实经济风险的基础cap; HAC beta标准误缓冲在此范围内预留, 不用为补偿而上调
FACTOR_CAP_HAC_SE_Z = 1.645  # z值: 单侧90% HAC SE(beta)保护; 通常无需调整
RESIDUAL_PAIR_WEIGHT_ALERT = 0.35   # 高残差相关资产对的合计目标权重提示线
RESIDUAL_CORRELATION_THRESHOLD = 0.35  # 报告列示与共同风险提示共用的残差相关阈值

# ── 全资产手续费执行 ────────────────────────────────────────────
# 先求无摩擦 BL-regime Optimal；若有真实持仓，再在全资产池中求
# “交易成本最小且至少保留该比例 BL Maximum Sharpe”的执行组合。
BL_SHARPE_RETENTION = 0.85

# ── 因子模型 ──────────────────────────────────────────────────
# 正式 F 因子模型（PCA 已移除）：
#   Mkt-RF + SMB + QMJ + BAB + Hardw49 + Chips49。
# 其中 QMJ/BAB 是 AQR 的每日多空因子；Hardw49/Chips49 是 Kenneth
# French 49-industry daily portfolio returns。它们不是 ETF 代理，故在
# EXTERNAL_RETURN_FACTORS 中按原始收益读取，并与 Mkt-RF/SMB 同步后
# 统一按正式残差化规则进入回归、BL 与风险协方差。
FF_FACTORS = ["Mkt-RF", "SMB"]

# 这些外部每日因子必须由可审计的本地缓存提供。缓存来源、格式、列名或
# 时效不合格时模型会 fail closed；不会静默改用 ETF 代理或旧数据。
EXTERNAL_RETURN_FACTORS = ["QMJ", "BAB", "Hardw49", "Chips49"]
EXTERNAL_FACTOR_LAYER = "Theme"
EXTERNAL_FACTOR_CACHE = Path(
    "/Users/yubohua/Documents/Codex/2026-09-26/ni/"
    "work/factor_candidate_groups_2y/source_cache"
)
EXTERNAL_FACTOR_MAX_STALENESS_DAYS = 45
EXTERNAL_FACTOR_REQUIRED_SHA256 = {
    "qmj_daily.xlsx": "6e0d68577fcd87e182c51902cd35d4dd7252d211d0f62327a87a1c7d9456de7e",
    "bab_daily_download.xlsx": "c34f1ac4130f69dd73440dd62a53caf22df47bc7cc50f5cf544f8e522c28a8c6",
    "49_Industry_Portfolios_Daily_CSV.zip": "8f394fe34bea54d41b9aafed410425ee8f8e252ede3c71a7c1cd20bab83040de",
}

# ETF 自定义因子已由正式 F 设计替换，不与 F 叠加以避免重复暴露。
# 保留该空配置以使通用因子构建器明确没有 ETF 代理因子。
# 自定义因子格式: "名称": (long, short, layer)
#
# 格式: "因子名": (long, short, layer)
#   long  = ticker 字符串        → 单资产收益
#         = [t1, t2, ...] 列表   → 多资产等权篮子 (消除个股噪声)
#   short = None                 → 原始收益 (如 IEF 作为利率代理)
#         = "Mkt"                → 减市场因子 Mkt-RF (板块超额)
#         = ticker 字符串        → 配对差价 (如 HYG-LQD 信用利差)
#   layer = "Macro" / "Sector" / "Theme"，自动决定只可对哪些低层因子残差化
#
# 加新因子只改这里, 不碰代码 (不用的因子首行加“#”即可)
CUSTOM_FACTORS = {}

# 最终因子列表 = F 的两个 FF 基础因子 + 四个外部每日因子。
FACTORS_TO_USE = (FF_FACTORS + list(CUSTOM_FACTORS.keys())
                  + EXTERNAL_RETURN_FACTORS)

# 自定义因子分层正交化: 所有自定义因子先去除FF成分；高层因子只有在
# 与低层因子重叠显著时才自动进一步残差化。同层因子不互相残差化。
# True 表示该处理直接进入回归、BL和最终权重, 不只是报告提示。
ORTHOGONALIZE_CUSTOM_FACTORS = True
ORTHOGONALIZE_AGAINST = FF_FACTORS
FACTOR_LAYER_ORDER = {"FF": 0, "Macro": 1, "Sector": 2, "Theme": 3}
AUTO_LOWER_LAYER_ORTHOGONALIZE_CORR = 0.60  # 新因子对低层因子的EW加权 |rho| 达此值则加入低层残差化
FACTOR_VIF_ALERT = 5.0                      # EW加权 VIF 提示线; 高于此值也加入低层残差化
# 报告中的重叠观察区间; 仅用于 Low / Medium / High 判断，不改变残差化门槛。
FACTOR_OVERLAP_MEDIUM_CORR = 0.40
FACTOR_OVERLAP_MEDIUM_VIF = 3.0
FACTOR_CONDITION_NUMBER_LOW_THRESHOLD = 10.0  # <10=低: 因子结构稳定，beta估计较可靠
FACTOR_CONDITION_NUMBER_ALERT = 30.0          # 10-30=中: 需观察；>=30=高: 共线性告警

# 全局大EW: 用于因子回归，以及以原始日度SPY估计 delta_market。
# delta_market 自动使用 DATA_PERIOD_YEARS 的窗口与这里的半衰期；它不使用
# 双层weak-law平滑或回归专用的小EW尾部修正。
USE_EW_REGRESSION = True
EW_REGRESSION_HALFLIFE_YEARS = 1.0
EW_REGRESSION_WEIGHT_POWER = 1.0  # 1.0=普通EW, 2.0=平方EW (halflife/2)

# HAC / Newey-West 推断: 自动覆盖双层rolling平滑造成的最长机械重叠。
# 最终观测覆盖 FIRST + SECOND - 1 个原始日, 因此最多重叠 FIRST + SECOND - 2 个滞后。
# 仅修正标准误/p-value, 不改变WLS beta或alpha。
HAC_MAX_LAGS = max(0, FIRST_SMOOTHING_DAYS + SECOND_SMOOTHING_OBS - 2)

# 风险协方差固定使用原始日收益的样本协方差估计。
# 平滑收益只用于因子/观点信号，不参与波动率与相关性计算。

# ── 历史指数基准 ──────────────────────────────────────────────
# 主模型报告不再输出 benchmark；正式1Y/2Y/3Y验证由独立walk-forward runner负责。
BACKTEST_ENABLED = False
BENCHMARK_ETFS = ["SPY", "QQQ"]

# ── 敏感度分析 ────────────────────────────────────────────────
SENSITIVITY_ENABLED = True
SENSITIVITY_DELTA = 0.15       # 置信度扰动幅度 (±15%)
# 先用基准组合附近的快速求解筛查; 权重变化达到此值时,
# 自动用完整优化复核该情景。
SENSITIVITY_RECHECK_THRESHOLD = 0.05

# ── 市场状态 / 空仓机制 ────────────────────────────────────────
# 空仓不是risk-free asset, 而是让优化器少用一部分风险预算。
# target exposure = 1 - cash buffer，并进入约束 sum(weights)=target exposure。
USE_REGIME_AWARE_OPTIMIZATION = True
REGIME_BENCHMARKS = ["SPY", "QQQ"]
# 三状态状态机：反弹但尚未收复前高的阶段归入 Neutral，
# 因此转移矩阵、信号后验和现金分配均只在 Bullish/Neutral/Bearish 上估计。
REGIME_STATES = ["Bullish", "Neutral", "Bearish"]
# Regime 使用SPY/QQQ日线数据, 每5个交易日合成一个状态观测。
REGIME_PERIODS_PER_YEAR = 52
REGIME_TRADING_DAYS_PER_PERIOD = 5
REGIME_SHORT_WINDOW_PERIODS = 4     # ≈ 20个交易日 / 1个月
REGIME_MEDIUM_WINDOW_PERIODS = 13   # ≈ 65个交易日 / 3个月
# Bearish: SPY/QQQ等权市场最近连续5个交易日累计下跌至少2%。
# 发生风险退出后、尚未收复下跌前高点的反弹阶段归入 Neutral；
# 收复前高后的增长归 Bullish。
REGIME_RISK_OFF_FIVE_DAY_RETURN = -0.02
REGIME_RECOVERY_LOOKBACK_DAYS = 10
# 状态概率 = 历史Markov转移概率与当前市场信号概率的线性混合。
REGIME_MARKOV_HISTORY_WEIGHT = 0.30
REGIME_LATEST_SIGNAL_WEIGHT = 0.70

REGIME_CASH_BUFFER = {
    "Bullish": 0.00,
    "Neutral": 0.15,
    "Bearish": 0.40,
}
MAX_CASH_BUFFER = 0.40
MIN_TARGET_EXPOSURE = 0.60

# 协方差混合始终只使用原始日收益。近期协方差捕捉当前风险结构，
# 长期协方差保留跨市场状态的稳定样本；近期部分固定为 60%。
RECENT_COVARIANCE_WINDOW_DAYS = 63
RECENT_COVARIANCE_WEIGHT = 0.60


# ============================================================
# COMMAND-LINE INPUT (kept after every configuration parameter)
# ============================================================

def parse_manual_objective_override(argv=None):
    """Read an optional operator-selected A/B objective before the model runs.

    ``--objective B`` is suitable for a scripted/manual terminal run. Passing
    ``--choose-objective`` opens one explicit prompt before any market data are
    fetched, so an operator can type ``B`` for the conservative objective.
    Omitting both flags uses ``PORTFOLIO_OBJECTIVE_MODE`` (default A).
    """
    parser = argparse.ArgumentParser(
        description="Factor-BL pipeline with a manually selectable optimizer objective"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--objective", choices=("A", "B"), default=None,
        help="A = Maximum Sharpe; B = Minimum Downside Semivariance",
    )
    group.add_argument(
        "--choose-objective", action="store_true",
        help="Prompt for A or B before downloading market data",
    )
    args = parser.parse_args(argv)
    if args.choose_objective:
        default = _normalized_portfolio_objective_mode()
        answer = input(
            "Select portfolio objective [A=Max Sharpe, B=Min Downside Semivariance] "
            f"(Enter={default}): "
        ).strip().upper()
        return default if not answer else _normalized_portfolio_objective_mode(answer)
    return _normalized_portfolio_objective_mode(args.objective)


# ============================================================
# STEP 1: 数据获取
# ============================================================

def fetch_market_caps(tickers):
    """Fetch live market-cap weights and stop on failure; use no defaults."""
    print("[Step 0] Fetching live market-cap data...")
    caps = {}
    failed = {}
    for t in tickers:
        last_error = None
        for attempt in range(3):
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.info
                mc = info.get("marketCap", 0)
                if not mc:
                    fast_info = ticker_obj.fast_info
                    if hasattr(fast_info, "get"):
                        mc = fast_info.get("market_cap", 0)
                    else:
                        mc = getattr(fast_info, "market_cap", 0)
                if not mc:
                    shares = info.get("sharesOutstanding", 0)
                    price = (info.get("regularMarketPrice", 0)
                             or info.get("previousClose", 0))
                    if shares and price:
                        mc = shares * price
                if not mc:
                    raise ValueError("marketCap is zero or unavailable")
                caps[t] = mc
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(1.5)
        if t not in caps:
            failed[t] = str(last_error)

    if failed:
        print("\n  ✗ Live market-cap retrieval failed for the following assets; execution stopped and no fallback weights were used:")
        for t, err in failed.items():
            print(f"    {t}: {err}")
        raise RuntimeError("Live market-cap data are incomplete; retry later or remove the failed tickers")

    total = sum(caps.values())
    if total <= 0:
        raise RuntimeError("Live market-cap data are empty; retry later or remove the failed tickers")
    weights = {t: caps[t] / total for t in tickers}
    return weights


def classify_ticker_history(prices, tickers, model_start_date):
    """Require sufficiently complete price history for every configured asset."""
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
            incomplete_tickers.append(
                f"{ticker} ({observed_sessions}/{expected_sessions} sessions)"
            )
        by_ticker[ticker] = {
            "observed_sessions": observed_sessions,
            "expected_sessions": expected_sessions,
            "missing_sessions": missing_sessions,
            "is_full_history": is_full_history,
        }

    if incomplete_tickers:
        raise ValueError(
            "Incomplete price history after the model start date; remove these "
            "tickers from TICKERS and retry: " + ", ".join(incomplete_tickers)
        )

    return {
        "reference_ticker": reference_ticker,
        "expected_sessions": expected_sessions,
        "by_ticker": by_ticker,
        "active_tickers": list(tickers),
        "model_start_date": model_start_date,
    }


def _transform_return_series(daily_returns, frequency):
    """Transform returns on each asset's actual timeline without contaminating peers."""
    clean = daily_returns.dropna()
    if frequency == "monthly":
        return clean.resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
    if frequency == "weekly_avg_daily":
        return _rolling_double_smoothed_average(clean)
    if frequency == "daily":
        return clean
    raise ValueError("FREQUENCY must be monthly, daily, or weekly_avg_daily")


def periods_per_year(frequency):
    """Return the annualization multiplier for the return scale."""
    if frequency == "monthly":
        return 12
    if frequency == "weekly_avg_daily":
        return 252
    if frequency == "daily":
        return 252
    raise ValueError(f"Unknown FREQUENCY: {frequency}")


def observations_per_year(frequency):
    """Return the number of observations per year for the selected frequency."""
    if frequency == "monthly":
        return 12
    if frequency == "weekly_avg_daily":
        return 252
    if frequency == "daily":
        return 252
    raise ValueError(f"Unknown FREQUENCY: {frequency}")


def observations_from_years(years, frequency):
    """Convert years to observations under the selected FREQUENCY."""
    return max(1, int(round(years * observations_per_year(frequency))))


def return_period_label(frequency):
    labels = {
        "monthly": "Monthly",
        "weekly_avg_daily": "Double-rolling smoothed daily returns",
        "daily": "Daily",
    }
    return labels.get(frequency, frequency)


def smoothing_horizon_days():
    """Return trading days covered by each double-rolling smoothed observation."""
    return FIRST_SMOOTHING_DAYS + SECOND_SMOOTHING_OBS - 1


def smoothing_kernel_weights():
    """Return triangular daily weights implied by double rolling weak-law smoothing."""
    return np.convolve(
        np.ones(FIRST_SMOOTHING_DAYS) / FIRST_SMOOTHING_DAYS,
        np.ones(SECOND_SMOOTHING_OBS) / SECOND_SMOOTHING_OBS
    )


def smoothing_first_peak_index():
    """Return the first index at the triangular/trapezoidal weight peak."""
    base_weights = smoothing_kernel_weights()
    return int(np.argmax(base_weights))


def smoothing_decline_start_index():
    """Return the first index where triangular/trapezoidal weight falls below its peak."""
    base_weights = smoothing_kernel_weights()
    peak_idx = smoothing_first_peak_index()
    peak_weight = base_weights[peak_idx]
    lower_after_peak = np.where(base_weights[peak_idx:] < peak_weight)[0]
    if len(lower_after_peak) == 0:
        return peak_idx
    return peak_idx + int(lower_after_peak[0])


def smoothing_ew_halflife_days():
    """Set small-EW half-life from the true decline start through the latest day."""
    latest_idx = len(smoothing_kernel_weights()) - 1
    decline_start_idx = smoothing_decline_start_index()
    return max(1, latest_idx - decline_start_idx)


def smoothing_ew_start_index():
    """
    Return the small-EW adjustment-window start.
    Extend backward from the first peak until the first in-window weight is not
    below the nearest weight outside the window.
    """
    base_weights = smoothing_kernel_weights()
    halflife = smoothing_ew_halflife_days()
    peak_idx = smoothing_first_peak_index()

    for start_idx in range(peak_idx, -1, -1):
        tail = base_weights[start_idx:]
        ages = np.arange(len(tail) - 1, -1, -1, dtype=float)
        ew_tail = 0.5 ** (ages / halflife)
        ew_tail = ew_tail ** RECENT_TAIL_EW_POWER
        ew_tail = ew_tail / ew_tail.sum() * tail.sum()
        outside_weight = base_weights[start_idx - 1] if start_idx > 0 else 0.0
        if ew_tail[0] >= outside_weight:
            return start_idx

    return 0


def smoothing_ew_tail_days():
    """Return small-EW adjustment length from its automatic start to the latest day."""
    return len(smoothing_kernel_weights()) - smoothing_ew_start_index()


def _rolling_double_smoothed_average(daily_returns):
    """
    Transform daily returns into double-rolling weak-law-smoothed observations.
    Layer one averages FIRST_SMOOTHING_DAYS trading days; layer two averages
    SECOND_SMOOTHING_OBS consecutive first-layer observations. The small-EW
    window extends automatically to avoid a boundary drop, with its half-life
    beginning where the kernel weights genuinely start to decline.
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

    base_weights = smoothing_kernel_weights()
    start_idx = smoothing_ew_start_index()
    tail_mass = base_weights[start_idx:].sum()
    tail_len = smoothing_ew_tail_days()
    ages = np.arange(tail_len - 1, -1, -1, dtype=float)  # 最新日age=0
    tail_halflife = smoothing_ew_halflife_days()
    tail_weights = 0.5 ** (ages / tail_halflife)
    tail_weights = tail_weights ** RECENT_TAIL_EW_POWER
    tail_weights = tail_weights / tail_weights.sum() * tail_mass

    latest_weights = base_weights.copy()
    latest_weights[start_idx:] = tail_weights
    latest_window = daily_returns.iloc[-len(latest_weights):]
    second_smoothed.iloc[-1] = latest_window.mul(latest_weights, axis=0).sum()
    return second_smoothed.dropna()


def _apply_recent_tail_ew_weights(weights):
    """
    Apply small-EW adjustment to the newest tail of final smoothed observations.
    The window extends automatically to avoid a boundary drop, with half-life
    beginning where kernel weights genuinely start to decline.
    """
    if FREQUENCY != "weekly_avg_daily" or weights is None:
        return weights

    adjusted = np.asarray(weights, dtype=float).copy()
    tail_len = min(len(adjusted), smoothing_ew_tail_days())
    if tail_len <= 1:
        return adjusted

    tail_mass = adjusted[-tail_len:].sum()
    ages = np.arange(tail_len - 1, -1, -1, dtype=float)  # 最新观测age=0
    tail_halflife = smoothing_ew_halflife_days()
    tail_weights = 0.5 ** (ages / tail_halflife)
    tail_weights = tail_weights ** RECENT_TAIL_EW_POWER
    tail_weights = tail_weights / tail_weights.sum() * tail_mass
    adjusted[-tail_len:] = tail_weights
    return adjusted


def fetch_stock_data(tickers, years, frequency, include_raw_daily=False,
                     include_history_info=False, history_tickers=None):
    """Fetch stock and benchmark returns, with history eligibility for stocks only."""
    end_date = datetime.now()
    # 额外 180 天仅供 SPY/QQQ 的 benchmark 与 regime 构造复用；
    # 模型收益仍截回原有 90 天缓冲起点，避免扩大回归样本期。
    model_start_date = end_date - timedelta(days=years * 365 + 90)
    download_start_date = end_date - timedelta(days=years * 365 + 180)

    print("  Downloading price data...")
    prices = yf.download(tickers, start=download_start_date, end=end_date,
                         auto_adjust=True, progress=False)["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices[tickers]
    eligible_tickers = list(tickers if history_tickers is None else history_tickers)
    history_info = classify_ticker_history(
        prices, eligible_tickers, model_start_date)

    raw_daily_returns = prices.pct_change(fill_method=None)
    raw_daily_returns = raw_daily_returns.loc[
        raw_daily_returns.index >= model_start_date
    ]
    transformed = {
        ticker: _transform_return_series(raw_daily_returns[ticker], frequency)
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
    """Build five-trading-day regime observations from downloaded SPY/QQQ returns."""
    benchmark_tickers = sorted(set(REGIME_BENCHMARKS))
    missing = [t for t in benchmark_tickers if t not in raw_daily_returns]
    if missing:
        raise ValueError(f"Raw daily returns are missing regime benchmarks: {missing}")
    daily_returns = raw_daily_returns[benchmark_tickers].dropna(how="any")

    rows = []
    idx = []
    period = REGIME_TRADING_DAYS_PER_PERIOD
    starts = list(range(len(daily_returns) - period, -1, -period))[::-1]
    for start in starts:
        block = daily_returns.iloc[start:start + period].dropna(how="all")
        if len(block) < period:
            continue

        block_ret = (1.0 + block[benchmark_tickers]).prod() - 1.0
        avg_daily = block[benchmark_tickers].mean(axis=1)
        signs = np.sign(avg_daily.values)
        signs[signs == 0] = 1
        flips = (np.mean(signs[1:] != signs[:-1])
                 if len(signs) > 1 else 0.0)

        if len(benchmark_tickers) >= 2:
            both_up = ((block[benchmark_tickers[0]] > 0)
                       & (block[benchmark_tickers[1]] > 0)).mean()
            both_down = ((block[benchmark_tickers[0]] < 0)
                         & (block[benchmark_tickers[1]] < 0)).mean()
        else:
            both_up = (block[benchmark_tickers[0]] > 0).mean()
            both_down = (block[benchmark_tickers[0]] < 0).mean()

        wealth = (1.0 + avg_daily).cumprod()
        period_range = float(wealth.max() / wealth.min() - 1.0)
        first_half = avg_daily.iloc[:max(2, len(avg_daily) // 2)]
        second_half = avg_daily.iloc[len(avg_daily) // 2:]
        first_vol = float(first_half.std()) if len(first_half) > 1 else 0.0
        second_vol = float(second_half.std()) if len(second_half) > 1 else 0.0
        vol_shift = second_vol / max(first_vol, 1e-8)

        row = block_ret.to_dict()
        row.update({
            "__intraperiod_up_ratio": float((avg_daily > 0).mean()),
            "__intraperiod_flips": float(flips),
            "__intraperiod_both_up": float(both_up),
            "__intraperiod_both_down": float(both_down),
            "__intraperiod_vol": float(avg_daily.std()
                                       * np.sqrt(252)),
            "__intraperiod_range": period_range,
            "__intraperiod_vol_shift": float(vol_shift),
        })
        rows.append(row)
        idx.append(block.index[-1])

    return pd.DataFrame(rows, index=idx).dropna(how="all")


def _read_ff_csv(url, date_format="%Y%m", date_len=6):
    """Download a ZIP from Kenneth French's site and parse its CSV."""
    import zipfile
    import requests

    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
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


def _market_risk_price_sample_weights(n_obs):
    """Global big-EW weights for raw daily SPY delta_market estimation."""
    if n_obs <= 0:
        return np.array([], dtype=float)
    if not USE_EW_REGRESSION:
        return np.full(n_obs, 1.0 / n_obs, dtype=float)

    halflife_days = observations_from_years(EW_REGRESSION_HALFLIFE_YEARS,
                                            "daily")
    log_weights = (np.log(2.0) * np.arange(n_obs, dtype=float)
                   / halflife_days * EW_REGRESSION_WEIGHT_POWER)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def estimate_market_risk_price(end_date, lookback_years=None,
                               raw_spy_returns=None):
    """Estimate delta_market from the model window's raw daily SPY returns.

    The window follows DATA_PERIOD_YEARS and the weights follow the global
    big-EW half-life. This remains separate from investor utility aversion.
    """
    lookback_years = (DATA_PERIOD_YEARS if lookback_years is None
                      else float(lookback_years))
    if lookback_years < 1:
        raise ValueError("DATA_PERIOD_YEARS must be at least one year to estimate delta_market")

    end_date = pd.Timestamp(end_date)
    start_date = end_date - pd.Timedelta(
        days=int(np.ceil(lookback_years * 365.25))
    )
    if raw_spy_returns is None:
        spy_prices = yf.download(
            "SPY", start=start_date - pd.Timedelta(days=7),
            end=end_date + pd.Timedelta(days=1), auto_adjust=True,
            progress=False,
        )["Close"]
        if isinstance(spy_prices, pd.DataFrame):
            spy_prices = spy_prices.iloc[:, 0]
        spy_returns = spy_prices.pct_change(fill_method=None).dropna()
    else:
        spy_returns = pd.Series(raw_spy_returns, dtype=float).loc[
            start_date:end_date
        ].dropna()
    daily_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / periods_per_year("daily")) - 1.0
    market_excess = spy_returns - daily_rf
    min_observations = int(lookback_years * periods_per_year("daily") * 0.8)
    if len(market_excess) < min_observations:
        raise ValueError(
            "Insufficient SPY daily returns to estimate delta_market: "
            f"{len(market_excess)} < {min_observations}"
        )

    weights = _market_risk_price_sample_weights(len(market_excess))
    daily_mean = float(np.dot(weights, market_excess.to_numpy(dtype=float)))
    centered = market_excess.to_numpy(dtype=float) - daily_mean
    variance_denominator = 1.0 - float(np.dot(weights, weights))
    if variance_denominator <= 1e-12:
        raise ValueError("SPY EW effective sample size is insufficient to estimate delta_market")
    daily_variance = float(np.dot(weights, centered ** 2)
                           / variance_denominator)
    annual_mean = daily_mean * periods_per_year("daily")
    annual_variance = daily_variance * periods_per_year("daily")
    if annual_variance <= 1e-12:
        raise ValueError("SPY annualized variance is invalid; cannot estimate delta_market")

    raw_delta_market = annual_mean / annual_variance
    # A negative realized market premium supplies no positive CAPM prior.
    # Keep pi neutral and let the BL view layer determine expected returns.
    return float(max(raw_delta_market, 0.0))


def _regression_sample_weights(n_obs):
    """Return the same EW observation weights used by factor regressions."""
    if n_obs <= 0:
        return np.array([], dtype=float)
    if not USE_EW_REGRESSION:
        return np.ones(n_obs, dtype=float)

    halflife = observations_from_years(EW_REGRESSION_HALFLIFE_YEARS,
                                       FREQUENCY)
    decay = np.log(2) / halflife
    weights = np.exp(decay * np.arange(n_obs, dtype=float))
    weights = weights ** EW_REGRESSION_WEIGHT_POWER
    weights = weights / weights.sum() * n_obs
    return _apply_recent_tail_ew_weights(weights)


def _weighted_correlation(left, right, weights):
    """EW-weighted Pearson correlation for aligned factor observations."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (np.isfinite(left) & np.isfinite(right) & np.isfinite(weights)
             & (weights > 0))
    if valid.sum() < 3:
        return np.nan
    left, right, weights = left[valid], right[valid], weights[valid]
    weights = weights / weights.sum()
    left_centered = left - np.sum(weights * left)
    right_centered = right - np.sum(weights * right)
    denominator = np.sqrt(np.sum(weights * left_centered ** 2)
                          * np.sum(weights * right_centered ** 2))
    if denominator <= 1e-15:
        return np.nan
    return float(np.clip(np.sum(weights * left_centered * right_centered)
                         / denominator, -1.0, 1.0))


def _weighted_vif(y, X, weights):
    """VIF of y against X using the factor regression's EW observation weights."""
    if X.empty:
        return 1.0
    model = sm.WLS(y, sm.add_constant(X, has_constant="add"),
                   weights=weights).fit()
    r_squared = float(np.clip(model.rsquared, 0.0, 1.0 - 1e-12))
    return float(1.0 / (1.0 - r_squared))


def _weighted_design_condition_number(factor_df):
    """Condition number of the standardized EW-weighted final factor design."""
    aligned = factor_df.dropna(how="any")
    if aligned.empty or aligned.shape[1] < 2:
        return np.nan
    weights = _regression_sample_weights(len(aligned))
    values = aligned.to_numpy(dtype=float)
    weights = weights / weights.sum()
    means = np.sum(values * weights[:, None], axis=0)
    scales = np.sqrt(np.sum(weights[:, None] * (values - means) ** 2,
                            axis=0))
    keep = scales > 1e-12
    if keep.sum() < 2:
        return np.nan
    design = (values[:, keep] - means[keep]) / scales[keep]
    try:
        return float(np.linalg.cond(design * np.sqrt(weights)[:, None]))
    except np.linalg.LinAlgError:
        return np.inf


def orthogonalize_custom_factors(custom_df, base_factors, base_names,
                                  factor_layers):
    """
    Preserve FF residualization for every custom factor, then automatically
    remove lower-layer custom-factor overlap only when EW diagnostics flag it.
    Factors in the same layer are never ordered against each other.
    """
    available_base = [f for f in base_names if f in base_factors.columns]
    if custom_df.empty:
        return custom_df, [], {"condition_number": np.nan, "alert": False}

    unknown_layers = sorted(set(factor_layers.values())
                            - set(FACTOR_LAYER_ORDER))
    if unknown_layers:
        raise ValueError(f"Unknown custom-factor layers: {unknown_layers}")

    original_order = {name: i for i, name in enumerate(custom_df.columns)}
    processing_order = sorted(
        custom_df.columns,
        key=lambda name: (FACTOR_LAYER_ORDER[factor_layers[name]],
                          original_order[name])
    )
    orthogonalized = pd.DataFrame(index=custom_df.index)
    diagnostics = []

    for name in processing_order:
        layer = factor_layers[name]
        layer_rank = FACTOR_LAYER_ORDER[layer]
        lower_names = [candidate for candidate in orthogonalized.columns
                       if FACTOR_LAYER_ORDER[factor_layers[candidate]]
                       < layer_rank]

        candidate_df = base_factors[available_base].copy()
        if lower_names:
            candidate_df = candidate_df.join(orthogonalized[lower_names],
                                             how="inner")
        aligned = pd.concat([custom_df[name], candidate_df], axis=1).dropna()
        if len(aligned) <= len(candidate_df.columns) + 2:
            orthogonalized[name] = custom_df[name]
            diagnostics.append({
                "factor": name,
                "layer": layer,
                "status": "skipped",
                "action": "skipped",
                "reason": "insufficient observations",
            })
            continue

        y = aligned[name]
        weights = _regression_sample_weights(len(aligned))
        base_vif = _weighted_vif(y, aligned[available_base], weights)
        full_vif = _weighted_vif(y, aligned[candidate_df.columns], weights)

        lower_corrs = {
            lower: _weighted_correlation(y, aligned[lower], weights)
            for lower in lower_names
        }
        finite_lower_corrs = {
            lower: abs(corr) for lower, corr in lower_corrs.items()
            if np.isfinite(corr)
        }
        strongest_lower = (max(finite_lower_corrs, key=finite_lower_corrs.get)
                           if finite_lower_corrs else None)
        max_lower_corr = (finite_lower_corrs.get(strongest_lower, np.nan)
                          if strongest_lower is not None else np.nan)
        corr_trigger = (np.isfinite(max_lower_corr)
                        and max_lower_corr >= AUTO_LOWER_LAYER_ORTHOGONALIZE_CORR)
        vif_trigger = bool(lower_names and full_vif >= FACTOR_VIF_ALERT)
        use_lower_layers = bool(lower_names and (corr_trigger or vif_trigger))
        regressors = (available_base + lower_names
                      if use_lower_layers else available_base)

        if ORTHOGONALIZE_CUSTOM_FACTORS and available_base:
            model = sm.OLS(y, sm.add_constant(aligned[regressors],
                                               has_constant="add")).fit()
            residual = pd.Series(index=custom_df.index, dtype=float)
            residual.loc[aligned.index] = model.resid
            orthogonalized[name] = residual
            action = "FF+lower" if use_lower_layers else "FF only"
            r_squared = float(model.rsquared)

            after_aligned = pd.concat(
                [orthogonalized[name], aligned[regressors]], axis=1
            ).dropna()
            after_weights = _regression_sample_weights(len(after_aligned))
            after_corrs = [
                abs(_weighted_correlation(after_aligned[name],
                                           after_aligned[regressor],
                                           after_weights))
                for regressor in regressors
            ]
            after_corrs = [value for value in after_corrs if np.isfinite(value)]
            max_corr_after = max(after_corrs) if after_corrs else np.nan
        else:
            orthogonalized[name] = custom_df[name]
            action = "disabled"
            r_squared = np.nan
            max_corr_after = np.nan

        diagnostics.append({
            "factor": name,
            "layer": layer,
            "status": "orthogonalized" if action != "disabled" else "disabled",
            "action": action,
            "lower_factors": lower_names,
            "strongest_lower": strongest_lower,
            "max_lower_corr": max_lower_corr,
            "base_vif": base_vif,
            "vif": full_vif,
            "r_squared": r_squared,
            "max_corr_after": max_corr_after,
            "auto_triggered": use_lower_layers,
        })

    orthogonalized = orthogonalized.reindex(columns=custom_df.columns)
    final_design = base_factors[available_base].join(orthogonalized, how="inner")
    condition_number = _weighted_design_condition_number(final_design)
    collinearity_info = {
        "condition_number": condition_number,
        "threshold": FACTOR_CONDITION_NUMBER_ALERT,
        "alert": bool(np.isfinite(condition_number)
                      and condition_number >= FACTOR_CONDITION_NUMBER_ALERT),
    }
    return orthogonalized.dropna(how="all"), diagnostics, collinearity_info


def _custom_factor_proxy_tickers():
    """Return the deduplicated proxy tickers required by CUSTOM_FACTORS."""
    tickers = set()
    for long_t, short_t, _ in CUSTOM_FACTORS.values():
        if isinstance(long_t, list):
            tickers.update(long_t)
        else:
            tickers.add(long_t)
        if short_t and short_t != "Mkt":
            tickers.add(short_t)
    return sorted(tickers)


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_cached_french_daily_factor_file(path):
    """Parse a cached Kenneth French daily ZIP without network fallback."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0]).decode("utf-8")
    lines = raw.splitlines()
    first_data = next(
        (i for i, line in enumerate(lines)
         if len(line.strip().split(",")[0].strip()) == 8
         and line.strip().split(",")[0].strip().isdigit()),
        None,
    )
    if first_data is None:
        raise ValueError(f"No daily observations found in cached factor file {path}")
    header = first_data - 1
    while header >= 0 and not lines[header].strip():
        header -= 1
    kept = []
    for line in lines[header:]:
        first = line.strip().split(",")[0].strip()
        if not line.strip():
            continue
        if not kept or (len(first) == 8 and first.isdigit()):
            kept.append(line.strip())
        elif not first.isdigit():
            break
    frame = pd.read_csv(io.StringIO("\n".join(kept)), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    return frame.apply(pd.to_numeric, errors="coerce").replace(
        [-99.99, -999.0, -999.99], np.nan) / 100.0


def _read_cached_aqr_daily_factor(path, sheet_name, factor_name):
    """Read AQR daily long-short returns; they are already decimal returns."""
    frame = pd.read_excel(path, sheet_name=sheet_name, skiprows=18)
    required = {"DATE", "USA"}
    if not required <= set(frame.columns):
        raise ValueError(
            f"Unexpected AQR layout in {path.name}; expected columns {sorted(required)}")
    dates = pd.to_datetime(frame["DATE"], errors="coerce")
    values = pd.to_numeric(frame["USA"], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float), index=dates,
                     name=factor_name).dropna()


def load_external_return_factors(start_date, end_date):
    """Load the exact non-ETF F factors and fail closed on source drift/staleness."""
    cache = Path(EXTERNAL_FACTOR_CACHE)
    required_paths = {name: cache / name
                      for name in EXTERNAL_FACTOR_REQUIRED_SHA256}
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "F factor source cache is incomplete; do not substitute proxy data: "
            + ", ".join(missing))
    changed = [name for name, path in required_paths.items()
               if _sha256_file(path) != EXTERNAL_FACTOR_REQUIRED_SHA256[name]]
    if changed:
        raise RuntimeError(
            "F factor source cache hash changed; review and update the approved "
            "source manifest before using it: " + ", ".join(changed))

    industry49 = _read_cached_french_daily_factor_file(
        required_paths["49_Industry_Portfolios_Daily_CSV.zip"])
    required_industries = {"Hardw", "Chips"}
    missing_industries = required_industries - set(industry49.columns)
    if missing_industries:
        raise RuntimeError(
            f"Cached French 49-industry file lacks required columns: {sorted(missing_industries)}")
    # Keep the current chronological union behavior explicit across Pandas
    # versions; the final sort remains an invariant check on the output index.
    external = pd.concat({
        "QMJ": _read_cached_aqr_daily_factor(
            required_paths["qmj_daily.xlsx"], "QMJ Factors", "QMJ"),
        "BAB": _read_cached_aqr_daily_factor(
            required_paths["bab_daily_download.xlsx"], "BAB Factors", "BAB"),
        "Hardw49": industry49["Hardw"],
        "Chips49": industry49["Chips"],
    }, axis=1, sort=True).sort_index()
    if list(external.columns) != EXTERNAL_RETURN_FACTORS:
        raise RuntimeError(
            f"External F factor column order changed: {list(external.columns)}")
    external = external.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if external.empty or external.isna().all(axis=None):
        raise RuntimeError("F factor sources have no usable observations for the model window")
    last_observation = pd.Timestamp(external.dropna(how="all").index.max())
    stale_days = (pd.Timestamp(end_date).normalize() - last_observation.normalize()).days
    if stale_days > EXTERNAL_FACTOR_MAX_STALENESS_DAYS:
        raise RuntimeError(
            "F factor source cache is stale by "
            f"{stale_days} calendar days (limit={EXTERNAL_FACTOR_MAX_STALENESS_DAYS}); "
            "refresh the approved AQR/French files before running the formal model")
    return external


def fetch_daily_factor_inputs(start_date, end_date, verbose=True):
    """Download raw daily factor inputs once for signal and risk factor frames."""
    if verbose:
        print("  Downloading Fama-French factor data...")
    ff5_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom_url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")
    ff5_data = _read_ff_csv(ff5_url, date_format="%Y%m%d", date_len=8)
    mom_data = _read_ff_csv(mom_url, date_format="%Y%m%d", date_len=8)
    if mom_data.shape[1] > 1:
        mom_data = mom_data.iloc[:, [0]]
    mom_data.columns = ["Mom"]
    daily_factors = ff5_data.join(mom_data, how="inner") / 100.0

    proxy_tickers = _custom_factor_proxy_tickers()
    custom_prices = None
    if proxy_tickers:
        if verbose:
            print(f"  Downloading custom factor proxies "
                  f"({len(proxy_tickers)} tickers)...")
        custom_start = pd.Timestamp(start_date) - pd.Timedelta(days=60)
        custom_prices = yf.download(
            proxy_tickers, start=custom_start, end=end_date,
            auto_adjust=True, progress=False,
        )["Close"]
        if isinstance(custom_prices, pd.Series):
            custom_prices = custom_prices.to_frame(proxy_tickers[0])

    external_factors = load_external_return_factors(start_date, end_date)
    return daily_factors, custom_prices, external_factors


def fetch_ff_factors(start_date, end_date, frequency, verbose=True,
                     daily_inputs=None):
    """Build factor returns from Kenneth French data and custom proxy inputs."""
    custom_prices = None
    external_factors = None
    if frequency == "monthly":
        if verbose:
            print("  Downloading Fama-French factor data...")
        FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                   "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
        MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                   "ken.french/ftp/F-F_Momentum_Factor_CSV.zip")
        ff5_data = _read_ff_csv(FF5_URL)
        mom_data = _read_ff_csv(MOM_URL)
    elif frequency in ("daily", "weekly_avg_daily"):
        if daily_inputs is None:
            factors, custom_prices, external_factors = fetch_daily_factor_inputs(
                start_date, end_date, verbose=verbose)
        else:
            factors, custom_prices, external_factors = daily_inputs
            factors = factors.copy()
    else:
        raise ValueError("FREQUENCY must be monthly, daily, or weekly_avg_daily")

    if frequency == "monthly":
        if mom_data.shape[1] > 1:
            mom_data = mom_data.iloc[:, [0]]
        mom_data.columns = ["Mom"]
        factors = ff5_data.join(mom_data, how="inner") / 100.0

    if frequency == "weekly_avg_daily":
        factors = _rolling_double_smoothed_average(factors)
        external_factors = _rolling_double_smoothed_average(external_factors)

    # 按日期范围过滤
    factors = factors.loc[start_date:end_date]

    # Monthly mode is not the formal F setting, but retain a causal compound
    # aggregation rule rather than treating daily external returns as prices.
    if frequency == "monthly":
        external_factors = load_external_return_factors(start_date, end_date)
        external_factors = (1.0 + external_factors).resample("ME").prod() - 1.0
        external_factors.index = external_factors.index.to_period("M").to_timestamp()

    # ── ETF 自定义因子 (当前 F 设计为空，保留通用构建器) ──
    proxy_tickers = _custom_factor_proxy_tickers()
    if proxy_tickers:
        if custom_prices is None:
            if verbose:
                print(f"  Downloading custom factor proxies "
                      f"({len(proxy_tickers)} tickers)...")
            custom_start = pd.Timestamp(start_date) - pd.Timedelta(days=60)
            custom_prices = yf.download(
                proxy_tickers, start=custom_start, end=end_date,
                auto_adjust=True, progress=False,
            )["Close"]
            if isinstance(custom_prices, pd.Series):
                custom_prices = custom_prices.to_frame(proxy_tickers[0])
        if frequency == "monthly":
            custom_prices = custom_prices.resample("ME").last()
            custom_rets = custom_prices.pct_change().dropna()
            custom_rets.index = custom_rets.index.to_period("M").to_timestamp()
        elif frequency == "weekly_avg_daily":
            custom_rets = custom_prices.pct_change().dropna()
            custom_rets = _rolling_double_smoothed_average(custom_rets)
        else:
            custom_rets = custom_prices.pct_change().dropna()

        # 需要先对齐 Mkt-RF 以支持 "Mkt" short
        aligned_factors = factors.join(custom_rets, how="inner")
        mkt_rf = aligned_factors["Mkt-RF"]

        # 按配置构造每个因子
        custom_df = pd.DataFrame(index=aligned_factors.index)
        for name, (long_t, short_t, _) in CUSTOM_FACTORS.items():
            # long: 单资产或篮子等权
            if isinstance(long_t, list):
                long_series = aligned_factors[long_t].mean(axis=1)
            else:
                long_series = aligned_factors[long_t]
            # short: None / "Mkt" / 配对
            if short_t is None:
                custom_df[name] = long_series
            elif short_t == "Mkt":
                custom_df[name] = long_series - mkt_rf
            else:
                custom_df[name] = long_series - aligned_factors[short_t]

        factor_layers = {
            name: layer for name, (_, _, layer) in CUSTOM_FACTORS.items()
        }
        custom_df, orth_diag, factor_collinearity = orthogonalize_custom_factors(
            custom_df, aligned_factors, ORTHOGONALIZE_AGAINST, factor_layers
        )

        # 只保留选择的 FF + ETF 自定义因子, 丢掉中间下载的原始 ticker 列
        factors = aligned_factors[FF_FACTORS + ["RF"]].join(custom_df,
                                                            how="inner")
        factors.attrs["orthogonalization"] = orth_diag
        factors.attrs["factor_collinearity"] = factor_collinearity

    # QMJ/BAB/Hardw49/Chips49 are external return series, not proxy prices.
    # All four share the Theme layer so they are residualized on the two FF
    # factors but are not mechanically residualized against one another.
    if external_factors is None:
        raise RuntimeError("F external return factors were not loaded")
    selected = factors[FF_FACTORS + ["RF"] + list(CUSTOM_FACTORS)]
    factors = selected.join(external_factors[EXTERNAL_RETURN_FACTORS], how="inner")
    factor_layers = ({name: layer for name, (_, _, layer) in CUSTOM_FACTORS.items()}
                     | {name: EXTERNAL_FACTOR_LAYER
                        for name in EXTERNAL_RETURN_FACTORS})
    custom_names = list(CUSTOM_FACTORS) + EXTERNAL_RETURN_FACTORS
    if custom_names:
        external_and_proxy = factors[custom_names]
        orthogonal, orth_diag, factor_collinearity = orthogonalize_custom_factors(
            external_and_proxy, factors[FF_FACTORS + ["RF"]],
            ORTHOGONALIZE_AGAINST, factor_layers)
        factors = factors[FF_FACTORS + ["RF"]].join(orthogonal, how="inner")
        factors.attrs["orthogonalization"] = orth_diag
        factors.attrs["factor_collinearity"] = factor_collinearity

    return factors



# ============================================================
# STEP 2: 因子回归
# ============================================================

def _fit_asset_factor_regression(series, factors, factor_names, ann_factor):
    """Estimate each full-history asset with exponentially weighted WLS/HAC."""
    required = list(factor_names) + (["RF"] if "RF" in factors else [])
    aligned = factors[required].join(series.rename("asset"), how="inner").dropna()
    min_observations = max(3, len(factor_names) + 2)
    if len(aligned) < min_observations:
        raise ValueError("Insufficient usable asset-return observations for WLS/HAC regression")

    rf = aligned["RF"] if "RF" in aligned.columns else 0.0
    y = aligned["asset"] - rf
    X_raw = aligned[factor_names]
    X = sm.add_constant(X_raw)
    weights = (_regression_sample_weights(len(aligned))
               if USE_EW_REGRESSION else None)
    regression = sm.WLS(y, X, weights=weights) if weights is not None else sm.OLS(y, X)
    model = regression.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAX_LAGS, "use_correction": True},
    )
    model_ols = sm.OLS(y, X).fit()
    return {
        "alpha": float(model.params["const"]),
        "beta": model.params[factor_names].to_numpy(dtype=float),
        "beta_stderr": model.bse[factor_names].to_numpy(dtype=float),
        "alpha_pvalue": float(model.pvalues["const"]),
        "factor_pvalues": model.pvalues[factor_names].to_numpy(dtype=float),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residuals": model_ols.resid,
        "n_obs": int(model.nobs),
        "estimator": "WLS-HAC",
    }


def run_factor_regression(stock_returns, factors, factor_names):
    """Run WLS/HAC factor regressions over the shared full-history asset universe."""
    ann_factor = periods_per_year(FREQUENCY)
    tickers = list(stock_returns.columns)
    orthogonalization_info = factors.attrs.get("orthogonalization", [])
    factor_collinearity_info = factors.attrs.get("factor_collinearity")

    def run_pass(active_factors, active_factor_names):
        rows = []
        regression_stats = {}
        residual_columns = {}
        for ticker in tickers:
            fitted = _fit_asset_factor_regression(
                stock_returns[ticker], active_factors, active_factor_names,
                ann_factor=ann_factor,
            )
            rows.append(fitted["beta"])
            residual_columns[ticker] = fitted["residuals"]
            regression_stats[ticker] = {
                "alpha_annualized": fitted["alpha"] * ann_factor,
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
        return (np.asarray(rows, dtype=float), regression_stats,
                pd.DataFrame(residual_columns))

    B, regression_stats, residuals_df = run_pass(factors, factor_names)

    pairs = []
    pair_observations = {}
    for i, ticker_i in enumerate(tickers):
        for ticker_j in tickers[i + 1:]:
            pair = residuals_df[[ticker_i, ticker_j]].dropna(how="any")
            if len(pair) < 3:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if not np.isfinite(corr):
                continue
            pairs.append((abs(corr), ticker_i, ticker_j, corr))
            pair_observations[(ticker_i, ticker_j)] = int(len(pair))
    pairs.sort(reverse=True)
    abs_corr = [pair[0] for pair in pairs]
    resid_diag = {
        "mean": float(np.mean(abs_corr)) if abs_corr else 0.0,
        "max": float(np.max(abs_corr)) if abs_corr else 0.0,
        "top_pairs": pairs[:5],
        "all_pairs": pairs,
        "pair_observations": pair_observations,
        "orthogonalization": orthogonalization_info,
        "factor_collinearity": factor_collinearity_info,
    }
    return B, regression_stats, resid_diag, factor_names, factors


# ============================================================
# STEP 3: 协方差矩阵构建
# ============================================================

def _annual_sample_covariance(returns):
    """Annualize the unbiased sample covariance of raw daily returns."""
    values = returns.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("At least two raw daily observations are required")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return (covariance + covariance.T) / 2.0 * periods_per_year("daily")


def _annual_downside_semicovariance(returns):
    """Annualize centered covariance after retaining only non-positive daily returns."""
    downside_returns = returns.clip(upper=0.0).fillna(0.0)
    return _annual_sample_covariance(downside_returns)


def _annual_raw_ledoit_wolf_covariance(returns):
    """Annualize Ledoit-Wolf covariance of complete raw daily returns.

    This estimator is used only by the Black-Litterman equilibrium prior and
    view uncertainty.  It shrinks raw-return covariance to a scaled identity;
    it must not be substituted for the downside risk covariance used by the
    optimizer and execution layer.
    """
    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(
        returns.to_numpy(dtype=float)
    )
    covariance = (estimator.covariance_ + estimator.covariance_.T) / 2.0
    return covariance * periods_per_year("daily"), float(estimator.shrinkage_)


def _annual_downside_ledoit_wolf_covariance(returns):
    """Annualize Ledoit-Wolf covariance of clipped daily downside returns.

    Positive daily returns are set to zero before fitting.  The estimator
    shrinks the centered downside covariance toward a scaled-identity target;
    its data-driven shrinkage intensity is returned for audit reporting.
    """
    downside_returns = returns.clip(upper=0.0).fillna(0.0)
    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(
        downside_returns.to_numpy(dtype=float)
    )
    covariance = (estimator.covariance_ + estimator.covariance_.T) / 2.0
    return covariance * periods_per_year("daily"), float(estimator.shrinkage_)


def build_equilibrium_prior_covariance(raw_daily_returns, tickers,
                                       recent_weight=0.0, return_info=False):
    """Build the raw-Ledoit-Wolf covariance used only for the BL return prior.

    The prior preserves complete raw daily returns, while
    :func:`build_covariance_matrices` separately constructs the
    Ledoit-Wolf-shrunk downside semivariance for risk and execution.  Setting
    ``return_info=True`` exposes shrinkage diagnostics to the production audit
    without changing the existing covariance-only caller interface.
    """
    max_obs = int(DATA_PERIOD_YEARS * periods_per_year("daily"))
    recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
    asset_returns = raw_daily_returns[tickers].dropna(how="any").tail(max_obs)
    if len(asset_returns) < len(tickers) + 2:
        raise ValueError("Insufficient raw daily returns for equilibrium-prior covariance")
    Sigma_long, long_shrinkage = _annual_raw_ledoit_wolf_covariance(asset_returns)
    info = {
        "equilibrium_prior_estimator": "raw_ledoit_wolf_covariance",
        "equilibrium_prior_ledoit_wolf_target": "scaled_identity",
        "equilibrium_prior_long_observations": int(len(asset_returns)),
        "equilibrium_prior_recent_observations": 0,
        "equilibrium_prior_recent_weight": 0.0,
        "equilibrium_prior_long_shrinkage": long_shrinkage,
        "equilibrium_prior_recent_shrinkage": None,
    }
    Sigma_prior = Sigma_long
    if recent_weight > 0.0:
        recent_returns = asset_returns.tail(RECENT_COVARIANCE_WINDOW_DAYS)
        if len(recent_returns) >= len(tickers) + 2:
            Sigma_recent, recent_shrinkage = _annual_raw_ledoit_wolf_covariance(recent_returns)
            Sigma_prior = recent_weight * Sigma_recent + (1.0 - recent_weight) * Sigma_long
            info.update({
                "equilibrium_prior_recent_observations": int(len(recent_returns)),
                "equilibrium_prior_recent_weight": recent_weight,
                "equilibrium_prior_recent_shrinkage": recent_shrinkage,
            })
    return (Sigma_prior, info) if return_info else Sigma_prior


def _fit_raw_risk_factor_regression(series, risk_factors, factor_names):
    """Estimate covariance-specific betas from raw daily returns, separately from smoothed-return regression."""
    required = list(factor_names) + (["RF"] if "RF" in risk_factors else [])
    aligned = risk_factors[required].join(series.rename("asset"), how="inner").dropna()
    min_observations = max(3, len(factor_names) + 2)
    if len(aligned) < min_observations:
        raise ValueError("Insufficient usable raw daily returns for risk-factor regression")
    y = aligned["asset"] - (aligned["RF"] if "RF" in aligned else 0.0)
    X_raw = aligned[factor_names]
    model = sm.OLS(y, sm.add_constant(X_raw)).fit()
    return model.params[factor_names].to_numpy(dtype=float), model.resid


def _automatic_risk_groups(tickers, Sigma, pair_info):
    """Automatically form common-risk asset groups with complete linkage and rho(total risk)."""
    if len(tickers) < 2:
        return []
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    vol = np.sqrt(np.maximum(np.diag(Sigma), 1e-16))
    corr = Sigma / np.outer(vol, vol)
    corr = np.clip((corr + corr.T) / 2, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    distance = 1.0 - corr
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(
        linkage_matrix, t=1.0 - AUTO_RISK_GROUP_CORRELATION,
        criterion="distance",
    )
    groups = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            continue
        members = [tickers[idx] for idx in indices]
        pair_stats = []
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                key = (tickers[left], tickers[right])
                info = pair_info.get(key, {})
                pair_stats.append({
                    "left": tickers[left],
                    "right": tickers[right],
                    "factor_corr": float(info.get("factor_corr", np.nan)),
                    "residual_corr": float(info.get("residual_corr", np.nan)),
                    "common_days": int(info.get("common_days", 0)),
                    "total_corr": float(corr[left, right]),
                })
        groups.append({"members": members, "pairs": pair_stats})
    return groups


def build_covariance_matrices(raw_daily_returns, tickers, risk_factors,
                              risk_factor_names, recent_weight=0.0,
                              verbose=True):
    """Build Ledoit-Wolf-shrunk downside semivariance for portfolio risk."""
    if verbose:
        print("\n[Step 3] Building Ledoit-Wolf-shrunk downside-semivariance risk covariance matrix...")
    max_obs = int(DATA_PERIOD_YEARS * periods_per_year("daily"))
    recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
    asset_returns = raw_daily_returns[tickers].dropna(how="any").tail(max_obs)
    if len(asset_returns) < len(tickers) + 2:
        raise ValueError("Insufficient raw daily returns for full-universe risk covariance estimation")

    Sigma_long, long_shrinkage = _annual_downside_ledoit_wolf_covariance(asset_returns)
    Sigma_assets_annual = Sigma_long
    covariance_info = {
        "risk_estimator": "downside_semivariance_ledoit_wolf",
        "ledoit_wolf_target": "scaled_identity",
        "long_observations": int(len(asset_returns)),
        "recent_observations": 0,
        "recent_weight": 0.0,
        "long_shrinkage": long_shrinkage,
        "recent_shrinkage": None,
        "asset_count": len(tickers),
    }
    if recent_weight > 0.0:
        recent_returns = asset_returns.tail(RECENT_COVARIANCE_WINDOW_DAYS)
        if len(recent_returns) >= len(tickers) + 2:
            Sigma_recent, recent_shrinkage = _annual_downside_ledoit_wolf_covariance(recent_returns)
            Sigma_assets_annual = (
                recent_weight * Sigma_recent
                + (1.0 - recent_weight) * Sigma_long
            )
            covariance_info.update({
                "recent_observations": int(len(recent_returns)),
                "recent_weight": recent_weight,
                "recent_shrinkage": recent_shrinkage,
            })

    factor_returns = risk_factors[risk_factor_names].dropna(how="any").tail(max_obs)
    if len(factor_returns) < len(risk_factor_names) + 2:
        raise ValueError("Insufficient raw daily factor returns for risk-group diagnostics")
    Sigma_factor_long = _annual_sample_covariance(factor_returns)
    Sigma_factor = Sigma_factor_long
    if recent_weight > 0.0:
        recent_factors = factor_returns.tail(RECENT_COVARIANCE_WINDOW_DAYS)
        if len(recent_factors) >= len(risk_factor_names) + 2:
            Sigma_factor = (
                recent_weight * _annual_sample_covariance(recent_factors)
                + (1.0 - recent_weight) * Sigma_factor_long
            )

    risk_betas = []
    residual_columns = {}
    for ticker in tickers:
        beta, residuals = _fit_raw_risk_factor_regression(
            raw_daily_returns[ticker], risk_factors, risk_factor_names,
        )
        risk_betas.append(beta)
        residual_columns[ticker] = residuals
    risk_betas = np.asarray(risk_betas, dtype=float)
    residuals_df = pd.DataFrame(residual_columns)
    factor_variances = np.einsum(
        "ij,jk,ik->i", risk_betas, Sigma_factor, risk_betas
    )

    pair_info = {}
    for i, ticker_i in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            ticker_j = tickers[j]
            pair_residuals = residuals_df[[ticker_i, ticker_j]].dropna(how="any")
            common_days = int(len(pair_residuals))
            residual_corr = (
                float(pair_residuals.iloc[:, 0].corr(pair_residuals.iloc[:, 1]))
                if common_days >= 3 else 0.0
            )
            if not np.isfinite(residual_corr):
                residual_corr = 0.0
            factor_cov = float(risk_betas[i] @ Sigma_factor @ risk_betas[j])
            factor_corr = factor_cov / np.sqrt(max(
                factor_variances[i] * factor_variances[j], 1e-16
            ))
            pair_info[(ticker_i, ticker_j)] = {
                "factor_corr": float(np.clip(factor_corr, -1.0, 1.0)),
                "residual_corr": float(np.clip(residual_corr, -0.95, 0.95)),
                "common_days": common_days,
            }

    covariance_info["risk_peer_groups"] = _automatic_risk_groups(
        tickers, Sigma_assets_annual, pair_info)
    return Sigma_assets_annual, covariance_info


# ============================================================
# STEP 4: Black-Litterman
# ============================================================

def compute_equilibrium_returns(delta_market, Sigma_annual, w_mkt):
    """Reverse optimization: pi = delta_market * Sigma * w_mkt (excess returns)."""
    pi = delta_market * Sigma_annual @ w_mkt
    return pi


def _factor_view_label(factor_name):
    labels = {
        "Mkt-RF": "Market excess return",
        "SMB": "Size factor (small minus big)",
        "HML": "Value factor (value minus growth)",
        "RMW": "Profitability factor (robust minus weak)",
        "CMA": "Investment factor (conservative minus aggressive)",
        "Mom": "Momentum factor (winners minus losers)",
    }
    return labels.get(factor_name, factor_name)


def _validate_factor_view(view):
    if len(view) != 4 or view[0] not in ("absolute", "relative"):
        raise ValueError(
            "FACTOR_VIEWS supports only the current format: "
            "('absolute', 'HML', -0.03, 0.4) or "
            "('relative', ('SMB', 'HML'), 0.05, 0.5)"
        )


def _bounded_confidence(value):
    """Bound BL view confidence to prevent Omega from becoming too small or too large."""
    return float(np.clip(value, 0.05, 0.95))


def _pvalue_evidence_score(pvalue):
    """Convert a p-value to bounded significance evidence, not a forecast-correctness probability."""
    p = float(np.clip(pvalue, 1e-12, 1.0))
    return float(np.clip(-np.log10(p) / 2.0, 0.0, 1.0))


def _asset_model_confidence(ticker, regression_stats):
    """Estimate statistical confidence in an asset view from regression quality."""
    if not regression_stats or ticker not in regression_stats:
        return None
    stats = regression_stats[ticker]
    adj_r2 = max(0.0, float(stats.get("adj_r_squared", 0.0)))
    r2_score = np.clip(adj_r2 / 0.80, 0.0, 1.0)
    alpha_p = float(stats.get("alpha_pvalue", 1.0))
    alpha_evidence = _pvalue_evidence_score(alpha_p)
    n_obs = float(stats.get("n_obs", 0.0))
    sample_score = np.clip(n_obs / (2.0 * observations_per_year(FREQUENCY)),
                           0.0, 1.0)
    raw = 0.70 * r2_score + 0.10 * alpha_evidence + 0.20 * sample_score
    return _bounded_confidence(0.15 + 0.70 * raw)


def _factor_model_confidence(factor_name, regression_stats):
    """Estimate factor-view confidence from exposure significance and model fit."""
    if not regression_stats:
        return None
    p_scores = []
    exposure_scores = []
    r2_scores = []
    for stats in regression_stats.values():
        betas = stats.get("factor_betas", {})
        pvals = stats.get("factor_pvalues", {})
        if factor_name not in betas:
            continue
        beta = abs(float(betas.get(factor_name, 0.0)))
        pval = float(pvals.get(factor_name, 1.0))
        p_scores.append(_pvalue_evidence_score(pval))
        exposure_scores.append(np.clip(beta / 1.0, 0.0, 1.0))
        r2_scores.append(np.clip(float(stats.get("adj_r_squared", 0.0)) / 0.80,
                                 0.0, 1.0))
    if not p_scores:
        return None
    raw = (
        0.15 * float(np.median(p_scores))
        + 0.50 * float(np.mean(exposure_scores))
        + 0.35 * float(np.mean(r2_scores))
    )
    return _bounded_confidence(0.15 + 0.70 * raw)


def _blend_view_confidence(manual_confidence, model_confidence):
    """Blend manual confidence with automatically estimated model confidence."""
    manual = _bounded_confidence(manual_confidence)
    if model_confidence is None:
        return manual
    model = _bounded_confidence(model_confidence)
    weight_model = float(MODEL_CONFIDENCE_WEIGHT)
    weight_manual = float(MANUAL_CONFIDENCE_WEIGHT)
    total = weight_model + weight_manual
    if total <= 0:
        return manual
    blended = (weight_model * model + weight_manual * manual) / total
    return _bounded_confidence(blended)


def _project_zero_mean_bounded(values, bound):
    """Project a finite cross-section onto sum=0 with symmetric hard bounds."""
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    result = pd.Series(0.0, index=series.index, dtype=float)
    if finite.empty:
        return result
    array = finite.to_numpy(dtype=float)
    lower = float(array.min() - bound)
    upper = float(array.max() + bound)
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        mean = float(np.clip(array - midpoint, -bound, bound).mean())
        if mean > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(array - 0.5 * (lower + upper), -bound, bound)
    result.loc[finite.index] = projected
    if abs(float(result.loc[finite.index].mean())) > 1e-12:
        raise RuntimeError("Automatic-alpha zero-mean projection failed")
    if float(result.abs().max()) > bound + 1e-12:
        raise RuntimeError("Automatic-alpha projection exceeded its hard cap")
    return result


def _robust_automatic_alpha_score(tickers, regression_stats):
    """Convert current as-of WLS/HAC alphas into the validated 17A score."""
    raw = pd.Series({
        ticker: regression_stats.get(ticker, {}).get(
            "alpha_annualized", np.nan)
        for ticker in tickers
    }, dtype=float).reindex(tickers)
    valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
    score = pd.Series(0.0, index=tickers, dtype=float)
    if len(valid) < 3:
        return score, raw
    lower = float(valid.quantile(AUTO_ALPHA_WINSOR_LOWER))
    upper = float(valid.quantile(AUTO_ALPHA_WINSOR_UPPER))
    clipped = valid.clip(lower=lower, upper=upper)
    median = float(clipped.median())
    mad = float((clipped - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        return score, raw
    standardized = (clipped - median) / scale
    standardized = _project_zero_mean_bounded(
        standardized, AUTO_ALPHA_SIGNAL_Z_CLIP)
    score.loc[standardized.index] = standardized
    return score, raw


def _build_runtime_automatic_alpha_calibration(tickers, decision_date):
    """Rebuild the two-year, causal five-day alpha calibration in memory.

    Historical signal dates are sampled five sessions apart. For each t, the
    WLS/HAC score and factor transform use data through t only; its five-day
    realised cross-sectional return is included only when it ended strictly
    before the live decision date. Nothing is persisted after the model run.
    """
    if FREQUENCY not in ("daily", "weekly_avg_daily"):
        raise RuntimeError(
            "Runtime 17A calibration supports the formal daily or "
            "weekly_avg_daily signal frequency only")
    decision = pd.Timestamp(decision_date).tz_localize(None).normalize()
    horizon = int(AUTO_ALPHA_FORECAST_HORIZON_DAYS)
    # Enough data for the earliest three-year regression, its 90-day formal
    # buffer, 104 five-session forecasts, and a conservative price buffer.
    history_days = (DATA_PERIOD_YEARS * 365 + 90
                    + (AUTO_ALPHA_CALIBRATION_MAX_DATES + 8) * horizon * 2
                    + 120)
    download_start = decision - pd.Timedelta(days=history_days)
    download_end = decision + pd.Timedelta(days=1)
    prices = yf.download(
        list(tickers), start=download_start, end=download_end,
        auto_adjust=True, progress=False,
    )["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=list(tickers)[0])
    prices = prices.reindex(columns=list(tickers)).sort_index()
    if prices.dropna(how="all").empty:
        raise RuntimeError("Runtime 17A calibration received no price history")
    raw_daily_returns = prices.pct_change(fill_method=None)
    trading_dates = raw_daily_returns.dropna(how="all").index
    if len(trading_dates) < horizon + 2:
        raise RuntimeError("Runtime 17A calibration has too few trading sessions")

    # Raw factors are fetched once. Per-date construction below is explicitly
    # capped at that date, preventing smoothing/residualization look-ahead.
    daily_inputs = fetch_daily_factor_inputs(download_start, download_end,
                                             verbose=False)
    last_signal_position = len(trading_dates) - horizon - 1
    entries = []
    skipped = 0
    for position in range(last_signal_position, -1, -horizon):
        as_of = pd.Timestamp(trading_dates[position]).tz_localize(None).normalize()
        horizon_dates = trading_dates[position + 1:position + 1 + horizon]
        if (len(horizon_dates) != horizon
                or pd.Timestamp(horizon_dates.max()) >= decision):
            continue
        model_start = as_of - pd.Timedelta(
            days=DATA_PERIOD_YEARS * 365 + 90)
        history_info = classify_ticker_history(
            prices.loc[:as_of], list(tickers), model_start)
        active = history_info["active_tickers"]
        if len(active) < 3:
            skipped += 1
            continue
        raw_window = raw_daily_returns.loc[model_start:as_of, active]
        signal_returns = pd.DataFrame({
            ticker: _transform_return_series(raw_window[ticker], FREQUENCY)
            for ticker in active
        }).loc[model_start:as_of]
        if signal_returns.dropna(how="all").empty:
            skipped += 1
            continue
        signal_start = signal_returns.dropna(how="all").index.min()
        try:
            factors_as_of = fetch_ff_factors(
                signal_start, as_of, FREQUENCY, verbose=False,
                daily_inputs=daily_inputs)
            factor_names = [name for name in FACTORS_TO_USE
                            if name in factors_as_of.columns]
            if not factor_names:
                raise ValueError("no active formal factors")
            _, regression_stats, _, _, _ = run_factor_regression(
                signal_returns, factors_as_of, factor_names)
        except ValueError:
            # Early dates without a full formal regression are excluded, never
            # replaced with a proxy score.
            skipped += 1
            continue
        score, _ = _robust_automatic_alpha_score(active, regression_stats)
        five_day = ((1.0 + raw_daily_returns.reindex(horizon_dates)[active])
                    .prod(axis=0, min_count=horizon) - 1.0)
        valid = five_day.dropna()
        if len(valid) < 3:
            skipped += 1
            continue
        five_day.loc[valid.index] = valid - float(valid.mean())
        entries.append({
            "signal_date": as_of,
            "horizon_end": pd.Timestamp(horizon_dates.max()).tz_localize(None).normalize(),
            "score": score.to_dict(),
            "realized_residual_5d": five_day.to_dict(),
        })
        if len(entries) >= AUTO_ALPHA_CALIBRATION_MAX_DATES:
            break

    entries.reverse()
    if len(entries) < AUTO_ALPHA_MIN_CALIBRATION_DATES:
        raise RuntimeError(
            "Runtime 17A calibration has too few mature signal dates: "
            f"{len(entries)} < {AUTO_ALPHA_MIN_CALIBRATION_DATES} "
            f"(skipped={skipped})")

    numerator = denominator = 0.0
    observations = 0
    rank_ics, sign_hits = [], []
    for entry in entries:
        aligned = pd.concat([
            pd.Series(entry["score"], dtype=float).rename("x"),
            pd.Series(entry["realized_residual_5d"], dtype=float).rename("y"),
        ], axis=1).dropna()
        if len(aligned) < 3:
            continue
        x_values = aligned["x"].to_numpy(dtype=float)
        y_values = aligned["y"].to_numpy(dtype=float)
        # Each mature decision date has equal weight, not each individual name.
        numerator += float(np.mean(x_values * y_values))
        denominator += float(np.mean(x_values ** 2))
        observations += len(aligned)
        rank_ic = aligned["x"].corr(aligned["y"], method="spearman")
        if np.isfinite(rank_ic):
            rank_ics.append(float(rank_ic))
        nonzero = np.abs(x_values) > 1e-12
        if np.any(nonzero):
            sign_hits.extend((np.sign(x_values[nonzero])
                              == np.sign(y_values[nonzero])).tolist())
    if denominator <= 1e-12 or observations <= 0:
        raise RuntimeError("Runtime 17A calibration has no usable score/outcome relation")
    unconstrained_slope = float(numerator / denominator)
    slope_5d = max(unconstrained_slope, 0.0) * (
        len(entries) / (len(entries) + AUTO_ALPHA_CALIBRATION_PRIOR_DATES))
    maturity_end = max(entry["horizon_end"] for entry in entries)
    if maturity_end >= decision:
        raise RuntimeError("Runtime 17A calibration violated maturity boundary")
    return {
        "version": "17A-calibration-v2-runtime",
        "decision_date": str(decision.date()),
        "latest_matured_horizon_end": str(maturity_end.date()),
        "calibration_dates": int(len(entries)),
        "calibration_observations": int(observations),
        "slope_5d": float(slope_5d),
        "unconstrained_slope_5d": float(unconstrained_slope),
        "rank_ic": float(np.mean(rank_ics)) if rank_ics else np.nan,
        "sign_accuracy": float(np.mean(sign_hits)) if sign_hits else np.nan,
        "alpha_cap": float(AUTO_ALPHA_MAX_ANNUAL_TILT),
        "view_confidence": float(AUTO_PEER_VIEW_CONFIDENCE),
        "factor_model_id": FACTOR_MODEL_ID,
        "storage": "memory_only",
    }


def _load_automatic_alpha_calibration(decision_date):
    """Validate this run's causal in-memory calibration state or fail closed."""
    payload = _RUNTIME_AUTO_ALPHA_CALIBRATION
    if payload is None:
        raise RuntimeError(
            "17A automatic views enabled but runtime calibration was not rebuilt")
    required = {
        "version", "decision_date", "latest_matured_horizon_end",
        "calibration_dates", "calibration_observations", "slope_5d",
        "alpha_cap", "view_confidence", "factor_model_id",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(
            f"17A calibration file is missing fields: {missing}")
    if payload["version"] != "17A-calibration-v2-runtime":
        raise RuntimeError("Unsupported 17A calibration version")
    if payload["factor_model_id"] != FACTOR_MODEL_ID:
        raise RuntimeError(
            "17A calibration factor model differs from production; "
            "refit a causally matured calibration for the active factor model")
    if int(payload["calibration_dates"]) < AUTO_ALPHA_MIN_CALIBRATION_DATES:
        raise RuntimeError("17A calibration has too few matured decision dates")
    if int(payload["calibration_observations"]) <= 0:
        raise RuntimeError("17A calibration has no matured observations")
    slope = float(payload["slope_5d"])
    if not np.isfinite(slope) or slope < 0.0:
        raise RuntimeError("17A calibration slope is invalid")
    if not np.isclose(float(payload["alpha_cap"]),
                      AUTO_ALPHA_MAX_ANNUAL_TILT):
        raise RuntimeError("17A calibration alpha cap differs from production")
    if not np.isclose(float(payload["view_confidence"]),
                      AUTO_PEER_VIEW_CONFIDENCE):
        raise RuntimeError("17A calibration confidence differs from production")
    decision = pd.Timestamp(decision_date).tz_localize(None).normalize()
    calibration_date = pd.Timestamp(
        payload["decision_date"]).tz_localize(None).normalize()
    maturity_end = pd.Timestamp(
        payload["latest_matured_horizon_end"]).tz_localize(None).normalize()
    if calibration_date != decision:
        raise RuntimeError(
            "17A runtime calibration decision date does not match this model run")
    if maturity_end >= calibration_date:
        raise RuntimeError(
            "17A calibration violates the strict maturity boundary")
    return payload


def _build_automatic_peer_alpha_components(
        tickers, pi, Sigma_annual, regression_stats, decision_date):
    """Build the validated 17A peer-relative P/Q/confidence components."""
    calibration = _load_automatic_alpha_calibration(decision_date)
    score, raw_alpha = _robust_automatic_alpha_score(
        tickers, regression_stats)
    annual_unbounded = (
        score * float(calibration["slope_5d"]) * 252.0
        / AUTO_ALPHA_FORECAST_HORIZON_DAYS
    )
    calibrated_alpha = _project_zero_mean_bounded(
        annual_unbounded, AUTO_ALPHA_MAX_ANNUAL_TILT)
    if float(calibrated_alpha.abs().max()) <= 1e-14:
        return (np.empty((0, len(tickers))), np.empty(0), np.empty(0), {
            "enabled": True, "view_count": 0, "selected_views": {},
            "raw_alpha": raw_alpha.to_dict(),
            "standardized_score": score.to_dict(),
            "calibrated_alpha": calibrated_alpha.to_dict(),
            "calibration": calibration,
        })

    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    rows, selected = [], {}
    peer_members = [
        ticker for members in AUTO_FIXED_PEER_GROUPS.values()
        for ticker in members
    ]
    duplicates = sorted({ticker for ticker in peer_members
                         if peer_members.count(ticker) > 1})
    unknown = sorted(set(peer_members) - set(TICKERS))
    if duplicates or unknown:
        raise RuntimeError(
            "Invalid 17A peer taxonomy: "
            f"duplicates={duplicates}, unknown={unknown}")

    for group, prespecified_members in AUTO_FIXED_PEER_GROUPS.items():
        members = [ticker for ticker in prespecified_members
                   if ticker in ticker_index]
        if len(members) < AUTO_PEER_MIN_GROUP_SIZE:
            continue
        relative = {}
        for ticker in members:
            peers = [peer for peer in members if peer != ticker]
            relative[ticker] = float(
                calibrated_alpha[ticker] - calibrated_alpha[peers].mean())
        chosen = sorted(
            members,
            key=lambda ticker: (-abs(relative[ticker]),
                                prespecified_members.index(ticker)),
        )[:AUTO_PEER_VIEWS_PER_GROUP]
        for ticker in chosen:
            peers = [peer for peer in members if peer != ticker]
            row = np.zeros(len(tickers), dtype=float)
            row[ticker_index[ticker]] = 1.0
            row[[ticker_index[peer] for peer in peers]] = -1.0 / len(peers)
            if not np.isclose(row.sum(), 0.0, rtol=0.0, atol=1e-13):
                raise RuntimeError("17A automatic peer row is not zero-sum")
            rows.append(row)
            selected[group] = {
                "ticker": ticker, "peers": peers,
                "incremental_spread": float(relative[ticker]),
            }

    if not rows:
        raise RuntimeError("17A automatic views enabled but no peer view was built")
    P = np.asarray(rows, dtype=float)
    alpha = calibrated_alpha.reindex(tickers).to_numpy(dtype=float)
    Q = P @ (np.asarray(pi, dtype=float) + alpha)
    confidence = np.full(len(P), AUTO_PEER_VIEW_CONFIDENCE, dtype=float)
    incremental = Q - P @ np.asarray(pi, dtype=float)
    if not np.allclose(incremental, P @ alpha, rtol=0.0, atol=1e-13):
        raise RuntimeError("17A automatic Q is not P@(pi+alpha)")
    return P, Q, confidence, {
        "enabled": True, "view_count": int(len(P)),
        "selected_views": selected,
        "raw_alpha": raw_alpha.to_dict(),
        "standardized_score": score.to_dict(),
        "calibrated_alpha": calibrated_alpha.to_dict(),
        "calibration": calibration,
        "alpha_cap_binding": bool(np.isclose(
            float(calibrated_alpha.abs().max()),
            AUTO_ALPHA_MAX_ANNUAL_TILT, atol=1e-12)),
        "alpha_mean_abs": float(calibrated_alpha.abs().mean()),
        "alpha_max_abs": float(calibrated_alpha.abs().max()),
        "confidence": AUTO_PEER_VIEW_CONFIDENCE,
    }


def _build_view_uncertainty(P, confidence_vals, Sigma_annual):
    """Build a diagonal Idzorek-style view-error covariance.

    The model keeps each view's confidence-derived marginal uncertainty, but
    does not infer off-diagonal forecast-error covariance from PΣP'.  View
    exposure correlation is not a substitute for observed forecast-error
    correlation, so Omega is intentionally diagonal until a separately
    validated causal error-covariance model is approved.
    """
    if len(P) == 0:
        raise ValueError("At least one BL view is required to build Omega")

    # Idzorek边际不确定性: ω_i = ((1-c)/c) * P_i Σ P_i'.
    # PΣP' 仅用于每个 view 的边际尺度；不把其非对角暴露协方差解释为
    # 未观察到的 forecast-error covariance。
    view_cov = P @ Sigma_annual @ P.T
    Omega_diag = []
    for i, conf in enumerate(confidence_vals):
        conf = np.clip(conf, 0.01, 0.99)
        view_var = max(float(view_cov[i, i]), 1e-12)
        omega_i = ((1.0 - conf) / conf) * view_var
        Omega_diag.append(omega_i)

    Omega = np.diag(np.maximum(Omega_diag, 1e-12))
    # Preserve a scale-aware numerical ridge even though the diagonal Idzorek
    # construction is already positive definite.
    ridge = max(float(np.mean(Omega_diag)), 1e-12) * 1e-8
    return Omega + np.eye(len(Omega)) * ridge


def _build_manual_view_components(tickers, factor_names, B, asset_views,
                                  factor_views, regression_stats=None,
                                  verbose=True):
    """Build manual view rows and track their direct asset coverage."""
    n_assets = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    factor_idx = {f: i for i, f in enumerate(factor_names)}

    P_rows = []
    Q_vals = []
    confidence_vals = []
    manual_asset_confidence = np.zeros(n_assets, dtype=float)

    for view in asset_views:
        view_type = view[0]
        if view_type == "absolute":
            _, ticker, expected_return, confidence = view
            model_conf = _asset_model_confidence(ticker, regression_stats)
            final_conf = _blend_view_confidence(confidence, model_conf)
            row = np.zeros(n_assets)
            row[ticker_idx[ticker]] = 1.0
            P_rows.append(row)
            Q_vals.append(expected_return)
            confidence_vals.append(final_conf)
            manual_asset_confidence[ticker_idx[ticker]] = max(
                manual_asset_confidence[ticker_idx[ticker]], final_conf
            )
            if verbose:
                print(f"  View: {ticker} expected excess return "
                      f"{expected_return:+.1%} "
                      f"(confidence {final_conf:.0%})")

        elif view_type == "relative":
            _, (long_ticker, short_ticker), spread, confidence = view
            model_confs = [
                _asset_model_confidence(long_ticker, regression_stats),
                _asset_model_confidence(short_ticker, regression_stats),
            ]
            valid_model_confs = [c for c in model_confs if c is not None]
            model_conf = (float(np.mean(valid_model_confs))
                          if valid_model_confs else None)
            final_conf = _blend_view_confidence(confidence, model_conf)
            row = np.zeros(n_assets)
            row[ticker_idx[long_ticker]] = 1.0
            row[ticker_idx[short_ticker]] = -1.0
            P_rows.append(row)
            Q_vals.append(spread)
            confidence_vals.append(final_conf)
            for ticker in (long_ticker, short_ticker):
                manual_asset_confidence[ticker_idx[ticker]] = max(
                    manual_asset_confidence[ticker_idx[ticker]], final_conf
                )
            if verbose:
                print(f"  View: {long_ticker} outperforms {short_ticker} "
                      f"by {spread:+.1%} (confidence {final_conf:.0%})")

    for view in factor_views:
        _validate_factor_view(view)
        view_type = view[0]
        if view_type == "absolute":
            _, factor_name, factor_return, confidence = view
            model_conf = _factor_model_confidence(factor_name,
                                                  regression_stats)
            final_conf = _blend_view_confidence(confidence, model_conf)
            f_idx = factor_idx[factor_name]
            row = B[:, f_idx].copy()
            P_rows.append(row)
            Q_vals.append(factor_return)
            confidence_vals.append(final_conf)

            if verbose:
                label = _factor_view_label(factor_name)
                print(f"  Factor view: {factor_name} ({label}) "
                      f"annual return {factor_return:+.1%} "
                      f"(confidence {final_conf:.0%})")

        elif view_type == "relative":
            _, (long_factor, short_factor), spread, confidence = view
            model_confs = [
                _factor_model_confidence(long_factor, regression_stats),
                _factor_model_confidence(short_factor, regression_stats),
            ]
            valid_model_confs = [c for c in model_confs if c is not None]
            model_conf = (float(np.mean(valid_model_confs))
                          if valid_model_confs else None)
            final_conf = _blend_view_confidence(confidence, model_conf)
            long_idx = factor_idx[long_factor]
            short_idx = factor_idx[short_factor]
            row = B[:, long_idx] - B[:, short_idx]
            P_rows.append(row)
            Q_vals.append(spread)
            confidence_vals.append(final_conf)

            if verbose:
                long_label = _factor_view_label(long_factor)
                short_label = _factor_view_label(short_factor)
                print(f"  Factor view: {long_factor} ({long_label}) outperforms "
                      f"{short_factor} ({short_label}) annually by {spread:+.1%} "
                      f"(confidence {final_conf:.0%})")

    return P_rows, Q_vals, confidence_vals, manual_asset_confidence


def build_views(tickers, factor_names, B, asset_views, factor_views,
                Sigma_annual, regression_stats=None, verbose=True):
    """Build the manual BL view matrix."""
    P_rows, Q_vals, confidence_vals, _ = _build_manual_view_components(
        tickers, factor_names, B, asset_views, factor_views,
        regression_stats=regression_stats, verbose=verbose
    )
    if not P_rows:
        raise ValueError("At least one manual asset or factor view is required")
    P = np.asarray(P_rows, dtype=float)
    Q = np.asarray(Q_vals, dtype=float)
    return P, Q, _build_view_uncertainty(P, confidence_vals, Sigma_annual)


def build_production_views(tickers, factor_names, B, pi, Sigma_annual,
                           regression_stats, decision_date,
                           asset_views=None, factor_views=None,
                           verbose=True):
    """Combine validated 17A automatic views and optional manual views."""
    asset_views = ASSET_VIEWS if asset_views is None else asset_views
    factor_views = FACTOR_VIEWS if factor_views is None else factor_views
    P_parts, Q_parts, confidence_parts = [], [], []
    diagnostics = {
        "automatic_peer_alpha": {"enabled": False, "view_count": 0},
        "manual_view_count": 0,
    }

    if AUTO_PEER_ALPHA_VIEWS_ENABLED:
        P_auto, Q_auto, confidence_auto, auto_diagnostics = (
            _build_automatic_peer_alpha_components(
                tickers, pi, Sigma_annual, regression_stats, decision_date))
        diagnostics["automatic_peer_alpha"] = auto_diagnostics
        if len(P_auto):
            P_parts.append(P_auto)
            Q_parts.append(Q_auto)
            confidence_parts.append(confidence_auto)
        if verbose:
            calibration = auto_diagnostics.get("calibration", {})
            print(
                "  Automatic 17A peer-alpha views: "
                f"{auto_diagnostics.get('view_count', 0)}; "
                f"alpha cap={AUTO_ALPHA_MAX_ANNUAL_TILT:.0%}; "
                f"confidence={AUTO_PEER_VIEW_CONFIDENCE:.0%}; "
                f"calibration through {calibration.get('decision_date', '-')}")
            for group, view in auto_diagnostics.get(
                    "selected_views", {}).items():
                peers = ",".join(view["peers"])
                print(
                    f"    {group}: {view['ticker']} vs [{peers}], "
                    f"incremental spread={view['incremental_spread']:+.2%}")

    if asset_views or factor_views:
        manual_rows, manual_q, manual_confidence, _ = (
            _build_manual_view_components(
                tickers, factor_names, B, asset_views, factor_views,
                regression_stats=regression_stats, verbose=verbose))
        diagnostics["manual_view_count"] = int(len(manual_rows))
        if manual_rows:
            P_parts.append(np.asarray(manual_rows, dtype=float))
            Q_parts.append(np.asarray(manual_q, dtype=float))
            confidence_parts.append(np.asarray(manual_confidence, dtype=float))

    if not P_parts:
        return None, None, None, diagnostics
    P = np.vstack(P_parts)
    Q = np.concatenate(Q_parts)
    confidence = np.concatenate(confidence_parts)
    Omega = _build_view_uncertainty(P, confidence, Sigma_annual)
    diagnostics["total_view_count"] = int(len(P))
    return P, Q, Omega, diagnostics


def black_litterman(pi, Sigma_annual, P, Q, Omega):
    """
    BL posterior:
      μ_BL = [Σ⁻¹ + P'Ω⁻¹P]⁻¹ · [Σ⁻¹π + P'Ω⁻¹Q]
    Use np.linalg.solve instead of inv for improved numerical stability.
    """
    n = len(pi)
    # solve(A, b) = A⁻¹b, 比显式求逆更稳定
    Sigma_inv_pi = np.linalg.solve(Sigma_annual, pi)
    Sigma_inv = np.linalg.solve(Sigma_annual, np.eye(n))

    # Omega 可以包含重叠观点的非对角协方差，因此完整求解其逆作用。
    Omega_inv = np.linalg.solve(Omega, np.eye(Omega.shape[0]))

    posterior_precision = Sigma_inv + P.T @ Omega_inv @ P

    posterior_mean = np.linalg.solve(
        posterior_precision,
        Sigma_inv_pi + P.T @ Omega_inv @ Q
    )

    return posterior_mean


def _window_return(series, window):
    """Return cumulative return over the latest window, using available observations if needed."""
    clean = series.dropna()
    if clean.empty:
        return 0.0, 0
    use = clean.iloc[-min(window, len(clean)):]
    return float((1.0 + use).prod() - 1.0), len(use)


def _recent_drawdown(series, lookback):
    """Return current drawdown from the highest wealth level in the lookback."""
    clean = series.dropna()
    if clean.empty:
        return 0.0
    use = clean.iloc[-min(lookback, len(clean)):]
    wealth = (1.0 + use).cumprod()
    peak = float(wealth.max())
    current = float(wealth.iloc[-1])
    return float(current / peak - 1.0) if peak > 0 else 0.0


def _regime_benchmark_columns(returns):
    """Prefer SPY/QQQ and otherwise fall back to the first two available columns."""
    if returns is None or returns.empty:
        return []
    cols = [c for c in REGIME_BENCHMARKS if c in returns.columns]
    if not cols:
        cols = list(returns.columns[:2])
    return cols


def _direction_chop_metrics(benchmark_returns, lookback):
    """
    Measure risk-appetite breadth and choppiness using SPY/QQQ co-moves and
    direction-switch frequency. Higher flips indicate more frequent alternation
    between recent gains and losses.
    """
    cols = _regime_benchmark_columns(benchmark_returns)
    if not cols:
        return {"flips": 0.50, "both_up": 0.0, "both_down": 0.0}

    use = benchmark_returns[cols].dropna(how="all")
    use = use.iloc[-min(lookback, len(use)):]
    if len(use) < 3:
        return {"flips": 0.50, "both_up": 0.0, "both_down": 0.0}

    if len(cols) >= 2:
        both_up = ((use[cols[0]] > 0) & (use[cols[1]] > 0)).mean()
        both_down = ((use[cols[0]] < 0) & (use[cols[1]] < 0)).mean()
    else:
        both_up = (use[cols[0]] > 0).mean()
        both_down = (use[cols[0]] < 0).mean()

    avg_ret = use.mean(axis=1)
    signs = np.sign(avg_ret.values)
    signs[signs == 0] = 1
    flips = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0

    return {
        "flips": float(flips),
        "both_up": float(both_up),
        "both_down": float(both_down),
    }


def _latest_intraperiod_metrics(benchmark_returns):
    """Read daily structural features within the latest five-trading-day period."""
    defaults = {
        "intraperiod_up_ratio": 0.50,
        "intraperiod_flips": 0.50,
        "intraperiod_both_up": 0.0,
        "intraperiod_both_down": 0.0,
        "intraperiod_vol": 0.0,
        "intraperiod_range": 0.0,
        "intraperiod_vol_shift": 1.0,
    }
    if benchmark_returns is None or benchmark_returns.empty:
        return defaults

    latest = benchmark_returns.iloc[-1]
    metrics = {}
    for key, default in defaults.items():
        col = "__" + key
        value = latest.get(col, default)
        metrics[key] = float(value) if pd.notna(value) else default
    return metrics


def _regime_market_daily_returns(raw_benchmark_returns):
    """Return equal-weight daily SPY/QQQ market returns for regime paths."""
    cols = _regime_benchmark_columns(raw_benchmark_returns)
    if not cols:
        return pd.Series(dtype=float)
    return raw_benchmark_returns[cols].dropna(how="any").mean(axis=1)


def _recent_risk_off_windows(market_daily_returns):
    """Find rolling five-day losses of at least 2% within the latest ten days."""
    market = market_daily_returns.dropna()
    tail = market.iloc[-REGIME_RECOVERY_LOOKBACK_DAYS:]
    windows = []
    for end in range(4, len(tail)):
        window = tail.iloc[end - 4:end + 1]
        cumulative_return = float((1.0 + window).prod() - 1.0)
        if cumulative_return <= REGIME_RISK_OFF_FIVE_DAY_RETURN:
            windows.append({
                "start": int(market.index.get_loc(window.index[0])),
                "end": int(market.index.get_loc(window.index[-1])),
                "return": cumulative_return,
            })
    return windows


def _recovery_path_from_recent_risk_off(market_daily_returns):
    """Identify a recent Bearish window followed by a visible upward path."""
    market = market_daily_returns.dropna()
    inactive = {
        "has_event": False,
        "post_up": False,
        "below_pre_peak": False,
        "reclaimed_peak": False,
        "post_return": 0.0,
    }
    eligible = [event for event in _recent_risk_off_windows(market)
                if event["end"] < len(market) - 1]
    if not eligible:
        return inactive

    event = eligible[-1]
    post = market.iloc[event["end"] + 1:]
    post_return = float((1.0 + post).prod() - 1.0)
    post_up_ratio = float((post > 0.0).mean()) if len(post) else 0.0
    post_up = bool(post_return > 0.0 and post_up_ratio >= 0.50)

    wealth = (1.0 + market).cumprod()
    pre_window = wealth.iloc[:event["start"] + 1]
    pre_peak = float(pre_window.max()) if len(pre_window) else 1.0
    current = float(wealth.iloc[-1])
    return {
        "has_event": True,
        "post_up": post_up,
        "below_pre_peak": bool(current < pre_peak - 1e-12),
        "reclaimed_peak": bool(current >= pre_peak - 1e-12),
        "post_return": post_return,
    }


def _classify_regime_state(metrics, market_daily_returns, previous_state):
    """Classify one state, allowing every prior state to enter Bearish."""
    market = market_daily_returns.dropna()
    if len(market) < REGIME_TRADING_DAYS_PER_PERIOD:
        return "Neutral", _recovery_path_from_recent_risk_off(market), 0.0

    five_day_return = float((1.0 + market.iloc[-REGIME_TRADING_DAYS_PER_PERIOD:]).prod() - 1.0)
    recovery_path = _recovery_path_from_recent_risk_off(market)
    choppy = bool(metrics.get("intraperiod_flips", 0.50) >= 0.50)

    if five_day_return <= REGIME_RISK_OFF_FIVE_DAY_RETURN:
        return "Bearish", recovery_path, five_day_return
    if previous_state == "Bearish" and five_day_return < 0.0:
        return "Bearish", recovery_path, five_day_return
    if choppy:
        return "Neutral", recovery_path, five_day_return
    if recovery_path["has_event"] and recovery_path["post_up"]:
        if recovery_path["below_pre_peak"]:
            return "Neutral", recovery_path, five_day_return
        if recovery_path["reclaimed_peak"]:
            return "Bullish", recovery_path, five_day_return
    if five_day_return > 0.0:
        return "Bullish", recovery_path, five_day_return
    return "Neutral", recovery_path, five_day_return


def _market_signal_probabilities(metrics, deterministic_state, recovery_path,
                                 five_day_return):
    """Dirichlet-smoothed current market-signal probabilities."""
    vol_ratio = metrics["vol_short"] / max(metrics["vol_medium"], 1e-8)
    intraday_flips = float(metrics.get("intraperiod_flips", 0.50))
    intraday_up_ratio = float(metrics.get("intraperiod_up_ratio", 0.50))
    intraday_both_up = float(metrics.get("intraperiod_both_up", 0.0))
    intraday_both_down = float(metrics.get("intraperiod_both_down", 0.0))
    intraday_vol = float(metrics.get("intraperiod_vol", 0.0))
    intraday_range = float(metrics.get("intraperiod_range", 0.0))
    intraday_vol_shift = float(metrics.get("intraperiod_vol_shift", 1.0))
    uneven_vol = intraday_vol_shift >= 2.0 or intraday_vol_shift <= 0.50

    recovering = bool(recovery_path["has_event"]
                      and recovery_path["post_up"]
                      and recovery_path["below_pre_peak"])
    reclaimed = bool(recovery_path["has_event"]
                     and recovery_path["post_up"]
                     and recovery_path["reclaimed_peak"])
    active_bearish = five_day_return <= REGIME_RISK_OFF_FIVE_DAY_RETURN
    scores = {state: 0.0 for state in REGIME_STATES}

    scores["Bullish"] += 1.2 * max(metrics["short_return"], 0.0) / 0.06
    scores["Bullish"] += 0.9 * max(metrics["medium_return"], 0.0) / 0.10
    scores["Bullish"] += 0.8 * metrics["both_up"]
    scores["Bullish"] += 0.5 if intraday_up_ratio >= 0.70 else 0.0
    scores["Bullish"] += 0.4 if intraday_flips <= 0.25 else 0.0
    scores["Bullish"] += 0.4 if intraday_both_up >= 0.50 else 0.0
    scores["Bullish"] += 0.6 if not recovering else 0.0
    scores["Bullish"] += 0.8 if reclaimed else 0.0
    scores["Bullish"] += 0.4 if vol_ratio < 1.10 else 0.0
    if recovering:
        scores["Bullish"] *= 0.55

    scores["Bearish"] += 1.3 * max(-metrics["short_return"], 0.0) / 0.06
    scores["Bearish"] += 1.0 * max(-metrics["medium_return"], 0.0) / 0.10
    scores["Bearish"] += 0.9 * metrics["both_down"]
    scores["Bearish"] += 0.4 if intraday_both_down >= 0.40 else 0.0
    scores["Bearish"] += 0.3 if intraday_vol >= 0.20 else 0.0
    scores["Bearish"] += 0.8 if metrics["drawdown"] < -0.10 else 0.0
    scores["Bearish"] += 0.5 if vol_ratio > 1.25 else 0.0
    scores["Bearish"] += 0.8 if active_bearish else 0.0
    scores["Bearish"] += 0.4 if (
        deterministic_state == "Bearish" and five_day_return < 0.0) else 0.0

    # 尚未收复前高的反弹属于 Neutral；保留原有路径证据，但不再
    # 生成独立 Recovery 后验状态。
    if recovering:
        scores["Neutral"] += 1.4
        scores["Neutral"] += 0.9
        scores["Neutral"] += 1.2 * min(
            max(recovery_path["post_return"] / 0.04, 0.0), 1.0)
        scores["Neutral"] += 0.8 * max(five_day_return, 0.0) / 0.06
        scores["Neutral"] += 0.5 * metrics["both_up"]
        scores["Neutral"] += 0.3 if vol_ratio < 1.25 else 0.0

    trend_strength = (abs(metrics["short_return"])
                      + 0.5 * abs(metrics["medium_return"]))
    scores["Neutral"] += 1.2 * metrics["flips"]
    scores["Neutral"] += 0.7 if intraday_flips >= 0.50 else 0.0
    scores["Neutral"] += 0.5 if intraday_range >= 0.035 else 0.0
    scores["Neutral"] += 0.3 if uneven_vol else 0.0
    scores["Neutral"] += 0.8 if trend_strength < 0.04 else 0.0
    scores["Neutral"] += 0.5 if (
        abs(metrics["both_up"] - metrics["both_down"]) < 0.15) else 0.0
    scores["Neutral"] += 0.4 if 0.95 <= vol_ratio <= 1.25 else 0.0

    score_arr = np.array([scores[state] for state in REGIME_STATES], dtype=float)
    score_arr = np.clip(score_arr, 0.0, 6.0)
    # One symmetric Dirichlet pseudocount per state prevents a zero-probability
    # signal while preserving the rule-score ordering without softmax extremes.
    signal_mass = 1.0 + score_arr
    return pd.Series(signal_mass / signal_mass.sum(), index=REGIME_STATES)


def _regime_metrics_from_history(benchmark_returns, ann_factor):
    """Calculate rolling market-path metrics required for the current regime assessment."""
    cols = _regime_benchmark_columns(benchmark_returns)
    if not cols:
        return None
    bench = benchmark_returns[cols].dropna(how="all").mean(axis=1)
    if bench.empty:
        return None

    short_ret, _ = _window_return(bench, REGIME_SHORT_WINDOW_PERIODS)
    medium_ret, _ = _window_return(bench, REGIME_MEDIUM_WINDOW_PERIODS)
    short_slice = bench.iloc[-min(REGIME_SHORT_WINDOW_PERIODS, len(bench)):]
    medium_slice = bench.iloc[-min(REGIME_MEDIUM_WINDOW_PERIODS, len(bench)):]
    vol_short = float(short_slice.std() * np.sqrt(ann_factor))
    vol_medium = float(medium_slice.std() * np.sqrt(ann_factor))
    drawdown = _recent_drawdown(bench, REGIME_MEDIUM_WINDOW_PERIODS)
    chop = _direction_chop_metrics(
        benchmark_returns[cols], REGIME_SHORT_WINDOW_PERIODS)

    metrics = {
        "short_return": short_ret,
        "medium_return": medium_ret,
        "vol_short": vol_short,
        "vol_medium": vol_medium,
        "drawdown": drawdown,
    }
    metrics.update(chop)
    metrics.update(_latest_intraperiod_metrics(benchmark_returns))
    return metrics


def _benchmark_regime_series(benchmark_returns, raw_benchmark_returns):
    """Generate historical states using only the raw daily path available then."""
    if benchmark_returns is None or benchmark_returns.empty:
        return pd.Series(dtype=object)
    if raw_benchmark_returns is None or raw_benchmark_returns.empty:
        return pd.Series(dtype=object)

    states = []
    idx = []
    previous_state = "Neutral"
    for end_date in benchmark_returns.index:
        block_history = benchmark_returns.loc[:end_date]
        metrics = _regime_metrics_from_history(
            block_history, REGIME_PERIODS_PER_YEAR)
        if metrics is None:
            continue
        market_history = _regime_market_daily_returns(
            raw_benchmark_returns.loc[:end_date])
        state, _, _ = _classify_regime_state(
            metrics, market_history, previous_state)
        states.append(state)
        idx.append(end_date)
        previous_state = state

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
    signal = signal_probs.reindex(REGIME_STATES).fillna(
        1.0 / len(REGIME_STATES))
    total_weight = (REGIME_MARKOV_HISTORY_WEIGHT
                    + REGIME_LATEST_SIGNAL_WEIGHT)
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
        value += float(probs.get(state, 0.0)) * float(
            mapping.get(state, default))
    return value


def compute_regime_context(stock_returns, benchmark_returns=None,
                           raw_benchmark_returns=None):
    """
    Return market state and dynamic parameters that enter the optimizer:
      cash buffer / target exposure
      fixed covariance blend
    """
    if not USE_REGIME_AWARE_OPTIMIZATION:
        return {
            "state": "Neutral",
            "cash_buffer": 0.0,
            "base_cash_buffer": 0.0,
            "target_exposure": WEIGHT_TARGET,
            "recent_covariance_weight": RECENT_COVARIANCE_WEIGHT,
            "transition_probs": None,
            "prior_probs": None,
            "signal_probs": None,
            "posterior_probs": None,
            "metrics": {},
        }

    source = benchmark_returns if benchmark_returns is not None else stock_returns
    raw_source = (raw_benchmark_returns if raw_benchmark_returns is not None
                  else source)
    regime_series = _benchmark_regime_series(source, raw_source)
    transition_probs = _transition_probabilities(regime_series)
    ann_factor = (REGIME_PERIODS_PER_YEAR if benchmark_returns is not None
                  else periods_per_year(FREQUENCY))
    metrics = _regime_metrics_from_history(source, ann_factor)
    previous_state = (str(regime_series.iloc[-2])
                      if len(regime_series) >= 2 else "Neutral")

    if metrics is None:
        metrics = {}
        deterministic_state = "Neutral"
        prior_probs = transition_probs.loc[previous_state]
        signal_probs = pd.Series(1.0 / len(REGIME_STATES), index=REGIME_STATES)
        five_day_return = 0.0
        recovery_path = _recovery_path_from_recent_risk_off(
            _regime_market_daily_returns(raw_source))
    else:
        market_returns = _regime_market_daily_returns(raw_source)
        deterministic_state, recovery_path, five_day_return = \
            _classify_regime_state(metrics, market_returns, previous_state)
        prior_probs = transition_probs.loc[previous_state]
        signal_probs = _market_signal_probabilities(
            metrics, deterministic_state, recovery_path, five_day_return)

    posterior_probs = _blend_regime_probabilities(prior_probs, signal_probs)
    current_state = str(posterior_probs.idxmax())

    cash_buffer = _weighted_regime_value(
        REGIME_CASH_BUFFER, posterior_probs, 0.10)
    cash_buffer = float(np.clip(cash_buffer, 0.0, MAX_CASH_BUFFER))
    target_exposure = float(max(MIN_TARGET_EXPOSURE,
                                WEIGHT_TARGET - cash_buffer))

    return {
        "state": current_state,
        "deterministic_state": deterministic_state,
        "cash_buffer": cash_buffer,
        "base_cash_buffer": cash_buffer,
        "target_exposure": target_exposure,
        "recent_covariance_weight": RECENT_COVARIANCE_WEIGHT,
        "transition_probs": transition_probs,
        "prior_probs": prior_probs,
        "signal_probs": signal_probs,
        "posterior_probs": posterior_probs,
        "metrics": metrics,
        "five_day_market_return": five_day_return,
        "recovery_path": recovery_path,
    }


def get_factor_exposure_cap(factor_name):
    """Return an exposure cap for one factor; None means unrestricted."""
    if not USE_FACTOR_EXPOSURE_CAPS:
        return None
    if factor_name in FACTOR_EXPOSURE_CAPS:
        return FACTOR_EXPOSURE_CAPS[factor_name]
    return DEFAULT_FACTOR_EXPOSURE_CAP


def build_factor_beta_stderr_matrix(tickers, factor_names, regression_stats):
    """Arrange HAC beta standard errors in optimizer asset/factor order."""
    values = []
    for ticker in tickers:
        stats = regression_stats.get(ticker, {})
        stderr_by_factor = stats.get("factor_beta_stderr", {})
        row = []
        for factor_name in factor_names:
            stderr = stderr_by_factor.get(factor_name)
            if stderr is None or not np.isfinite(stderr) or stderr < 0:
                raise ValueError(
                    f"{ticker}/{factor_name} is missing a valid HAC beta standard error"
                )
            row.append(float(stderr))
        values.append(row)
    return np.asarray(values, dtype=float)


# ============================================================
# STEP 5: 组合优化
# ============================================================

def _normalized_portfolio_objective_mode(objective_mode=None):
    """Return the editable optimizer mode label (A or B).

    A uses the BL expected-return prior in a Maximum-Sharpe objective. B uses
    the same constraints, risk matrix, state cash target and universe, but
    minimizes downside semivariance inside the state-implied stock budget.
    """
    mode = PORTFOLIO_OBJECTIVE_MODE if objective_mode is None else objective_mode
    mode = str(mode).strip().upper()
    if mode not in {"A", "B"}:
        raise ValueError(
            "PORTFOLIO_OBJECTIVE_MODE must be 'A' (Max Sharpe) or 'B' "
            "(Minimum Downside Semivariance)"
        )
    return mode


def _minimum_weight_support_masks(reference_weights, target_exposure,
                                  max_single_weight):
    """Return deterministic candidate supports for the 0-or-minimum rule.

    The objectives are nonlinear, while a 0-or-minimum allocation is a
    mixed-integer constraint. The model ranks the unconstrained continuous
    solution and tests every feasible top-k support, selecting the best
    feasible re-solved portfolio. This is reproducible and respects existing
    hard constraints, but is not a global mixed-integer optimum.
    """
    minimum = float(MIN_WEIGHT_IF_HELD)
    target = float(target_exposure)
    maximum = float(max_single_weight)
    n_assets = len(reference_weights)
    if minimum < 0.0 or minimum > maximum + 1e-12:
        raise ValueError(
            "MIN_WEIGHT_IF_HELD must lie in [0, MAX_WEIGHT_SINGLE]")
    if target <= 1e-12 or minimum <= 1e-12:
        return [np.zeros(n_assets, dtype=bool)]
    min_names = int(np.ceil((target - 1e-12) / maximum))
    max_names = int(np.floor((target + 1e-12) / minimum))
    if min_names > max_names or min_names > n_assets:
        raise RuntimeError(
            "The target stock exposure cannot satisfy the configured "
            "0-or-minimum position-size rule")
    reference = np.nan_to_num(np.asarray(reference_weights, dtype=float),
                              nan=-np.inf)
    ranking = np.lexsort((np.arange(n_assets), -reference))
    masks = []
    for count in range(min_names, min(max_names, n_assets) + 1):
        mask = np.zeros(n_assets, dtype=bool)
        mask[ranking[:count]] = True
        masks.append(mask)
    return masks


def _minimum_weight_rule_satisfied(weights, tolerance=1e-7):
    """Check that every nonzero equity allocation is at least the hard floor."""
    minimum = float(MIN_WEIGHT_IF_HELD)
    values = np.asarray(weights, dtype=float)
    if minimum <= tolerance:
        return True
    positive = values > tolerance
    return bool(np.all(values[positive] >= minimum - tolerance))


def portfolio_downside_semivariance(weights, Sigma):
    """Return annual downside semivariance under the supplied risk matrix."""
    weights = np.asarray(weights, dtype=float)
    value = float(weights @ Sigma @ weights)
    return max(value, 0.0)


def optimize_portfolio(mu, Sigma, current_weights=None,
                       B=None, factor_names=None, target_exposure=None,
                       B_stderr=None, fast_mode=False, initial_weights=None,
                       tradable_mask=None, objective_mode=None):
    """Solve either editable constrained portfolio objective.

    Mode A — ``Max Sharpe`` — maximizes BL expected excess return divided by
    downside semideviation. Mode B — ``Minimum Downside Semivariance`` —
    minimizes the same Ledoit-Wolf downside-semivariance risk matrix while
    holding the regime-implied stock exposure fixed. Transaction-cost-aware
    execution is handled separately by :func:`build_min_cost_execution`.
    """
    mode = _normalized_portfolio_objective_mode(objective_mode)
    n = len(mu)
    target = WEIGHT_TARGET if target_exposure is None else float(
        target_exposure)
    target = float(np.clip(target, 0.0, WEIGHT_TARGET))
    max_single = max(0.0, min(float(MAX_WEIGHT_SINGLE), WEIGHT_TARGET))
    if n > 0 and target > n * max_single + 1e-9:
        max_single = min(WEIGHT_TARGET, target / n + 1e-9)
    active_factor_caps = []
    if USE_FACTOR_EXPOSURE_CAPS and B is not None and factor_names is not None:
        beta_stderr = np.asarray(B_stderr, dtype=float)
        if beta_stderr.shape != B.shape or not np.all(np.isfinite(beta_stderr)):
            raise ValueError("Factor caps require a valid HAC beta-standard-error matrix with the same shape as B")
        beta_stderr = np.maximum(beta_stderr, 0.0)
        for idx, factor_name in enumerate(factor_names):
            cap = get_factor_exposure_cap(factor_name)
            if cap is not None:
                active_factor_caps.append(
                    (factor_name, B[:, idx], beta_stderr[:, idx],
                     float(cap)))

    def objective_value(w):
        if mode == "A":
            return portfolio_sharpe(w, mu, Sigma)
        return portfolio_downside_semivariance(w, Sigma)

    def solver_objective(w):
        # SLSQP minimizes. A therefore negates Sharpe; B already minimizes risk.
        return -objective_value(w) if mode == "A" else objective_value(w)

    def factor_cap_constraints_for_w():
        constraints = []
        for _, beta_vec, stderr_vec, cap in active_factor_caps:
            # Long-only weights make this a linear robust counterpart:
            # beta exposure +/- z * sum(w_i * HAC_SE(beta_i)) remains in [-cap, cap].
            buffer_vec = FACTOR_CAP_HAC_SE_Z * stderr_vec
            constraints.append({
                "type": "ineq",
                "fun": lambda w, b=beta_vec, u=buffer_vec, c=cap:
                    c - (w @ b) - (w @ u)
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda w, b=beta_vec, u=buffer_vec, c=cap:
                    c + (w @ b) - (w @ u)
            })
        return constraints

    long_only_constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - target
    }] + factor_cap_constraints_for_w()

    def constraint_is_satisfied(weights, bounds, tolerance=1e-7):
        """Check the optimizer's equality, bounds, and robust factor caps."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,) or not np.all(np.isfinite(weights)):
            return False
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        if (np.any(weights < lower - tolerance)
                or np.any(weights > upper + tolerance)
                or abs(weights.sum() - target) > tolerance):
            return False
        return all(float(constraint["fun"](weights)) >= -tolerance
                   for constraint in long_only_constraints[1:])

    def feasible_anchor(bounds):
        """Find one deterministic point satisfying all linear constraints."""
        A_ub, b_ub = [], []
        for _, beta_vec, stderr_vec, cap in active_factor_caps:
            uncertainty = FACTOR_CAP_HAC_SE_Z * stderr_vec
            A_ub.extend([beta_vec + uncertainty, -beta_vec + uncertainty])
            b_ub.extend([cap, cap])
        result = linprog(
            c=np.zeros(n, dtype=float),
            A_ub=np.asarray(A_ub, dtype=float) if A_ub else None,
            b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
            A_eq=np.ones((1, n), dtype=float),
            b_eq=np.array([target], dtype=float),
            bounds=bounds,
            method="highs",
        )
        if not result.success or not constraint_is_satisfied(result.x, bounds):
            return None
        return np.asarray(result.x, dtype=float)

    def project_reference(reference, anchor, bounds):
        """Project one economic reference portfolio onto the feasible set."""
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (n,) or not np.all(np.isfinite(reference)):
            return None
        result = minimize(
            lambda weights: 0.5 * np.sum((weights - reference) ** 2),
            anchor,
            method="SLSQP",
            bounds=bounds,
            constraints=long_only_constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and constraint_is_satisfied(result.x, bounds):
            return np.asarray(result.x, dtype=float)
        return None

    def deterministic_starts(bounds):
        """Return reproducible feasible starts ordered by economic relevance."""
        anchor = feasible_anchor(bounds)
        if anchor is None:
            return []

        starts = []

        def add_if_distinct(candidate):
            if (candidate is not None
                    and not any(np.allclose(candidate, prior,
                                              atol=1e-8, rtol=0.0)
                                for prior in starts)):
                starts.append(candidate)

        # Main runs supply current holdings when available, otherwise the
        # market prior. Equal weights give a neutral, deterministic contrast.
        for reference in (initial_weights, current_weights,
                          np.full(n, target / n, dtype=float)):
            if reference is not None:
                add_if_distinct(project_reference(reference, anchor, bounds))

        min_variance = minimize(
            lambda weights: portfolio_downside_semivariance(weights, Sigma),
            anchor,
            method="SLSQP",
            bounds=bounds,
            constraints=long_only_constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if min_variance.success:
            candidate = np.asarray(min_variance.x, dtype=float)
            if constraint_is_satisfied(candidate, bounds):
                add_if_distinct(candidate)
        add_if_distinct(anchor)
        return starts

    def solve_with_bounds(bounds):
        starts = deterministic_starts(bounds)
        if fast_mode:
            starts = starts[:1]
        else:
            starts = starts[:max(1, N_OPTIM_RESTARTS)]

        best_result = None
        best_value = -np.inf if mode == "A" else np.inf
        for w0 in starts:
            result = minimize(solver_objective, w0, method="SLSQP",
                              bounds=bounds,
                              constraints=long_only_constraints,
                              options={"maxiter": 1000, "ftol": 1e-12})
            if result.success and constraint_is_satisfied(result.x, bounds):
                value = objective_value(result.x)
                improved = value > best_value if mode == "A" else value < best_value
                if improved:
                    best_result = result
                    best_value = value
        return best_result, best_value

    bounds = [(0.0, max_single)] * n
    if tradable_mask is not None:
        if current_weights is None:
            raise ValueError("Sparse execution requires valid current_weights")
        tradable_mask = np.asarray(tradable_mask, dtype=bool)
        reference = np.asarray(current_weights, dtype=float)
        if tradable_mask.shape != (n,) or reference.shape != (n,):
            raise ValueError("tradable_mask/current_weights dimensions do not match the asset count")
        bounds = [((0.0, max_single) if tradable_mask[i]
                   else (float(reference[i]), float(reference[i])))
                  for i in range(n)]
    best_result, _ = solve_with_bounds(bounds)

    if best_result is None:
        raise RuntimeError("Optimizer did not converge; no default equal-weight portfolio was used. Adjust constraints or views and retry.")

    if float(MIN_WEIGHT_IF_HELD) > 1e-12:
        if tradable_mask is not None:
            raise RuntimeError(
                "0-or-minimum position sizing is not supported with fixed "
                "sparse-execution bounds")
        continuous_weights = np.asarray(best_result.x, dtype=float)
        selected_result = None
        selected_value = -np.inf if mode == "A" else np.inf
        for support in _minimum_weight_support_masks(
                continuous_weights, target, max_single):
            sized_bounds = [
                ((float(MIN_WEIGHT_IF_HELD), max_single) if included else (0.0, 0.0))
                for included in support
            ]
            candidate, value = solve_with_bounds(sized_bounds)
            if candidate is None:
                continue
            improved = value > selected_value if mode == "A" else value < selected_value
            if improved:
                selected_result, selected_value = candidate, value
        if selected_result is None:
            raise RuntimeError(
                "No support selected from the continuous solution can satisfy "
                "the 0-or-minimum position-size rule and existing constraints")
        best_result = selected_result

    weights = np.maximum(np.asarray(best_result.x, dtype=float), 0.0)
    weights[np.abs(weights) < 1e-10] = 0.0
    if not _minimum_weight_rule_satisfied(weights):
        raise RuntimeError("Optimizer returned a sub-minimum nonzero position")
    return weights

def portfolio_sharpe(weights, mu, Sigma):
    """Annual expected excess return divided by raw-risk annual volatility."""
    weights = np.asarray(weights, dtype=float)
    annual_variance = float(weights @ Sigma @ weights)
    if annual_variance <= 1e-16:
        return -1e12
    return float(weights @ mu) / np.sqrt(annual_variance)


def _execution_constraints_satisfied(weights, B, factor_names, B_stderr):
    """Check whether the current portfolio violates non-exposure hard constraints requiring immediate repair."""
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < -1e-8) or np.any(weights > MAX_WEIGHT_SINGLE + 1e-8):
        return False
    if not USE_FACTOR_EXPOSURE_CAPS or B is None or factor_names is None:
        return True

    for idx, factor_name in enumerate(factor_names):
        cap = get_factor_exposure_cap(factor_name)
        if cap is None:
            continue
        beta = weights @ B[:, idx]
        uncertainty_buffer = FACTOR_CAP_HAC_SE_Z * (weights @ B_stderr[:, idx])
        if beta + uncertainty_buffer > cap + 1e-6:
            return False
        if -beta + uncertainty_buffer > cap + 1e-6:
            return False
    return True


def build_min_cost_execution(full_target, current_weights, mu, Sigma,
                             B, factor_names, B_stderr, target_exposure,
                             rebalance_threshold, objective_mode=None):
    """Minimize turnover while retaining the target quality of objective A or B.

    A retains the existing BL Maximum-Sharpe floor exactly. B instead requires
    annual downside semivariance to remain within the configured proportional
    ceiling of the full B target; using a Sharpe floor for B would silently
    turn the conservative objective back into A during execution.
    """
    objective_mode = _normalized_portfolio_objective_mode(objective_mode)
    full_target = np.asarray(full_target, dtype=float)
    current_weights = np.asarray(current_weights, dtype=float)
    n_assets = len(full_target)
    current_is_feasible = (
        _execution_constraints_satisfied(
            current_weights, B, factor_names, B_stderr)
        and _minimum_weight_rule_satisfied(current_weights))
    exposure_deviation = abs(current_weights.sum() - target_exposure)

    if objective_mode == "A":
        full_objective = portfolio_sharpe(full_target, mu, Sigma)
        objective_scale = max(abs(full_objective), 0.01)
        objective_floor = full_objective - (1.0 - BL_SHARPE_RETENTION) * objective_scale
        current_objective = portfolio_sharpe(current_weights, mu, Sigma)
        execution_constraint = lambda weights: (
            portfolio_sharpe(weights, mu, Sigma) - objective_floor)
        is_retained = lambda weights: (
            portfolio_sharpe(weights, mu, Sigma) >= objective_floor - 1e-7)
        retained = lambda weights: max(0.0, min(
            1.0, 1.0 - max(0.0, full_objective - portfolio_sharpe(weights, mu, Sigma))
            / objective_scale))
        target_retention = float(BL_SHARPE_RETENTION)
        objective_label = "A_Max_Sharpe"
    else:
        full_objective = portfolio_downside_semivariance(full_target, Sigma)
        if full_objective <= 1e-16:
            raise RuntimeError("B target has non-positive downside semivariance")
        target_retention = float(B_DOWNSIDE_VARIANCE_RETENTION)
        if not 0.0 < target_retention <= 1.0:
            raise ValueError("B_DOWNSIDE_VARIANCE_RETENTION must lie in (0, 1]")
        objective_ceiling = full_objective / target_retention
        current_objective = portfolio_downside_semivariance(current_weights, Sigma)
        execution_constraint = lambda weights: (
            objective_ceiling - portfolio_downside_semivariance(weights, Sigma))
        is_retained = lambda weights: (
            portfolio_downside_semivariance(weights, Sigma) <= objective_ceiling + 1e-7)
        retained = lambda weights: max(0.0, min(
            1.0, full_objective / max(portfolio_downside_semivariance(weights, Sigma), 1e-16)))
        objective_label = "B_Minimum_Downside_Semivariance"

    if (current_is_feasible
            and exposure_deviation <= rebalance_threshold
            and is_retained(current_weights)):
        return current_weights.copy(), {
            "mode": "no_rebalance",
            "trade_count": 0,
            "objective_retention": retained(current_weights),
            "target_retention": target_retention,
            "objective_label": objective_label,
        }

    # x = [w, t], where t_i >= |w_i - current_i|. The linear t objective
    # minimizes total traded weight without a candidate-asset or trade-count cap.
    x0 = np.concatenate([full_target, np.abs(full_target - current_weights)])
    trade_bounds = [
        (0.0, max(WEIGHT_TARGET, float(current_weights[i]),
                  float(MAX_WEIGHT_SINGLE)))
        for i in range(n_assets)
    ]
    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x[:n_assets]) - target_exposure,
    }, {
        "type": "ineq",
        "fun": lambda x: x[n_assets:] - (x[:n_assets] - current_weights),
    }, {
        "type": "ineq",
        "fun": lambda x: x[n_assets:] + (x[:n_assets] - current_weights),
    }, {
        "type": "ineq",
        "fun": lambda x: execution_constraint(x[:n_assets]),
    }]
    if USE_FACTOR_EXPOSURE_CAPS and B is not None and factor_names is not None:
        for idx, factor_name in enumerate(factor_names):
            cap = get_factor_exposure_cap(factor_name)
            if cap is None:
                continue
            beta = np.asarray(B[:, idx], dtype=float)
            buffer = FACTOR_CAP_HAC_SE_Z * np.maximum(
                np.asarray(B_stderr[:, idx], dtype=float), 0.0)
            effective_cap = float(cap)
            constraints.extend([{
                "type": "ineq",
                "fun": lambda x, b=beta, u=buffer, c=effective_cap:
                    c - x[:n_assets] @ b - x[:n_assets] @ u,
            }, {
                "type": "ineq",
                "fun": lambda x, b=beta, u=buffer, c=effective_cap:
                    c + x[:n_assets] @ b - x[:n_assets] @ u,
            }])

    def solve_execution_with_weight_bounds(weight_bounds,
                                           require_minimum_weight=True):
        result = minimize(
            lambda x: float(np.sum(x[n_assets:])), x0, method="SLSQP",
            bounds=(list(weight_bounds) + trade_bounds), constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12})
        if not result.success:
            return None
        candidate = np.maximum(np.asarray(result.x[:n_assets], dtype=float), 0.0)
        valid = (
            abs(candidate.sum() - target_exposure) <= 1e-6
            and _execution_constraints_satisfied(
                candidate, B, factor_names, B_stderr)
            and (not require_minimum_weight
                 or _minimum_weight_rule_satisfied(candidate))
            and is_retained(candidate))
        return candidate if valid else None

    base_weight_bounds = [(0.0, MAX_WEIGHT_SINGLE)] * n_assets
    continuous_execution = solve_execution_with_weight_bounds(
        base_weight_bounds, require_minimum_weight=False)
    executed = continuous_execution
    if float(MIN_WEIGHT_IF_HELD) > 1e-12 and continuous_execution is not None:
        selected_execution = None
        selected_turnover = np.inf
        for support in _minimum_weight_support_masks(
                continuous_execution, target_exposure, MAX_WEIGHT_SINGLE):
            sized_bounds = [
                ((float(MIN_WEIGHT_IF_HELD), float(MAX_WEIGHT_SINGLE))
                 if included else (0.0, 0.0))
                for included in support
            ]
            candidate = solve_execution_with_weight_bounds(sized_bounds)
            if candidate is None:
                continue
            turnover = float(np.abs(candidate - current_weights).sum())
            if turnover < selected_turnover - 1e-12:
                selected_execution, selected_turnover = candidate, turnover
        executed = selected_execution

    if executed is None:
        executed = full_target.copy()
        mode = "objective_fallback"
    else:
        mode = "min_cost"
    return executed, {
        "mode": mode,
        "trade_count": int(np.sum(np.abs(executed - current_weights) > 1e-6)),
        "objective_retention": retained(executed),
        "target_retention": target_retention,
        "objective_label": objective_label,
    }


# ============================================================
# STEP 6: 报告生成
# ============================================================

def _w(text, width):
    """Right-align text using terminal display width (CJK characters occupy two cells)."""
    dw = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
             for c in text)
    return " " * max(0, width - dw) + text


def _risk_contribution(weights, Sigma):
    """Return portfolio risk contributions; signed contributions sum to 100%."""
    port_var = weights @ Sigma @ weights
    if port_var <= 1e-12:
        return np.zeros_like(weights), np.zeros_like(weights)
    marginal = Sigma @ weights
    signed = weights * marginal / port_var
    abs_total = np.sum(np.abs(signed))
    abs_share = np.abs(signed) / abs_total if abs_total > 1e-12 else signed
    return signed, abs_share


def generate_report(tickers, regression_stats, B, factor_names,
                    mu_bl, w_optimal, Sigma_annual,
                    rf_annual, resid_diag=None,
                    current_weights=None, rebalance_threshold=None,
                    regime_info=None, covariance_info=None,
                    full_target_weights=None, execution_info=None):
    """Generate the full report; expected returns are annualized total model returns."""

    print("\n" + "=" * 70)
    print("  FACTOR MODEL + BLACK-LITTERMAN ANALYSIS REPORT")
    print("=" * 70)
    # --- 1. 因子回归摘要 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  1. Factor Regression                                    │")
    print("└─────────────────────────────────────────────────────────┘")

    # 因子说明 (每个因子代表什么)
    factor_labels = {
        # FF6 (Fama-French 五因子 + 动量)
        "Mkt-RF":   "Market excess return",
        "SMB":      "Size (small minus big)",
        "HML":      "Value (high minus low book-to-market)",
        "RMW":      "Profitability (robust minus weak)",
        "CMA":      "Investment (conservative minus aggressive)",
        "Mom":      "Momentum (winners minus losers)",
        "QMJ":      "Quality minus junk (AQR daily US long-short factor)",
        "BAB":      "Betting against beta (AQR daily US long-short factor)",
        "Hardw49":  "Hardware industry (Ken French 49-industry daily return)",
        "Chips49":  "Semiconductor industry (Ken French 49-industry daily return)",
        # 宏观
        "Credit":   "Credit spread (HYG-LQD; negative = widening spread / recession)",
        # 行业
        "BankUS":   "US banks (KBWB sector excess return)",
        "EuroBank": "European financials (EUFN sector excess return, includes HSBC/BCS)",
        # 主题
        "AIPrem":   "AI premium (SMH cap-weighted minus SOXX equal-weighted)",
        # 可选行业
        "Staples":  "Consumer staples (XLP sector excess return)",
        "Disc":     "Consumer discretionary (XLY sector excess return)",
        "Health":   "Health care (XLV sector excess return)",
        "REIT":     "Real estate (XLRE sector excess return)",
        "Industry": "Industrials (XLI sector excess return)",
        "Comm":     "Communication services (XLC sector excess return)",
        "Clean":    "Clean energy (ICLN sector excess return)",
    }
    print("\n  Factor definitions:")
    for f in factor_names:
        print(f"    {f:>8s}  =  {factor_labels.get(f, f)}")

    # 紧凑列宽, 避免因子较多时终端自动换行
    COL = 9
    factor_header = "".join(f"{f[:7]:>{COL}s}" for f in factor_names)
    print(f"\n  {'Ticker':>8s}  {'Alpha':>9s}{factor_header}{'R²':>5s}")
    total_width = 8 + 2 + 9 + COL * len(factor_names) + 5
    print("  " + "-" * total_width)

    for ticker in tickers:
        stats = regression_stats[ticker]
        betas = stats["factor_betas"]
        pvals = stats["factor_pvalues"]

        alpha_ann = stats['alpha_annualized']
        a_stars = ("**" if stats["alpha_pvalue"] < 0.05 else
                   "*"   if stats["alpha_pvalue"] < 0.10 else "")
        alpha_cell = f"{alpha_ann:>+7.0%}{a_stars:<2s}"

        beta_cells = []
        for f in factor_names:
            b = betas.get(f, 0)
            p = pvals.get(f, 1)
            stars = ("**" if p < 0.05 else "*" if p < 0.10 else "")
            beta_cell = f"{b:>+6.2f}{stars:<3s}"
            beta_cells.append(beta_cell)

        r2_str = f"{stats['r_squared']:.3f}"
        print(f"  {ticker:>8s}  {alpha_cell}"
              f"{''.join(beta_cells)}{r2_str:>5s}")

    print("\n  ** HAC-adjusted p<0.05  * HAC-adjusted p<0.10")
    period_label = return_period_label(FREQUENCY)
    print(f"  Observations: {regression_stats[tickers[0]]['n_obs']} {period_label}")

    # 自定义因子层级与自动重叠诊断
    orth_info = resid_diag.get("orthogonalization") if resid_diag else None
    if orth_info:
        print("\n  Custom-factor hierarchy / overlap check (in model):")
        print("  EW trigger: lower-layer |rho| >= "
              f"{AUTO_LOWER_LAYER_ORTHOGONALIZE_CORR:.2f} or "
              f"VIF >= {FACTOR_VIF_ALERT:.1f}")
        print(f"  {_w('Factor', 8)}  {_w('Layer', 7)}  "
              f"{_w('Max |rho|', 10)}  {_w('EW-VIF', 7)}  "
              f"{_w('Action', 10)}  {_w('R² (against FF factors)', 23)}")
        print("  " + "-" * 79)
        overlap_groups = {"High": [], "Medium": [], "Low": []}
        for item in orth_info:
            if item.get("status") == "skipped":
                print(f"  {item.get('factor', ''):>8s}  "
                      f"{item.get('layer', '-'):>7s}  {'-':>10s}  "
                      f"{'-':>7s}  {'skipped':>10s}  {'-':>23s}")
                continue
            max_lower_corr = item.get("max_lower_corr", np.nan)
            corr_text = (f"{max_lower_corr:.2f}"
                         if np.isfinite(max_lower_corr) else "-")
            vif = item.get("vif", np.nan)
            vif_text = f"{vif:.2f}" if np.isfinite(vif) else "-"
            r_squared = item.get("r_squared", np.nan)
            r2_text = f"{r_squared:.2f}" if np.isfinite(r_squared) else "-"
            action = item.get("action", "-")
            high_overlap = (
                action == "FF+lower"
                or (np.isfinite(max_lower_corr)
                    and max_lower_corr >= AUTO_LOWER_LAYER_ORTHOGONALIZE_CORR)
                or (np.isfinite(vif) and vif >= FACTOR_VIF_ALERT)
            )
            medium_overlap = (
                (np.isfinite(max_lower_corr)
                 and max_lower_corr >= FACTOR_OVERLAP_MEDIUM_CORR)
                or (np.isfinite(vif) and vif >= FACTOR_OVERLAP_MEDIUM_VIF)
            )
            overlap_level = (
                "High" if high_overlap else "Medium" if medium_overlap else "Low"
            )
            overlap_groups[overlap_level].append(item.get("factor", ""))
            print(f"  {item.get('factor', ''):>8s}  "
                  f"{item.get('layer', '-'):>7s}  {corr_text:>10s}  "
                  f"{vif_text:>7s}  {action:>10s}  {r2_text:>23s}")

        print("  Lower-layer overlap assessment (High=auto residualize; "
              "Medium=watch):")
        for level in ("High", "Medium", "Low"):
            names = overlap_groups[level]
            print(f"    {level}: {', '.join(names) if names else 'none'}")

        factor_collinearity = resid_diag.get("factor_collinearity")
        if factor_collinearity:
            condition_number = factor_collinearity.get("condition_number", np.nan)
            if np.isfinite(condition_number):
                if condition_number < FACTOR_CONDITION_NUMBER_LOW_THRESHOLD:
                    condition_assessment = "Low: stable factor structure; beta estimates are relatively reliable"
                elif condition_number < FACTOR_CONDITION_NUMBER_ALERT:
                    condition_assessment = "Medium: some factor overlap; continue monitoring beta estimates"
                else:
                    condition_assessment = "High: substantial factor collinearity; beta estimates may be unstable"
                print("  Final standardized EW factor-design condition number: "
                      f"{condition_number:.1f} ({condition_assessment})")
                if factor_collinearity.get("alert"):
                    print("  ⚠ Factor design may be collinear; review the "
                          "highlighted layer relationships")

    # 残差相关性诊断
    if resid_diag and resid_diag.get("top_pairs"):
        reported_pairs = [
            pair for pair in resid_diag.get("all_pairs", [])
            if pair[3] >= RESIDUAL_CORRELATION_THRESHOLD
        ]
        print(f"\n  ρ(residual) ≥ +{RESIDUAL_CORRELATION_THRESHOLD:.2f}:")
        if reported_pairs:
            print(f"  {'Rank':>4s}  {'Asset 1':>6s}  {'Asset 2':>6s}  "
                  f"{'ρ(residual)':>12s}")
            print("  " + "-" * 40)
            for i, (_, t1, t2, c) in enumerate(reported_pairs):
                print(f"  {i+1:>4d}  {t1:>6s}  {t2:>6s}  {c:>+12.3f}")
        else:
            print("  -")

        print(f"\n  ρ(residual): mean={resid_diag['mean']:.3f}, "
              f"maximum={resid_diag['max']:.3f}")
        if resid_diag["max"] > 0.3:
            print("  ⚠ High residual correlation exists; some assets retain common risk "
                  "not fully explained by the factors")

        target_weights = dict(zip(tickers, w_optimal))
        concentrated_pairs = []
        for _, t1, t2, c in resid_diag.get("all_pairs", []):
            combined_weight = (target_weights.get(t1, 0.0)
                               + target_weights.get(t2, 0.0))
            if (combined_weight > RESIDUAL_PAIR_WEIGHT_ALERT
                    and abs(c) >= RESIDUAL_CORRELATION_THRESHOLD):
                concentrated_pairs.append((t1, t2, c, combined_weight))

        if concentrated_pairs:
            print("\n  ⚠ Residual-risk concentration "
                  f"(|ρ|≥{RESIDUAL_CORRELATION_THRESHOLD:.2f}, "
                  f"combined target weight>{RESIDUAL_PAIR_WEIGHT_ALERT:.0%}):")
            print(f"  {'Asset 1':>6s}  {'Asset 2':>6s}  {'ρ(residual)':>12s}  "
                  f"{'Combined target':>15s}")
            print("  " + "-" * 48)
            for t1, t2, c, combined_weight in concentrated_pairs:
                print(f"  {t1:>6s}  {t2:>6s}  {c:>+12.3f}  "
                      f"{combined_weight:>12.1%}")

    risk_peer_groups = ((covariance_info or {}).get("risk_peer_groups")
                        or [])
    if risk_peer_groups:
        print("\n  Automatic high-common-risk groups:")
        group_width = max(
            len("Assets"),
            *(len(", ".join(group["members"])) for group in risk_peer_groups),
        )
        pair_width = max(
            len("Pair"),
            *(len(f"{pair['left']}, {pair['right']}")
              for group in risk_peer_groups for pair in group["pairs"]),
        )
        print(f"  {'Assets':<{group_width}s}  {'Pair':<{pair_width}s}  "
              f"{'rho(factor)':>11s}  {'rho(resid.)':>11s}  "
              f"{'Common days':>11s}  {'rho(total)':>10s}")
        print("  " + "-" * (group_width + pair_width + 54))
        for group in risk_peer_groups:
            group_text = ", ".join(group["members"])
            for pair_index, pair in enumerate(group["pairs"]):
                factor_corr = pair["factor_corr"]
                residual_corr = pair["residual_corr"]
                factor_text = (f"{factor_corr:+.2f}"
                               if np.isfinite(factor_corr) else "-")
                residual_text = (f"{residual_corr:+.2f}"
                                 if np.isfinite(residual_corr) else "-")
                pair_text = f"{pair['left']}, {pair['right']}"
                asset_group = group_text if pair_index == 0 else ""
                print(f"  {asset_group:<{group_width}s}  "
                      f"{pair_text:<{pair_width}s}  {factor_text:>11s}  "
                      f"{residual_text:>11s}  {pair['common_days']:>11d}  "
                      f"{pair['total_corr']:>+10.2f}")

    # --- 2. 预期收益与风险 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  2. Expected Return & Risk                              │")
    print("└─────────────────────────────────────────────────────────┘")

    asset_volatilities = np.sqrt(np.maximum(np.diag(Sigma_annual), 0.0))
    print(f"  Risk-free rate (annual): {rf_annual:.2%}")
    sep2 = "-" * (8 + 2 + 16 + 2 + 18)
    print(f"{'Ticker':>8s}  {_w('Expected return', 16)}  {_w('Expected Std.Dev.', 18)}")
    print(sep2)
    for i, ticker in enumerate(tickers):
        print(f"{ticker:>8s}  {mu_bl[i] + rf_annual:>+12.2%}  "
              f"{asset_volatilities[i]:>16.2%}")

    if regime_info is not None:
        print("\n  Market Regime / Cash Buffer:")
        print(f"    Forecast state (next 5 trading days): {regime_info['state']}")
        if regime_info.get("deterministic_state") is not None:
            print(f"    Latest 5-trading-day market-signal state: "
                  f"{regime_info['deterministic_state']}")
        print(f"    Cash buffer: {regime_info['cash_buffer']:.1%}")
        print(f"    Target risk exposure: {regime_info['target_exposure']:.1%}")
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
    print("    Portfolio objective: Maximum Sharpe "
          "(BL annual excess return / raw-risk annual Std.Dev.)")
    # --- 3. 再平衡信号 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  3. Rebalancing Signal                                  │")
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

    mode = execution_info.get("mode")
    if mode == "bl_optimal_no_holdings":
        print("  CURRENT_HOLDINGS=None: no-cost execution optimization; displaying full objective target")
    elif mode == "min_cost":
        print("  Global min-cost model target "
              f"({execution_info.get('objective_label', 'A_Max_Sharpe')}): "
              f"involves {execution_info['trade_count']} assets and retains "
              f"{execution_info.get('objective_retention', float('nan')):.1%} of objective quality")
        print()
    elif mode == "objective_fallback":
        print("  Full-universe minimum-cost solver did not converge; using full objective target")
    elif mode == "no_rebalance":
        print("  Current portfolio satisfies the new regime constraints, exposure threshold, "
              "and configured A/B execution-quality floor; no rebalancing is required")

    if not has_current_holdings:
        print(f"  {_w('Stock sleeve', 12)}  {_w('BL Optimal', 12)}")
        print("  " + "-" * 24)
        for i, ticker in enumerate(tickers):
            print(f"  {ticker:>12s}  {w_optimal[i]:>12.1%}")
    else:
        execution_delta = w_optimal - report_current_weights
        position_tolerance = 1e-6
        need_rebalance = np.abs(execution_delta) > REBALANCE_THRESHOLD
        actionable_delta = np.where(need_rebalance, execution_delta, 0.0)
        deferred_count = int(np.sum(
            (np.abs(execution_delta) > position_tolerance) & ~need_rebalance))
        print(f"  {_w('Ticker', 8)}  {_w('Current', 8)}  "
              f"{_w('BL target', 10)}  {_w('Execution', 10)}  "
              f"{_w('Change', 8)}  {'Action':>6s}")
        print("  " + "-" * 66)
        print(f"  Per-asset trade-signal threshold: {REBALANCE_THRESHOLD:.1%} "
              "(applies uniformly to buys, sells, and existing-position adjustments)")
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
                  f"{actionable_delta[i]:>+8.1%}  "
                  f"{_w(action, 6)}")

        if any_action:
            gross_trade_notional = np.sum(np.abs(actionable_delta))
            cash_change = abs(actionable_delta.sum())
            turnover = (gross_trade_notional + cash_change) / 2
            tc_cost = gross_trade_notional * TRANSACTION_COST_BPS / 10000
            print(f"\n  → Rebalancing recommended: turnover={turnover:.1%}, "
                  f"estimated transaction cost={tc_cost:.2%}")
            if deferred_count:
                print(f"    {deferred_count} model adjustments below {REBALANCE_THRESHOLD:.0%} "
                      "were not included in the trade instructions")
        elif deferred_count:
            print(f"\n  → All {deferred_count} model adjustments are below "
                  f"{REBALANCE_THRESHOLD:.0%}; no action is required")

    # --- 4. 组合风险特征与关键发现 ---
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  4. Portfolio Risk & Findings                           │")
    print("└─────────────────────────────────────────────────────────┘")

    port_excess = w_optimal @ mu_bl
    port_return = port_excess + rf_annual
    port_var = w_optimal @ Sigma_annual @ w_optimal
    port_vol = np.sqrt(port_var)
    sharpe = port_excess / port_vol if port_vol > 1e-12 else np.nan

    # 因子暴露
    port_factor_betas = w_optimal @ B
    B_stderr = build_factor_beta_stderr_matrix(
        tickers, factor_names, regression_stats)

    print(f"  {_w('Factor', 8)}  {_w('Exposure', 10)}  "
          f"{_w('z*sum|w|StDevb', 16)}  {_w('Effective cap', 13)}  {_w('Status', 6)}")
    print(f"  {'-' * (8 + 2 + 10 + 2 + 16 + 2 + 8 + 2 + 6)}")
    for j, f in enumerate(factor_names):
        base_cap = get_factor_exposure_cap(f)
        if base_cap is None:
            cap_text = "-"
            buffer_text = "-"
            status = ""
        else:
            effective_cap = float(base_cap)
            hac_buffer = (
                FACTOR_CAP_HAC_SE_Z
                * (np.abs(w_optimal) @ B_stderr[:, j])
            )
            cap_text = f"±{effective_cap:.2f}"
            buffer_text = f"+{hac_buffer:.3f}"
            status = ("OK" if (abs(port_factor_betas[j]) + hac_buffer
                                <= effective_cap + 1e-6)
                      else "Breach")
        print(f"  {f:>8s}  {port_factor_betas[j]:>+10.3f}  "
              f"{buffer_text:>16s}  {cap_text:>8s}  {_w(status, 6)}")

    asset_rc, asset_abs_rc = _risk_contribution(w_optimal, Sigma_annual)
    top_asset_idx = np.argsort(asset_abs_rc)[::-1][:RISK_CONTRIBUTION_TOP_N]
    print(f"\n  Top {len(top_asset_idx)} asset risk contributions:")
    if covariance_info and covariance_info.get("equilibrium_prior_estimator") == "raw_ledoit_wolf_covariance":
        if covariance_info.get("equilibrium_prior_recent_weight", 0.0) > 0:
            print("  Prior covariance: Ledoit-Wolf-shrunk raw returns, "
                  f"{covariance_info['equilibrium_prior_recent_weight']:.0%} recent "
                  f"({covariance_info['equilibrium_prior_recent_observations']}d; shrinkage "
                  f"{covariance_info.get('equilibrium_prior_recent_shrinkage', 0.0):.1%}) / "
                  f"{1.0 - covariance_info['equilibrium_prior_recent_weight']:.0%} long "
                  f"({covariance_info['equilibrium_prior_long_observations']}d; shrinkage "
                  f"{covariance_info.get('equilibrium_prior_long_shrinkage', 0.0):.1%})")
        else:
            print("  Prior covariance: Ledoit-Wolf-shrunk raw returns")

    if covariance_info and covariance_info.get("recent_weight", 0.0) > 0:
        print("  Risk covariance: Ledoit-Wolf-shrunk downside semivariance, "
              f"{covariance_info['recent_weight']:.0%} recent "
              f"({covariance_info['recent_observations']}d; shrinkage "
              f"{covariance_info.get('recent_shrinkage', 0.0):.1%}) / "
              f"{1.0 - covariance_info['recent_weight']:.0%} long "
              f"({covariance_info['long_observations']}d; shrinkage "
              f"{covariance_info.get('long_shrinkage', 0.0):.1%})")
    else:
        print("  Risk covariance: Ledoit-Wolf-shrunk downside semivariance")
    print(f"  {_w('Ticker', 8)}  {_w('Weight', 8)}  "
          f"{_w('Contribution', 12)}  {_w('Absolute share', 14)}")
    print("  " + "-" * 42)
    for idx in top_asset_idx:
        print(f"  {tickers[idx]:>8s}  {w_optimal[idx]:>8.1%}  "
              f"{asset_rc[idx]:>+8.1%}  {asset_abs_rc[idx]:>10.1%}")

    max_factor_idx = np.argmax(np.abs(port_factor_betas))
    max_factor = factor_names[max_factor_idx]
    max_beta = port_factor_betas[max_factor_idx]

    factor_labels = {
        "Mkt-RF": "Market risk", "SMB": "Size (small vs big)",
        "HML": "Value vs growth", "RMW": "Profitability quality",
        "CMA": "Investment style", "Mom": "Momentum"
    }

    print("\n  Key findings:")
    print(f"  • Execution-target Sharpe Ratio: {sharpe:.3f}")

    print(f"  • Largest factor exposure: {max_factor} "
          f"({factor_labels.get(max_factor, '')}) = {max_beta:+.3f}")

    best_alpha_ticker = max(
        tickers,
        key=lambda t: abs(regression_stats[t]["alpha_annualized"]))
    best_alpha = regression_stats[best_alpha_ticker]["alpha_annualized"]
    best_alpha_p = regression_stats[best_alpha_ticker]["alpha_pvalue"]
    sig_str = "significant" if best_alpha_p < 0.05 else "not significant"
    print(f"  • Largest alpha: {best_alpha_ticker} = {best_alpha:+.1%} "
          f"(p={best_alpha_p:.3f}, {sig_str})")

    print("\n" + "=" * 70)

    return {
        "port_return": port_return,
        "port_vol": port_vol,
        "sharpe": sharpe,
        "weights": dict(zip(tickers, w_optimal)),
        "factor_betas": dict(zip(factor_names, port_factor_betas)),
    }


def run_sensitivity_analysis(tickers, factor_names, B, asset_views,
                             factor_views, Sigma_annual, pi,
                             w_baseline, regression_stats=None,
                             current_weights=None, regime_info=None,
                             B_stderr=None):
    """Perturb each view confidence and use screening plus full rechecks to assess weight stability."""
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  Sensitivity: Confidence ±{:.0%} Weight Impact             │".format(
        SENSITIVITY_DELTA))
    print("└─────────────────────────────────────────────────────────┘")

    # 收集所有观点标签
    view_labels = []
    for v in asset_views:
        if v[0] == "absolute":
            view_labels.append(f"{v[1]} absolute")
        else:
            view_labels.append(f"{v[1][0]} vs {v[1][1]}")
    for v in factor_views:
        _validate_factor_view(v)
        if v[0] == "absolute":
            view_labels.append(f"{v[1]} factor absolute")
        else:
            view_labels.append(f"{v[1][0]} vs {v[1][1]} factor")

    n_asset_views = len(asset_views)
    n_total = n_asset_views + len(factor_views)

    any_sensitive = False

    for idx in range(n_total):
        for sign, direction in [(1, "+"), (-1, "-")]:
            # 复制观点并扰动第 idx 个
            perturbed_av = [list(v) for v in asset_views]
            perturbed_fv = [list(v) for v in factor_views]

            if idx < n_asset_views:
                old_conf = perturbed_av[idx][3]
                perturbed_av[idx][3] = np.clip(
                    old_conf * (1 + sign * SENSITIVITY_DELTA), 0.01, 0.99)
                perturbed_av = [tuple(v) for v in perturbed_av]
                perturbed_fv = [tuple(v) for v in perturbed_fv]
            else:
                fi = idx - n_asset_views
                old_conf = perturbed_fv[fi][3]
                perturbed_fv[fi][3] = np.clip(
                    old_conf * (1 + sign * SENSITIVITY_DELTA), 0.01, 0.99)
                perturbed_av = [tuple(v) for v in perturbed_av]
                perturbed_fv = [tuple(v) for v in perturbed_fv]

            # 重新计算 (静默模式)
            P_p, Q_p, Omega_p = build_views(
                tickers, factor_names, B,
                perturbed_av, perturbed_fv,
                Sigma_annual, regression_stats=regression_stats,
                verbose=False)
            mu_p = black_litterman(pi, Sigma_annual, P_p, Q_p, Omega_p)
            optimization_kwargs = dict(
                current_weights=current_weights,
                B=B, B_stderr=B_stderr, factor_names=factor_names,
                target_exposure=(
                    regime_info.get("target_exposure")
                    if regime_info is not None else None),
            )

            # 局部筛查使用基准权重作起点；显著权重漂移时再完整复核。
            try:
                w_p = optimize_portfolio(
                    mu_p, Sigma_annual,
                    fast_mode=True, initial_weights=w_baseline,
                    **optimization_kwargs
                )
                max_change = np.max(np.abs(w_p - w_baseline))
                full_recheck = max_change >= SENSITIVITY_RECHECK_THRESHOLD
            except RuntimeError:
                # 局部起点未收敛时，直接使用完整优化复核该情景。
                full_recheck = True
            if full_recheck:
                w_p = optimize_portfolio(
                    mu_p, Sigma_annual,
                    **optimization_kwargs
                )
                max_change = np.max(np.abs(w_p - w_baseline))

            flag = ""
            if max_change > 0.10:
                flag = " ⚠ Sensitive"
                any_sensitive = True

            label = view_labels[idx]
            print(f"  {_w(label, 21)}  confidence {direction}"
                  f"{SENSITIVITY_DELTA:.0%}: "
                  f"maximum weight change {max_change:.1%}"
                  f"{' [full recheck]' if full_recheck else ''}{flag}")

    if not any_sensitive:
        print("\n  ✓ Weight changes remain below 10% after all view perturbations; the model is relatively robust")
    else:
        print("\n  ⚠ Some views have large weight effects; calibrate their confidence carefully")


# ============================================================
# STEP 8: 历史指数基准
# ============================================================

def run_backtest(raw_daily_returns, rf_annual, model_results):
    """
    Display the current BL target estimate alongside historical SPY/QQQ benchmarks.
    Do not present current BL weights as a historical realized backtest; BL's
    unavailable historical metrics are explicitly shown as "-".
    """
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  BL Target vs Historical Index Benchmarks                   │")
    print("└─────────────────────────────────────────────────────────────┘")
    ann_factor = periods_per_year("daily")
    obs_factor = observations_per_year("daily")

    available_benchmarks = [c for c in BENCHMARK_ETFS
                            if c in raw_daily_returns.columns]
    if not available_benchmarks:
        raise ValueError("Historical index benchmarks are missing: "
                         + ", ".join(BENCHMARK_ETFS))

    aligned_returns = raw_daily_returns[available_benchmarks].dropna(
        how="any").copy()
    max_obs_total = int(DATA_PERIOD_YEARS * periods_per_year("daily"))
    aligned_returns = aligned_returns.tail(max_obs_total)
    n_obs = len(aligned_returns)
    n_years = n_obs / obs_factor

    def calc_stats(rets, name):
        cum = np.cumprod(1 + rets)
        cum_total = cum[-1] - 1
        years = len(rets) / obs_factor
        if years > 0 and cum_total > -1:
            ann_ret = (1 + cum_total) ** (1 / years) - 1
        else:
            ann_ret = np.nan

        ann_vol = np.std(rets, ddof=1) * np.sqrt(ann_factor)
        excess = ann_ret - rf_annual
        sharpe = excess / ann_vol if ann_vol > 0 and np.isfinite(ann_ret) else 0

        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = np.min(drawdown)

        return {
            "name": name, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "cum_total": cum_total, "max_dd": max_dd,
        }

    benchmark_stats = {
        benchmark: calc_stats(aligned_returns[benchmark].values, benchmark)
        for benchmark in available_benchmarks
    }
    rows = [{
        "name": "BL Target", "basis": "Model estimate",
        "ann_ret": model_results["port_return"],
        "ann_vol": model_results["port_vol"],
        "sharpe": model_results["sharpe"],
        "max_dd": np.nan, "cum_total": np.nan,
    }]
    for benchmark in BENCHMARK_ETFS:
        historical = benchmark_stats.get(benchmark)
        if historical is None:
            rows.append({
                "name": benchmark, "basis": "Historical realized",
                "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "max_dd": np.nan, "cum_total": np.nan,
            })
        else:
            rows.append({"basis": "Historical realized", **historical})

    strategy_width = max(12, max(len(row["name"]) for row in rows))
    basis_width = max(18, max(len(row["basis"]) for row in rows))
    cum_ret_label = f"Cum.Ret({n_years:.1f}Y)"
    cum_col_width = max(14, len(cum_ret_label))
    table_width = (strategy_width + 2 + basis_width + 2 + 10 + 2 + 10 + 2
                   + 8 + 2 + 10 + 2
                   + cum_col_width)
    print(f"  {'Strategy':>{strategy_width}s}  {'Basis':>{basis_width}s}  "
          f"{'Ann.Ret':>10s}  {'Ann.StdDev':>10s}  "
          f"{'Sharpe':>8s}  {'Max DD':>10s}  {cum_ret_label:>{cum_col_width}s}")
    print("  " + "-" * table_width)
    for row in rows:
        ann_ret = "-" if not np.isfinite(row["ann_ret"]) else f"{row['ann_ret']:+.2%}"
        ann_vol = "-" if not np.isfinite(row["ann_vol"]) else f"{row['ann_vol']:.2%}"
        sharpe = "-" if not np.isfinite(row["sharpe"]) else f"{row['sharpe']:.3f}"
        max_dd = "-" if not np.isfinite(row["max_dd"]) else f"{row['max_dd']:.2%}"
        cum_total = ("-" if not np.isfinite(row["cum_total"])
                     else f"{row['cum_total']:+.2%}")
        print(f"  {row['name']:>{strategy_width}s}  "
              f"{row['basis']:>{basis_width}s}  {ann_ret:>10s}  "
              f"{ann_vol:>10s}  {sharpe:>8s}  {max_dd:>10s}  "
              f"{cum_total:>{cum_col_width}s}")

    print(f"\n  Historical period: {n_obs} raw daily return observations "
          f"({n_years:.1f} years)")
    print(f"  Risk-free rate: {rf_annual:.1%} annualized")

    return rows


def align_current_holdings(tickers, holdings, verbose=True):
    """
    Convert CURRENT_HOLDINGS to a ticker-aligned weight array.
    Only non-negative weights are supported; each input is preserved as entered
    and is not normalized by the total.
    """
    if holdings is None:
        if verbose:
            print("\n  (CURRENT_HOLDINGS is unset; displaying frictionless BL Optimal directly)")
        return None

    unknown = sorted(set(holdings) - set(tickers))
    if unknown and verbose:
        print(f"\n  ⚠ CURRENT_HOLDINGS contains assets outside TICKERS; ignored: {unknown}")

    curr_w = np.array([holdings.get(t, 0.0) for t in tickers], dtype=float)
    if np.any(curr_w < 0):
        negative = [
            f"{ticker}={curr_w[i]:.1%}"
            for i, ticker in enumerate(tickers) if curr_w[i] < 0
        ]
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
        names = [f"{t}={curr_w[i]:.1%}"
                 for i, t in enumerate(tickers) if over_limit[i]]
        print(f"  ⚠ Current holdings contain individual asset weights above "
              f"MAX_WEIGHT_SINGLE={MAX_WEIGHT_SINGLE:.0%}: "
              f"{', '.join(names)}")

    return curr_w


# ============================================================
# MAIN
# ============================================================

def main(objective_mode=None):
    global _RUNTIME_AUTO_ALPHA_CALIBRATION
    # Resolve a manual A/B selection once, before any data are fetched.  The same
    # label is passed to both target optimization and min-cost execution.
    objective_mode = _normalized_portfolio_objective_mode(objective_mode)
    print("=" * 70)
    print("  Factor-BL Pipeline")
    print("=" * 70)

    # --- Step 0: 实时市值权重 ---
    mkt_cap_weights = fetch_market_caps(TICKERS)

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
        risk_factor_start = history_info["model_start_date"]
        risk_factor_end = raw_daily_returns.index.max()
        if FREQUENCY in ("daily", "weekly_avg_daily"):
            daily_factor_inputs = fetch_daily_factor_inputs(
                risk_factor_start, risk_factor_end)
            factors = fetch_ff_factors(
                start_date, end_date, FREQUENCY,
                verbose=False, daily_inputs=daily_factor_inputs)
            risk_factors = fetch_ff_factors(
                risk_factor_start, risk_factor_end, "daily",
                verbose=False, daily_inputs=daily_factor_inputs)
        else:
            factors = fetch_ff_factors(start_date, end_date, FREQUENCY)
            risk_factors = fetch_ff_factors(
                risk_factor_start, risk_factor_end, "daily", verbose=False)
        regime_benchmark_returns = fetch_regime_benchmark_returns(
            raw_daily_all_returns)

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
            print(f"  ⚠ Missing factors: {missing}; using: {available_factors}")
    risk_factor_names = [f for f in FACTORS_TO_USE
                         if f in risk_factors.columns]
    if not risk_factor_names:
        raise ValueError("Raw daily risk factors are unavailable")

    raw_regime_benchmark_returns = raw_daily_all_returns[REGIME_BENCHMARKS].dropna(
        how="any")
    regime_info = compute_regime_context(
        stock_returns, regime_benchmark_returns, raw_regime_benchmark_returns)

    # --- Step 2: 因子回归 ---
    print("\n[Step2] Running factor regression...")
    B, regression_stats, resid_diag, available_factors, factors = \
        run_factor_regression(stock_returns, factors, available_factors)
    B_stderr = build_factor_beta_stderr_matrix(
        model_tickers, available_factors, regression_stats)

    if AUTO_PEER_ALPHA_VIEWS_ENABLED:
        print("\n[Step 2b] Rebuilding in-memory two-year 17A calibration...")
        _RUNTIME_AUTO_ALPHA_CALIBRATION = \
            _build_runtime_automatic_alpha_calibration(model_tickers, end_date)
        print(
            "  Mature 5-day dates: "
            f"{_RUNTIME_AUTO_ALPHA_CALIBRATION['calibration_dates']}; "
            "slope="
            f"{_RUNTIME_AUTO_ALPHA_CALIBRATION['slope_5d']:.6f}; "
            "storage=memory only")
    else:
        _RUNTIME_AUTO_ALPHA_CALIBRATION = None

    # --- Step 3: 协方差矩阵 ---
    Sigma_annual, covariance_info = build_covariance_matrices(
        raw_daily_returns, model_tickers, risk_factors, risk_factor_names,
        recent_weight=regime_info["recent_covariance_weight"])
    # Return prior and portfolio risk intentionally use separate estimators:
    # raw-return Ledoit-Wolf covariance generates the equilibrium prior, while
    # Sigma_annual remains Ledoit-Wolf-shrunk downside semivariance for risk.
    Sigma_prior_annual, prior_covariance_info = build_equilibrium_prior_covariance(
        raw_daily_returns, model_tickers,
        recent_weight=regime_info["recent_covariance_weight"],
        return_info=True)
    covariance_info.update(prior_covariance_info)

    # --- Step 4: Black-Litterman ---
    print(f"\n[Step 4] Running Black-Litterman model...")

    w_mkt = np.array([mkt_cap_weights[t] for t in model_tickers])
    w_mkt = w_mkt / w_mkt.sum()

    delta_market = estimate_market_risk_price(
        end_date, raw_spy_returns=raw_daily_all_returns["SPY"])
    pi = compute_equilibrium_returns(delta_market, Sigma_prior_annual, w_mkt)

    if AUTO_PEER_ALPHA_VIEWS_ENABLED or ASSET_VIEWS or FACTOR_VIEWS:
        # Automatic 17A and optional manual views update only the raw-LW return
        # prior. Downside covariance remains reserved for risk/execution.
        P, Q, Omega, view_diagnostics = build_production_views(
            model_tickers, available_factors, B, pi, Sigma_prior_annual,
            regression_stats, end_date, ASSET_VIEWS, FACTOR_VIEWS,
            verbose=True)
        if P is None:
            mu_final = pi.copy()
        else:
            mu_final = black_litterman(
                pi, Sigma_prior_annual, P, Q, Omega)
    else:
        mu_final = pi.copy()
        view_diagnostics = {
            "automatic_peer_alpha": {"enabled": False, "view_count": 0},
            "manual_view_count": 0, "total_view_count": 0,
        }

    # --- Step 5: 优化 ---
    objective_label = ("A / Maximum-Sharpe" if objective_mode == "A"
                       else "B / Minimum Downside Semivariance")
    print(f"\n[Step 5] Running {objective_label} optimization...")

    curr_w = align_current_holdings(
        model_tickers, CURRENT_HOLDINGS, verbose=False)

    # 完整 BL-regime Optimal 不含执行摩擦，是所有执行方案的共同基准。
    w_full_target = optimize_portfolio(
        mu_final, Sigma_annual,
        initial_weights=(curr_w if curr_w is not None else w_mkt),
        B=B, B_stderr=B_stderr,
        factor_names=available_factors,
        target_exposure=regime_info["target_exposure"],
        objective_mode=objective_mode,
    )
    if curr_w is None:
        w_optimal = w_full_target.copy()
        execution_info = {
            "mode": "bl_optimal_no_holdings",
            "trade_count": 0,
            "objective_retention": 1.0,
            "target_retention": (BL_SHARPE_RETENTION if objective_mode == "A"
                                 else B_DOWNSIDE_VARIANCE_RETENTION),
            "objective_label": ("A_Max_Sharpe" if objective_mode == "A"
                                else "B_Minimum_Downside_Semivariance"),
        }
    else:
        w_optimal, execution_info = build_min_cost_execution(
            w_full_target, curr_w, mu_final, Sigma_annual,
            B, available_factors, B_stderr,
            regime_info["target_exposure"], REBALANCE_THRESHOLD,
            objective_mode=objective_mode)

    # --- Step 6: 报告 ---
    results = generate_report(
        model_tickers, regression_stats, B, available_factors,
        mu_final, w_optimal, Sigma_annual,
        RISK_FREE_RATE, resid_diag,
        current_weights=curr_w,
        regime_info=regime_info,
        covariance_info=covariance_info,
        full_target_weights=w_full_target,
        execution_info=execution_info,
    )
    results["view_diagnostics"] = view_diagnostics

    if SENSITIVITY_ENABLED and (ASSET_VIEWS or FACTOR_VIEWS):
        run_sensitivity_analysis(
            model_tickers, available_factors, B, ASSET_VIEWS, FACTOR_VIEWS,
            Sigma_annual, pi, w_full_target,
            regression_stats=regression_stats,
            current_weights=None,
            regime_info=regime_info,
            B_stderr=B_stderr
        )

    # --- Step 8: 回测 ---
    if BACKTEST_ENABLED:
        run_backtest(raw_daily_returns, RISK_FREE_RATE, results)

    return results


if __name__ == "__main__":
    # The default remains A; --objective B or --choose-objective is an explicit,
    # per-run manual operator override and does not alter the saved default.
    results = main(objective_mode=parse_manual_objective_override())
