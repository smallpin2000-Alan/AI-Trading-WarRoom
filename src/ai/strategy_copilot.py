# src/ai/strategy_copilot.py
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.schema.strategy import StrategySchema, ConditionNode, LogicalGroup, LeafCondition


@dataclass
class CopilotParseResult:
    """AI 自然語言轉 Strategy Schema 結果"""
    strategy: StrategySchema
    summary_text: str
    logic_tree_lines: List[str]
    clarifications: List[str] = field(default_factory=list)


class AIStrategyCopilot:
    """
    AI 自然語言策略協作與視覺化邏輯樹模組 (AI Strategy Copilot)
    - 將中文口語交易想法轉換為嚴格驗證的 StrategySchema (JSON)。
    - 主動進行條件澄清 (Clarification) 與強制停損安全審查。
    - 產生結構化視覺邏輯樹與白話文摘要，確保人類與引擎認知 100% 一致。
    """

    @classmethod
    def parse_natural_language(
        cls,
        prompt: str,
        symbol: str = "TMF",
        timeframe: str = "5m",
        session: str = "day_and_night",
    ) -> CopilotParseResult:
        text = (prompt or "").strip()
        if not text:
            raise ValueError("請輸入策略的中文自然語言描述。")

        clarifications: List[str] = []
        indicators: Dict[str, Dict[str, Any]] = {}
        long_conds: List[Dict[str, Any]] = []
        short_conds: List[Dict[str, Any]] = []

        # 1. 解析移動平均線 (SMA / EMA) 與突破通道 (HIGHEST / LOWEST)
        ema_matches = re.findall(r"(\d+)\s*(?:EMA|ema|指數均線)", text)
        sma_matches = re.findall(r"(\d+)\s*(?:MA|ma|SMA|sma|均線|日線|分線)", text)

        if "通道" in text or "最高" in text or "最低" in text or "突破" in text and not (ema_matches or sma_matches):
            ch_match = re.search(r"(\d+)\s*(?:根|期|天|日)?(?:最高|最低|通道|高點|低點)", text)
            ch_len = int(ch_match.group(1)) if ch_match else 20
            indicators["hh"] = {"type": "HIGHEST", "source": "high", "length": ch_len}
            indicators["ll"] = {"type": "LOWEST", "source": "low", "length": ch_len}
            long_conds.append({"cross_above": ["close", "hh"]})
            short_conds.append({"cross_below": ["close", "ll"]})
            strat_id = f"channel_breakout_{ch_len}"
        elif "布林" in text or "BB" in text.upper():
            bb_match = re.search(r"(\d+)\s*(?:期|根)?(?:布林|BB)", text, re.IGNORECASE)
            bb_len = int(bb_match.group(1)) if bb_match else 20
            indicators["bb_up"] = {"type": "BB_UPPER", "source": "close", "length": bb_len}
            indicators["bb_low"] = {"type": "BB_LOWER", "source": "close", "length": bb_len}
            long_conds.append({"cross_above": ["close", "bb_up"]})
            short_conds.append({"cross_below": ["close", "bb_low"]})
            strat_id = f"bb_breakout_{bb_len}"
        else:
            use_ema = len(ema_matches) >= 2 or ("EMA" in text.upper() and not sma_matches)
            raw_nums = [int(x) for x in (ema_matches if use_ema else sma_matches)]
            if len(raw_nums) < 2:
                all_nums = [int(x) for x in re.findall(r"\b(\d{1,3})\b", text) if 2 <= int(x) <= 240]
                if len(all_nums) >= 2:
                    raw_nums = sorted(list(set(all_nums[:2])))
                else:
                    raw_nums = [5, 20]
                    clarifications.append("未明確指定兩條均線期數，已自動採用標準雙均線 (5 期與 20 期)。")
            else:
                raw_nums = sorted(list(set(raw_nums)))[:2]

            fast_n, slow_n = raw_nums[0], raw_nums[1]
            ind_type = "EMA" if use_ema else "SMA"
            indicators["fast_ma"] = {"type": ind_type, "source": "close", "length": fast_n}
            indicators["slow_ma"] = {"type": ind_type, "source": "close", "length": slow_n}
            long_conds.append({"cross_above": ["fast_ma", "slow_ma"]})
            short_conds.append({"cross_below": ["fast_ma", "slow_ma"]})
            strat_id = f"{ind_type.lower()}_{fast_n}_{slow_n}_cross"

        # 2. 解析附加指標濾網 (RSI / KD / ADX)
        if "RSI" in text.upper():
            rsi_len_m = re.search(r"RSI\s*\(?(\d+)\)?", text, re.IGNORECASE)
            rsi_len = int(rsi_len_m.group(1)) if rsi_len_m else 14
            rsi_th_m = re.search(r"RSI[^0-9]*?(?:大於|>|高於)\s*(\d+)", text, re.IGNORECASE)
            rsi_th = float(rsi_th_m.group(1)) if rsi_th_m else 50.0
            indicators["rsi"] = {"type": "RSI", "source": "close", "length": rsi_len}
            long_conds.append({"gt": ["rsi", rsi_th]})
            short_conds.append({"lt": ["rsi", 100.0 - rsi_th]})

        if "KD" in text.upper():
            indicators["kd_k"] = {"type": "KD_K", "source": "close", "length": 9}
            indicators["kd_d"] = {"type": "KD_D", "source": "close", "length": 9}
            long_conds.append({"gt": ["kd_k", "kd_d"]})
            short_conds.append({"lt": ["kd_k", "kd_d"]})

        if "ADX" in text.upper():
            adx_th_m = re.search(r"ADX[^0-9]*?(?:大於|>|高於)\s*(\d+)", text, re.IGNORECASE)
            adx_th = float(adx_th_m.group(1)) if adx_th_m else 25.0
            indicators["adx"] = {"type": "ADX", "source": "close", "length": 14}
            long_conds.append({"gt": ["adx", adx_th]})
            short_conds.append({"gt": ["adx", adx_th]})

        # 3. 解析停損、停利與移動停損 (強制安全檢查)
        sl_match = re.search(r"停損\s*(\d+(?:\.\d+)?)\s*(%|點)?", text)
        tp_match = re.search(r"停利\s*(\d+(?:\.\d+)?)\s*(%|點)?", text)
        ts_match = re.search(r"移動停損\s*(\d+(?:\.\d+)?)\s*(%|點)?", text)

        stop_loss_dict: Optional[Dict[str, Any]] = None
        trailing_stop_dict: Optional[Dict[str, Any]] = None
        take_profit_dict: Optional[Dict[str, Any]] = None

        if sl_match:
            val = float(sl_match.group(1))
            unit = sl_match.group(2) or "點"
            sl_type = "percent" if unit == "%" else "fixed_points"
            stop_loss_dict = {"type": sl_type, "value": val}

        if ts_match:
            val = float(ts_match.group(1))
            unit = ts_match.group(2) or "點"
            ts_type = "percent" if unit == "%" else "points"
            trailing_stop_dict = {"type": ts_type, "value": val, "activation_points": val}

        if tp_match:
            val = float(tp_match.group(1))
            unit = tp_match.group(2) or "點"
            tp_type = "percent" if unit == "%" else "fixed_points"
            take_profit_dict = {"type": tp_type, "value": val}

        if stop_loss_dict is None and trailing_stop_dict is None:
            stop_loss_dict = {"type": "fixed_points", "value": 50.0}
            clarifications.append(
                "⚠️ 交易安全提醒：您的口語描述中未包含停損條件！依據系統最高安全原則（無停損絕不執行），已自動強制補上「固定 50 點停損」，請確認是否合乎預期。"
            )

        # 4. 解析 AI 信心度門檻 (Confidence Threshold)
        conf_match = re.search(r"(?:信心度|門檻|閾值)[^0-9]*?(\d+(?:\.\d+)?)\s*%?", text)
        conf_filter_dict = {"enabled": False, "threshold": 60.0, "regime_adaptive": True}
        if conf_match:
            c_val = float(conf_match.group(1))
            conf_filter_dict = {
                "enabled": True,
                "threshold": max(1.0, min(99.0, c_val)),
                "regime_adaptive": ("動態" in text or "狀態" in text or "自動" in text or True),
            }

        # 5. 解析方向 (是否僅做多或僅做空)
        only_long = ("只做多" in text or "僅做多" in text) and "做空" not in text
        only_short = ("只做空" in text or "僅做空" in text) and "做多" not in text
        if not only_long and not only_short and "做空" not in text and "空單" not in text:
            clarifications.append("口語中未特別提及空單條件，已自動配置對稱反向做空條件。")

        def _pack_cond(c_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not c_list:
                return None
            if len(c_list) == 1:
                return c_list[0]
            return {"all": c_list}

        schema_dict = {
            "id": strat_id,
            "version": 1,
            "symbol": symbol if symbol in ("TMF", "MXF", "TXF") else "TMF",
            "timeframe": timeframe,
            "session": session,
            "indicators": indicators,
            "entry_long": None if only_short else _pack_cond(long_conds),
            "entry_short": None if only_long else _pack_cond(short_conds),
            "exit": {
                "stop_loss": stop_loss_dict,
                "take_profit": take_profit_dict,
                "trailing_stop": trailing_stop_dict,
            },
            "execution": {
                "signal_on": "bar_close",
                "fill_at": "next_bar_open",
                "order_type": "range_market",
                "slippage_points": 1.0,
            },
            "position_sizing": {"type": "fixed", "contracts": 1},
            "confidence_filter": conf_filter_dict,
        }

        strategy = StrategySchema.model_validate(schema_dict)
        summary = cls.generate_plain_summary(strategy)
        tree_lines = cls.generate_logic_tree_lines(strategy)
        return CopilotParseResult(
            strategy=strategy,
            summary_text=summary,
            logic_tree_lines=tree_lines,
            clarifications=clarifications,
        )

    @classmethod
    def _format_condition_node(cls, node: Optional[ConditionNode], indent: int = 2) -> List[str]:
        if node is None:
            return [" " * indent + "└── (無)"]
        prefix = " " * indent
        if isinstance(node, LogicalGroup):
            if node.all is not None:
                lines = [f"{prefix}├── [AND 全部滿足]"]
                for child in node.all:
                    lines.extend(cls._format_condition_node(child, indent + 4))
                return lines
            if node.any is not None:
                lines = [f"{prefix}├── [OR 任一滿足]"]
                for child in node.any:
                    lines.extend(cls._format_condition_node(child, indent + 4))
                return lines
        if isinstance(node, LeafCondition):
            if node.cross_above:
                return [f"{prefix}└── 📈 向上突破 (cross_above): {node.cross_above[0]} ↗ {node.cross_above[1]}"]
            if node.cross_below:
                return [f"{prefix}└── 📉 向下跌破 (cross_below): {node.cross_below[0]} ↘ {node.cross_below[1]}"]
            if node.gt:
                return [f"{prefix}└── ➕ 大於 (gt): {node.gt[0]} > {node.gt[1]}"]
            if node.gte:
                return [f"{prefix}└── ➕ 大於等於 (gte): {node.gte[0]} >= {node.gte[1]}"]
            if node.lt:
                return [f"{prefix}└── ➖ 小於 (lt): {node.lt[0]} < {node.lt[1]}"]
            if node.lte:
                return [f"{prefix}└── ➖ 小於等於 (lte): {node.lte[0]} <= {node.lte[1]}"]
        return [f"{prefix}└── {node}"]

    @classmethod
    def generate_logic_tree_lines(cls, strategy: StrategySchema) -> List[str]:
        """產生視覺化策略邏輯樹結構"""
        lines = [
            f"🌳 策略邏輯樹: {strategy.id} (v{strategy.version}) | 週期: {strategy.timeframe} | 時段: {strategy.session}",
            "├── 📊 指標定義 (Indicators)",
        ]
        for k, spec in strategy.indicators.items():
            lines.append(f"│   └── {k}: {spec.type}(source={spec.source}, length={spec.length})")

        lines.append("├── 🟢 多方進場條件 (Entry Long)")
        lines.extend(cls._format_condition_node(strategy.entry_long, indent=4))

        lines.append("├── 🔴 空方進場條件 (Entry Short)")
        lines.extend(cls._format_condition_node(strategy.entry_short, indent=4))

        lines.append("└── 🛡️ 強制風控與出場條件 (Exit & Stop Loss)")
        sl = strategy.exit.stop_loss
        tp = strategy.exit.take_profit
        ts = strategy.exit.trailing_stop
        if sl:
            lines.append(f"    ├── 停損 (Stop Loss): {sl.type} = {sl.value if sl.value is not None else '指標交叉'}")
        if tp:
            lines.append(f"    ├── 停利 (Take Profit): {tp.type} = {tp.value}")
        if ts:
            lines.append(f"    ├── 移動停損 (Trailing Stop): {ts.type} = {ts.value} (啟動門檻: {ts.activation_points})")
        cf = strategy.confidence_filter
        if cf and cf.enabled:
            lines.append(f"    ├── 🧠 AI 信心度門檻 (Confidence Filter): >= {cf.threshold}% (市場狀態動態調節: {'開啟' if cf.regime_adaptive else '關閉'})")
        lines.append(f"    └── 執行規則: 收盤確認 ({strategy.execution.signal_on}) → 次根開盤成交 ({strategy.execution.fill_at})")
        return lines

    @classmethod
    def generate_plain_summary(cls, strategy: StrategySchema) -> str:
        """產生白話文策略摘要"""
        ind_desc = "、".join([f"`{k}` ({v.type} {v.length}期)" for k, v in strategy.indicators.items()])
        sl = strategy.exit.stop_loss
        tp = strategy.exit.take_profit
        ts = strategy.exit.trailing_stop

        sl_text = f"{sl.type} ({sl.value})" if sl and sl.value else ("指標反轉停損" if sl else "無固定停損")
        tp_text = f"{tp.type} ({tp.value})" if tp and tp.value else "未設定"
        ts_text = f"{ts.type} ({ts.value})" if ts else "未設定"

        return (
            f"本策略 **`{strategy.id}`** 運行於 **`{strategy.timeframe}`** 週期（`{strategy.session}`），"
            f"使用指標：{ind_desc}。訊號於 K 棒收盤確認、次根開盤價成交（含滑價 {strategy.execution.slippage_points} 點）；"
            f"風控出場設定為：**停損 = `{sl_text}`**、**停利 = `{tp_text}`**、**移動停損 = `{ts_text}`**。"
        )
