from __future__ import annotations
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

TimeframeType = Literal["1m", "5m", "15m", "30m", "60m", "1D"]
SessionType = Literal["day_only", "day_and_night"]
IndicatorType = Literal[
    "SMA",
    "EMA",
    "RSI",
    "KD_K",
    "KD_D",
    "MACD",
    "ATR",
    "ADX",
    "BB_UPPER",
    "BB_LOWER",
    "BB_WIDTH",
    "HIGHEST",
    "LOWEST",
    "ROC",
    "BIAS",
    "MA_SLOPE",
    "VOL_RATIO",
]
PriceSource = Literal["open", "high", "low", "close", "volume"]
BuiltinSeries = {"open", "high", "low", "close", "volume"}


class IndicatorSpec(BaseModel):
    type: IndicatorType
    source: PriceSource = "close"
    length: int = Field(gt=0, le=500, description="指標計算期數")
    timeframe: Optional[TimeframeType] = Field(default=None)


class BarsSinceExpr(BaseModel):
    # 條件節點禁止未知欄位：打錯字的運算子不可被靜默忽略
    model_config = ConfigDict(extra="forbid")

    bars_since: ConditionNode


OperandType = Union[str, float, int, BarsSinceExpr]


class LeafCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cross_above: Optional[tuple[str, Union[str, float]]] = None
    cross_below: Optional[tuple[str, Union[str, float]]] = None
    gt: Optional[tuple[OperandType, OperandType]] = None
    gte: Optional[tuple[OperandType, OperandType]] = None
    lt: Optional[tuple[OperandType, OperandType]] = None
    lte: Optional[tuple[OperandType, OperandType]] = None

    @model_validator(mode="after")
    def check_exactly_one_operator(self) -> LeafCondition:
        active_ops = [k for k, v in self.model_dump().items() if v is not None]
        if len(active_ops) != 1:
            raise ValueError(f"每個條件節點必須且只能指定一個運算子，目前指定了: {active_ops}")
        return self


class LogicalGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all: Optional[List[ConditionNode]] = Field(default=None, min_length=1)
    any: Optional[List[ConditionNode]] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_all_or_any(self) -> LogicalGroup:
        if (self.all is None and self.any is None) or (self.all is not None and self.any is not None):
            raise ValueError("邏輯群組必須在 'all' (AND) 與 'any' (OR) 之間擇一填寫")
        return self


ConditionNode = Union[LogicalGroup, LeafCondition]
BarsSinceExpr.model_rebuild()
LogicalGroup.model_rebuild()


class StopLossSpec(BaseModel):
    # 出場規格禁止未知欄位：風控參數打錯字時必須直接拒絕，不可靜默套用預設值
    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed_points", "percent", "atr_multiple", "indicator_cross"]
    value: Optional[float] = Field(default=None, gt=0)
    atr_length: int = Field(default=14, gt=0, le=500, description="atr_multiple 使用的 ATR 期數")
    long: Optional[LeafCondition] = None
    short: Optional[LeafCondition] = None

    @model_validator(mode="after")
    def validate_stop_loss_params(self) -> StopLossSpec:
        if self.type == "indicator_cross":
            if not self.long and not self.short:
                raise ValueError("當停損類型為 indicator_cross 時，必須提供 long 或 short 出場條件")
        else:
            if self.value is None:
                raise ValueError(f"當停損類型為 {self.type} 時，必須提供大於 0 的 value")
        return self


class TrailingStopSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["points", "percent", "atr_multiple"]
    value: float = Field(gt=0)
    activation_points: float = Field(default=0.0, ge=0)
    atr_length: int = Field(default=14, gt=0, le=500, description="atr_multiple 使用的 ATR 期數")


class TimeExitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before_settlement: bool = True
    minutes_before_close: int = Field(default=15, ge=1, le=60)


class ExitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_loss: Optional[StopLossSpec] = None
    take_profit: Optional[StopLossSpec] = None
    trailing_stop: Optional[TrailingStopSpec] = None
    time_exit: TimeExitSpec = Field(default_factory=TimeExitSpec)

    @model_validator(mode="after")
    def enforce_mandatory_stop_loss(self) -> ExitSpec:
        if self.stop_loss is None and self.trailing_stop is None:
            raise ValueError("安全攔截：策略必須至少設定 stop_loss 或 trailing_stop 其中一種停損機制！")
        # indicator_cross 只在 K 棒收盤才判斷，盤中沒有任何價位保護；必須搭配一道價位型移動停損
        if self.stop_loss is not None and self.stop_loss.type == "indicator_cross" and self.trailing_stop is None:
            raise ValueError(
                "安全攔截：indicator_cross 停損沒有價位保護，必須同時設定 trailing_stop 作為價位型停損！"
            )
        return self


class ExecutionSpec(BaseModel):
    signal_on: Literal["bar_close"] = "bar_close"
    fill_at: Literal["next_bar_open"] = "next_bar_open"
    order_type: Literal["range_market", "limit"] = "range_market"
    slippage_points: float = Field(default=1.0, ge=0)


class PositionSizingSpec(BaseModel):
    type: Literal["fixed", "risk_percent"] = "fixed"
    contracts: int = Field(default=1, ge=1, le=20)
    risk_pct: Optional[float] = Field(default=None, gt=0, le=0.05)


class ConfidenceFilterSpec(BaseModel):
    enabled: bool = Field(default=False, description="是否啟用 AI 信心度閾值過濾")
    threshold: float = Field(default=60.0, ge=0.0, le=100.0, description="進場最低信心度門檻 (0~100%)")
    regime_adaptive: bool = Field(default=True, description="是否依市場狀態 (趨勢/盤整) 動態調節門檻")


class StrategySchema(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    version: int = Field(ge=1)
    symbol: Literal["TMF", "MXF", "TXF"] = "TMF"
    timeframe: TimeframeType = "60m"
    session: SessionType = "day_and_night"
    indicators: Dict[str, IndicatorSpec]
    entry_long: Optional[ConditionNode] = None
    entry_short: Optional[ConditionNode] = None
    exit: ExitSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    position_sizing: PositionSizingSpec = Field(default_factory=PositionSizingSpec)
    confidence_filter: ConfidenceFilterSpec = Field(default_factory=ConfidenceFilterSpec)

    @model_validator(mode="after")
    def validate_strategy_integrity(self) -> StrategySchema:
        if self.entry_long is None and self.entry_short is None:
            raise ValueError("策略必須至少定義 entry_long 或 entry_short 其中之一")

        valid_names = set(self.indicators.keys()) | BuiltinSeries

        def _check_node(node: Optional[ConditionNode]):
            if node is None:
                return
            if isinstance(node, LogicalGroup):
                for child in (node.all or node.any or []):
                    _check_node(child)
            elif isinstance(node, LeafCondition):
                for op_val in [node.cross_above, node.cross_below, node.gt, node.gte, node.lt, node.lte]:
                    if op_val is not None:
                        for item in op_val:
                            if isinstance(item, str) and item not in valid_names:
                                raise ValueError(f"條件中引用了未定義的指標或欄位名稱: '{item}'")
                            elif isinstance(item, BarsSinceExpr):
                                _check_node(item.bars_since)

        _check_node(self.entry_long)
        _check_node(self.entry_short)
        if self.exit.stop_loss and self.exit.stop_loss.type == "indicator_cross":
            # 有哪個方向的進場，就必須有該方向的指標出場條件，避免單邊部位沒有停損
            if self.entry_long is not None and self.exit.stop_loss.long is None:
                raise ValueError("安全攔截：策略定義了 entry_long，indicator_cross 停損必須提供 long 出場條件")
            if self.entry_short is not None and self.exit.stop_loss.short is None:
                raise ValueError("安全攔截：策略定義了 entry_short，indicator_cross 停損必須提供 short 出場條件")
            _check_node(self.exit.stop_loss.long)
            _check_node(self.exit.stop_loss.short)

        return self
