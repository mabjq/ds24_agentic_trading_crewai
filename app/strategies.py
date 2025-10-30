from typing import Optional
import backtrader as bt
import datetime
import logging
import math
import pandas as pd
from config.config import AppConfig

logger = logging.getLogger(__name__)

class GaussianKijunStrategy(bt.Strategy):
    """
    Gaussian + Kijun + VAPI + ATR + SMMA200 strategy for Backtrader.
    Uses either fixed USD position sizing (config.trading.fixed_position_size)
    or risk-based sizing (config.trading.risk_pct, 1% per trade) if fixed size is 0.
    Includes custom breakeven (+2R), ATR trailing stop (ATR*4), TP1 partial exit (30% at 2R),
    and trendbreak exit logic. Relies on indicators from indicators.py, executed in
    backtest.py as part of the analysis step in the ETL pipeline.
    """

    params = (
        ("app_config", AppConfig()),
        ("external_trades", []),
    )

    def __init__(self) -> None:
        """Initiate and prepare variables for trade management and indicators."""
        cfg: AppConfig = self.p.app_config
        self.cfg = cfg.trading
        
        # Data feed with extra lines for indicators
        self.data_extras = self.datas[0]
        
        # External trades override from agent (for backtest integration)
        self.external_trades = self.p.external_trades
        self.executed_external = set()

        # Trade management
        self.today = None
        self.trades_today = 0
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.initial_atr: Optional[float] = None
        self.entry_risk: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.breakeven_active = False
        self.highest_since_entry: Optional[float] = None
        self.lowest_since_entry: Optional[float] = None
        self.close_reason = ""
        self.entry_order = None
        self.exit_order = None

    def log(self, txt: str, dt: Optional[datetime.datetime] = None) -> None:
        """Log strategy events with timestamp.

        Args:
            txt: Message to log.
            dt: Optional datetime for the log (default: current bar time).
        """
        dt = dt or self.data.datetime.datetime(0)
        logger.debug(f"{dt.isoformat()} - {txt}")

    def next(self) -> None:
        """Execute strategy logic on each bar.
        Evaluates entry/exit conditions using indicators (gauss, kijun, vapi, adx, smma)
        and manages trade limits (max_trades_per_day) from config.
        """
        dt = self.data.datetime.datetime(0)
        if self.today is None or dt.date() != self.today:
            self.today = dt.date()
            self.trades_today = 0

        if len(self.data) < self.cfg.min_bars:
            return

        # Get indicators
        try:
            close = float(self.data.close[0])
            prev_close = float(self.data.close[-1])
            high = float(self.data.high[0])
            low = float(self.data.low[0])
            gauss = float(self.data_extras.gauss[0])
            gauss_prev = float(self.data_extras.gauss[-1])
            kijun = float(self.data_extras.kijun[0])
            kijun_prev = float(self.data_extras.kijun[-1])
            vapi = float(self.data_extras.vapi[0])
            vapi_prev = float(self.data_extras.vapi[-1])
            adx = float(self.data_extras.adx[0])
            smma = float(self.data_extras.smma[0])
            atr = float(self.data_extras.atr[0])
            swing_low = float(self.data_extras.swing_low[0])
            swing_high = float(self.data_extras.swing_high[0])
        except (ValueError, TypeError):
            return

        if pd.isna(adx) or adx <= self.cfg.adx_threshold:
            return

        if self.trades_today >= self.cfg.max_trades_per_day:
            return

    # Override: Force external trades from agent (e.g., for low ADX scenarios)
        if self.external_trades:
            for trade in self.external_trades:
                if trade.get('entry') and trade['entry'] not in self.executed_external:
                    entry_price = float(trade['entry'].split('@')[1])
                    direction = trade.get('direction', 'long')  # Default long
                    size = trade.get('size', self._determine_size(entry_price, entry_price - atr))  # Risk-baserad
                    if direction == 'long' and abs(close - entry_price) < atr * 0.5:  # Nära current close
                        self._enter_long(close, size, swing_low, atr)
                        self.executed_external.add(trade['entry'])
                        self.log(f"External LONG override: {size}@{close:.2f} from agent trade {trade['entry']}")
                        return  # En per bar för enkelhet
                    elif direction == 'short' and abs(close - entry_price) < atr * 0.5:
                        self._enter_short(close, size, swing_high, atr)
                        self.executed_external.add(trade['entry'])
                        self.log(f"External SHORT override: {size}@{close:.2f} from agent trade {trade['entry']}")
                        return

        gauss_up = gauss > gauss_prev
        vapi_up = vapi > vapi_prev

        if not self.position:
            self.exit_order = None
            self.close_reason = ""

            # LONG entry
            if gauss_up and vapi_up and close > smma and close > gauss and swing_low < close:
                size = self._determine_size(close, swing_low)
                if size > 0:
                    self._enter_long(close, size, swing_low, atr)
                    return

            # SHORT entry
            if not gauss_up and not vapi_up and close < smma and close < gauss and swing_high > close:
                size = self._determine_size(close, swing_high, short=True)
                if size > 0:
                    self._enter_short(close, size, swing_high, atr)
                    return
        else:
            self._update_position_management(close, high, low, kijun)

            if self.position.size > 0 and close < kijun and prev_close < kijun_prev and not self.close_reason:
                self.close()
                self.close_reason = "Trendbreak LONG (close under Kijun)"
                self.log(self.close_reason)
            elif self.position.size < 0 and close > kijun and prev_close > kijun_prev and not self.close_reason:
                self.close()
                self.close_reason = "Trendbreak SHORT (close over Kijun)"
                self.log(self.close_reason)

    def _determine_size(self, entry: float, stop: float, short: bool = False) -> int:
        """Return contract size using fixed USD size or risk-based sizing.

        Args:
            entry: Entry price for the trade.
            stop: Stop-loss price for the trade.
            short: Flag for short position (default: False).

        Returns:
            int: Calculated position size (contracts).
        """
        if self.cfg.fixed_position_size > 0:
            usd = self.cfg.fixed_position_size
            return max(1, int(usd / entry))
        return self.calculate_size(entry, stop, short)

    def _enter_long(self, close: float, size: int, stop: float, atr: float) -> None:
        """Initiate a long position with specified size, stop-loss, and take-profit.
        Sets custom TP1 (2R) and tracks initial ATR for trailing.

        Args:
            close: Current close price (entry price).
            size: Number of contracts to buy.
            stop: Initial stop-loss price.
            atr: Current ATR value for trailing setup.
        """
        self.entry_order = self.buy(size=size)
        self.entry_price = close
        self.stop_price = stop
        self.initial_atr = atr
        self.entry_risk = close - stop
        self.tp_price = close + self.cfg.tp_r_multiple * self.entry_risk
        self.breakeven_active = False
        self.highest_since_entry = close
        self.trades_today += 1
        self.log(f"LONG ENTRY: {size}@{close:.2f} SL={stop:.2f} TP={self.tp_price:.2f}")

    def _enter_short(self, close: float, size: int, stop: float, atr: float) -> None:
        """Initiate a short position with specified size, stop-loss, and take-profit.
        Sets custom TP1 (2R) and tracks initial ATR for trailing.

        Args:
            close: Current close price (entry price).
            size: Number of contracts to sell.
            stop: Initial stop-loss price.
            atr: Current ATR value for trailing setup.
        """
        self.entry_order = self.sell(size=size)
        self.entry_price = close
        self.stop_price = stop
        self.initial_atr = atr
        self.entry_risk = stop - close
        self.tp_price = close - self.cfg.tp_r_multiple * self.entry_risk
        self.breakeven_active = False
        self.lowest_since_entry = close
        self.trades_today += 1
        self.log(f"SHORT ENTRY: {size}@{close:.2f} SL={stop:.2f} TP={self.tp_price:.2f}")

    def _update_position_management(self, close: float, high: float, low: float, kijun_v: float) -> None:
        """Updates stop for breakeven and trailing.
        Implements custom breakeven (+2R), ATR trailing stop (ATR*4), and TP1 (30% partial).

        Args:
            close: Current close price.
            high: Current high price.
            low: Current low price.
            kijun_v: Current Kijun-Sen value.
        """
        if self.entry_price is None or self.stop_price is None or self.entry_risk is None:
            return

        if self.position.size > 0:  # Long position
            # Update highest since entry
            self.highest_since_entry = max(self.highest_since_entry or self.entry_price, high)
            
            # Breakeven at +2R
            be_price = self.entry_price + self.cfg.tp_r_multiple * self.entry_risk
            if close >= be_price and not self.breakeven_active:
                self.stop_price = self.entry_price
                self.breakeven_active = True
                self.log(f"Breakeven activated for LONG at {self.stop_price:.2f}")

            # Trailing stop: highest_since_entry - ATR * 4
            if self.initial_atr is not None:
                trail_stop = self.highest_since_entry - self.initial_atr * self.cfg.trailing_atr_mult
                if trail_stop > self.stop_price:
                    self.stop_price = trail_stop
                    self.log(f"Trailing stop updated for LONG to {self.stop_price:.2f} (high={self.highest_since_entry:.2f})")

            # Check TP (2R) - use limit order if not already placed
            if self.tp_price is not None and high >= self.tp_price and self.exit_order is None:
                # Partial close: 30% at TP
                tp_size = int(math.floor(abs(self.position.size) * 0.3))
                if tp_size > 0:
                    self.exit_order = self.sell(size=tp_size, exectype=bt.Order.Limit, price=self.tp_price)
                    self.log(f"TP1 order placed for LONG: {tp_size} contracts at {self.tp_price:.2f}")

            # Stop loss check
            if close <= self.stop_price and self.close_reason == "":
                self.close()
                self.close_reason = f"Stop loss LONG at {self.stop_price:.2f}"
                self.log(self.close_reason)

        else:  # Short position
            # Update lowest since entry
            self.lowest_since_entry = min(self.lowest_since_entry or self.entry_price, low)
            
            # Breakeven at -2R
            be_price = self.entry_price - self.cfg.tp_r_multiple * self.entry_risk
            if close <= be_price and not self.breakeven_active:
                self.stop_price = self.entry_price
                self.breakeven_active = True
                self.log(f"Breakeven activated for SHORT at {self.stop_price:.2f}")

            # Trailing stop: lowest_since_entry + ATR * 4
            if self.initial_atr is not None:
                trail_stop = self.lowest_since_entry + self.initial_atr * self.cfg.trailing_atr_mult
                if trail_stop < self.stop_price:
                    self.stop_price = trail_stop
                    self.log(f"Trailing stop updated for SHORT to {self.stop_price:.2f} (low={self.lowest_since_entry:.2f})")

            # Check TP (2R for short)
            if self.tp_price is not None and low <= self.tp_price and self.exit_order is None:
                # Partial close: 30% at TP
                tp_size = int(math.floor(abs(self.position.size) * 0.3))
                if tp_size > 0:
                    self.exit_order = self.buy(size=tp_size, exectype=bt.Order.Limit, price=self.tp_price)
                    self.log(f"TP1 order placed for SHORT: {tp_size} contracts at {self.tp_price:.2f}")

            # Stop loss check
            if close >= self.stop_price and self.close_reason == "":
                self.close()
                self.close_reason = f"Stop loss SHORT at {self.stop_price:.2f}"
                self.log(self.close_reason)

    def notify_order(self, order) -> None:
        """Handle order completion notifications.

        Args:
            order: Backtrader Order object.
        """
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            if getattr(self, "exit_order", None) is order:
                self.exit_order = None
        elif order.status in [order.Canceled, order.Rejected]:
            if getattr(self, "exit_order", None) is order:
                self.exit_order = None

    def notify_trade(self, trade) -> None:
        """Handle trade closure notifications and log PnL.

        Args:
            trade: Backtrader Trade object.
        """
        if trade.isclosed:
            self.log(f"TRADE CLOSED: PnL Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}, Reason: {self.close_reason if self.close_reason else 'Unknown'}")
            self.close_reason = ""  # Reset for next trade

    def calculate_size(self, entry: float, stop: float, short: bool = False) -> int:
        """Calculate position size based on risk.

        Args:
            entry: Entry price for the trade.
            stop: Stop-loss price for the trade.
            short: Flag for short position (default: False).

        Returns:
            int: Calculated position size (contracts) based on risk_pct.
        """
        equity = self.broker.getvalue()
        risk_amount = equity * self.cfg.risk_pct
        distance = abs(entry - stop)
        if distance <= 0:
            return 0
        raw_size = risk_amount / (distance * self.cfg.contract_multiplier)
        return max(0, int(math.floor(raw_size)))
