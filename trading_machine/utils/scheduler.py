"""
utils/scheduler.py — APScheduler-based scheduler for market-hours trading.
Handles EST timezone, weekends, and US market holidays 2024-2027.
"""

from datetime import datetime, date, time
from typing import Callable, Set, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone as pytz_timezone

from utils.config import get_config
from utils.logger import get_logger

# US Market Holidays 2024-2027 (hardcoded exact dates)
US_MARKET_HOLIDAYS: Set[date] = {
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving Day
    date(2024, 12, 25), # Christmas Day
    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day (observed 7/3 if Saturday, but 7/4 is Friday)
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving Day
    date(2025, 12, 25), # Christmas Day
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed, 7/4 is Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving Day
    date(2026, 12, 25), # Christmas Day
    # 2027
    date(2027, 1, 1),   # New Year's Day
    date(2027, 1, 18),  # Martin Luther King Jr. Day
    date(2027, 2, 15),  # Presidents' Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 5, 31),  # Memorial Day
    date(2027, 6, 18),  # Juneteenth (observed, 6/19 is Saturday)
    date(2027, 7, 5),   # Independence Day (observed, 7/4 is Sunday)
    date(2027, 9, 6),   # Labor Day
    date(2027, 11, 25), # Thanksgiving Day
    date(2027, 12, 24), # Christmas Day (observed, 12/25 is Saturday)
}


def _is_trading_day(d: date) -> bool:
    """Return True if the given date is a trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in US_MARKET_HOLIDAYS:
        return False
    return True


class SchedulerManager:
    """Manages scheduled jobs around market hours in EST timezone.

    Handles weekends and US market holidays automatically.
    """

    def __init__(self):
        config = get_config()
        self._timezone = pytz_timezone(config.TIMEZONE)
        self._scheduler = BackgroundScheduler(timezone=self._timezone)
        self._logger = get_logger()
        self._market_open_str = config.MARKET_OPEN
        self._market_close_str = config.MARKET_CLOSE
        self._polling_seconds = config.POLLING_INTERVAL_SECONDS

        # Parse market hours
        open_h, open_m = map(int, self._market_open_str.split(":"))
        close_h, close_m = map(int, self._market_close_str.split(":"))
        self._market_open_time = time(open_h, open_m)
        self._market_close_time = time(close_h, close_m)

    def _is_market_open_now(self) -> bool:
        """Check if the market is currently open (EST timezone)."""
        now_est = datetime.now(self._timezone)
        today = now_est.date()
        if not _is_trading_day(today):
            return False
        current_time = now_est.time()
        return self._market_open_time <= current_time <= self._market_close_time

    def _market_hours_filter(self) -> bool:
        """Filter for intraday polling: only run when market is open."""
        return self._is_market_open_now()

    def start(self) -> None:
        """Start the background scheduler."""
        self._scheduler.start()
        self._logger.info("Scheduler started (timezone: EST)")

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._scheduler.shutdown(wait=False)
        self._logger.info("Scheduler stopped")

    def add_market_open_job(self, func: Callable) -> None:
        """Schedule func to run at MARKET_OPEN every trading day."""
        open_h, open_m = map(int, self._market_open_str.split(":"))
        self._scheduler.add_job(
            func,
            CronTrigger(
                hour=open_h,
                minute=open_m,
                day_of_week="mon-fri",
                timezone=self._timezone,
            ),
            id=f"market_open_{func.__name__}",
            replace_existing=True,
            name=f"Market open job: {func.__name__}",
        )
        self._logger.info(f"Market open job added: {func.__name__} at {self._market_open_str}")

    def add_market_close_job(self, func: Callable) -> None:
        """Schedule func to run at MARKET_CLOSE every trading day."""
        close_h, close_m = map(int, self._market_close_str.split(":"))
        self._scheduler.add_job(
            func,
            CronTrigger(
                hour=close_h,
                minute=close_m,
                day_of_week="mon-fri",
                timezone=self._timezone,
            ),
            id=f"market_close_{func.__name__}",
            replace_existing=True,
            name=f"Market close job: {func.__name__}",
        )
        self._logger.info(f"Market close job added: {func.__name__} at {self._market_close_str}")

    def add_overnight_job(self, func: Callable) -> None:
        """Schedule func to run at 18:00 every day (including weekends)."""
        self._scheduler.add_job(
            func,
            CronTrigger(hour=18, minute=0, timezone=self._timezone),
            id=f"overnight_{func.__name__}",
            replace_existing=True,
            name=f"Overnight job: {func.__name__}",
        )
        self._logger.info(f"Overnight job added: {func.__name__} at 18:00")

    def add_premarket_job(self, func: Callable) -> None:
        """Schedule func to run at 08:00 every trading day."""
        self._scheduler.add_job(
            func,
            CronTrigger(
                hour=8,
                minute=0,
                day_of_week="mon-fri",
                timezone=self._timezone,
            ),
            id=f"premarket_{func.__name__}",
            replace_existing=True,
            name=f"Premarket job: {func.__name__}",
        )
        self._logger.info(f"Premarket job added: {func.__name__} at 08:00")

    def add_weekend_job(self, func: Callable) -> None:
        """Schedule func to run at Saturday 02:00."""
        self._scheduler.add_job(
            func,
            CronTrigger(
                hour=2,
                minute=0,
                day_of_week="sat",
                timezone=self._timezone,
            ),
            id=f"weekend_{func.__name__}",
            replace_existing=True,
            name=f"Weekend job: {func.__name__}",
        )
        self._logger.info(f"Weekend job added: {func.__name__} at Saturday 02:00")

    def add_intraday_polling_job(self, func: Callable) -> None:
        """Schedule func to run every POLLING_INTERVAL_SECONDS during market hours only."""
        # Wrap func to check market hours before executing
        def _polling_wrapper():
            if self._is_market_open_now():
                func()
            # Silently skip if market is closed (scheduler might fire at edges)

        self._scheduler.add_job(
            _polling_wrapper,
            IntervalTrigger(seconds=self._polling_seconds, timezone=self._timezone),
            id=f"intraday_polling_{func.__name__}",
            replace_existing=True,
            name=f"Intraday polling: {func.__name__}",
        )
        self._logger.info(
            f"Intraday polling job added: {func.__name__} every {self._polling_seconds}s"
        )

    @property
    def scheduler(self) -> BackgroundScheduler:
        return self._scheduler


_scheduler_instance: SchedulerManager | None = None


def get_scheduler() -> SchedulerManager:
    """Return the singleton SchedulerManager instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SchedulerManager()
    return _scheduler_instance
