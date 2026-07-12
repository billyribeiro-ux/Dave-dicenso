"""
models/rl_agent.py — PPO trading agent with custom Gymnasium environment.

The RL agent never sees prices — it only sees the 256-dim latent vector from
the trained world model. All trading decisions are made in latent space.
"""

import os
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.config import get_config
from utils.logger import get_logger


# ============================================================================
# TradingEnvironment — Custom Gymnasium environment
# ============================================================================

class TradingEnvironment(gym.Env):
    """Custom Gymnasium environment for trading.

    Action space: Discrete(3) — 0=Long, 1=Short, 2=Hold/Exit
    Observation space: Box(256,) — latent vector from world model
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_model=None,
        initial_capital: float = 100000.0,
        max_steps: int = 2000,
    ):
        super().__init__()

        self.world_model = world_model
        self.initial_capital = initial_capital
        self.max_steps = max_steps

        # Action space: 0=Long, 1=Short, 2=Hold/Exit
        self.action_space = spaces.Discrete(3)

        # Observation space: 256-dim latent vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(256,),
            dtype=np.float32,
        )

        # State variables (initialized in reset)
        self.position: int = 0
        self.entry_price: Optional[float] = None
        self.current_price: float = 0.0
        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.position_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.trade_history: List[dict] = []
        self.current_step: int = 0
        self.latent_state: np.ndarray = np.zeros(256, dtype=np.float32)
        self._leverage = 1.0  # Implied leverage for PnL calculation

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.position = 0
        self.entry_price = None
        self.current_price = 0.0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.position_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_history = []
        self.current_step = 0
        self.latent_state = np.zeros(256, dtype=np.float32)

        info = {
            "equity": self.equity,
            "position": self.position,
            "step": self.current_step,
        }
        return self.latent_state.copy(), info

    def set_latent_state(self, latent_vector: np.ndarray) -> None:
        """Set the current latent state (called externally by training loop)."""
        self.latent_state = latent_vector.astype(np.float32).flatten()
        if len(self.latent_state) != 256:
            raise ValueError(f"Latent vector must be 256-dim, got {len(self.latent_state)}")

    def set_current_price(self, price: float) -> None:
        """Update the current market price."""
        self.current_price = price
        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:
                self.unrealized_pnl = (price - self.entry_price) / self.entry_price * self.equity * self._leverage
            else:
                self.unrealized_pnl = (self.entry_price - price) / self.entry_price * self.equity * self._leverage

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step.

        Args:
            action: 0=Long, 1=Short, 2=Hold/Exit

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        reward = 0.0
        info: Dict[str, Any] = {
            "equity": self.equity,
            "position": self.position,
            "step": self.current_step,
            "pnl": 0.0,
            "trade_closed": False,
        }

        max_drawdown_pct = get_config().MAX_DRAWDOWN_PCT

        # --- Execute action ---
        if action == 0:  # LONG
            if self.position <= 0:
                # Close existing short if any
                if self.position == -1 and self.entry_price is not None:
                    pnl = self.calculate_pnl(self.current_price, self.position)
                    self.equity += pnl
                    self.position_pnl += pnl
                    self.trade_history.append({
                        "direction": "SHORT",
                        "entry_price": self.entry_price,
                        "exit_price": self.current_price,
                        "pnl": pnl,
                        "step": self.current_step,
                    })
                    reward = pnl / self.initial_capital * 100.0
                    info["pnl"] = pnl
                    info["trade_closed"] = True

                # Enter long
                self.position = 1
                self.entry_price = self.current_price
                self.unrealized_pnl = 0.0
                info["position"] = 1

        elif action == 1:  # SHORT
            if self.position >= 0:
                # Close existing long if any
                if self.position == 1 and self.entry_price is not None:
                    pnl = self.calculate_pnl(self.current_price, self.position)
                    self.equity += pnl
                    self.position_pnl += pnl
                    self.trade_history.append({
                        "direction": "LONG",
                        "entry_price": self.entry_price,
                        "exit_price": self.current_price,
                        "pnl": pnl,
                        "step": self.current_step,
                    })
                    reward = pnl / self.initial_capital * 100.0
                    info["pnl"] = pnl
                    info["trade_closed"] = True

                # Enter short
                self.position = -1
                self.entry_price = self.current_price
                self.unrealized_pnl = 0.0
                info["position"] = -1

        elif action == 2:  # HOLD/EXIT
            if self.position != 0 and self.entry_price is not None:
                # Exit position
                pnl = self.calculate_pnl(self.current_price, self.position)
                self.equity += pnl
                self.position_pnl += pnl
                self.trade_history.append({
                    "direction": "LONG" if self.position == 1 else "SHORT",
                    "entry_price": self.entry_price,
                    "exit_price": self.current_price,
                    "pnl": pnl,
                    "step": self.current_step,
                })
                reward = pnl / self.initial_capital * 100.0
                info["pnl"] = pnl
                info["trade_closed"] = True
                self.position = 0
                self.entry_price = None
                self.unrealized_pnl = 0.0
            # If already flat, do nothing — reward stays 0

        # --- Post-action reward adjustments ---
        if self.position != 0:
            # Small penalty for holding (prevents infinite holding)
            reward -= 0.001

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # --- Termination checks ---
        drawdown = (self.peak_equity - self.equity) / self.initial_capital
        terminated = False
        truncated = False

        # Blow-up check
        if self.equity <= self.initial_capital * (1.0 - max_drawdown_pct):
            terminated = True
            reward -= 10.0  # Large penalty for blow-up
            info["termination_reason"] = "max_drawdown"

        # Max steps
        if self.current_step >= self.max_steps:
            truncated = True
            # Auto-close any open position
            if self.position != 0 and self.entry_price is not None:
                pnl = self.calculate_pnl(self.current_price, self.position)
                self.equity += pnl
                self.trade_history.append({
                    "direction": "LONG" if self.position == 1 else "SHORT",
                    "entry_price": self.entry_price,
                    "exit_price": self.current_price,
                    "pnl": pnl,
                    "step": self.current_step,
                })
                self.position = 0
                self.entry_price = None
            info["termination_reason"] = "max_steps"

        info["equity"] = self.equity
        info["drawdown_pct"] = drawdown
        info["position"] = self.position

        return self.latent_state.copy(), reward, terminated, truncated, info

    def calculate_pnl(self, exit_price: float, direction: int) -> float:
        """Calculate P&L in dollar amount.

        For LONG:  pnl = (exit - entry) / entry * equity * leverage
        For SHORT: pnl = (entry - exit) / entry * equity * leverage
        """
        if self.entry_price is None or self.entry_price == 0:
            return 0.0

        if direction == 1:  # Long
            pnl = (exit_price - self.entry_price) / self.entry_price * self.equity * self._leverage
        else:  # Short
            pnl = (self.entry_price - exit_price) / self.entry_price * self.equity * self._leverage

        return pnl

    def render(self):
        """Not required for training."""
        return None


# ============================================================================
# RLAgent — PPO-based trading agent
# ============================================================================

class TrainingCallback(BaseCallback):
    """Callback to log training metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.logger = get_logger()
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            avg_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            avg_length = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            self.episode_rewards.append(float(avg_reward))
            self.episode_lengths.append(float(avg_length))


class RLAgent:
    """PPO trading agent that operates on latent states from the world model.

    Each RL agent is trained for a specific ticker. The agent learns
    to map latent vectors (256-dim) to trading actions (Long/Short/Hold).
    """

    def __init__(
        self,
        ticker: str,
        world_model=None,
        device: torch.device | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.ticker = ticker
        self.device = device
        self.world_model = world_model
        self.logger = get_logger()

        # Create the trading environment
        config = get_config()
        self.env = TradingEnvironment(
            world_model=world_model,
            initial_capital=config.INITIAL_CAPITAL,
            max_steps=2000,
        )

        # PPO policy kwargs
        policy_kwargs = {
            "net_arch": [
                dict(pi=[512, 256, 128], vf=[512, 256, 128])
            ],
        }

        # Create PPO agent
        log_dir = os.path.join(config.LOG_DIR, "tensorboard", ticker)
        os.makedirs(log_dir, exist_ok=True)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device=device,
        )

        self.training_history: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "total_pnl": [],
        }

    def train_on_historical(
        self,
        ticker_prices: np.ndarray,
        total_timesteps: int = 1_000_000,
        world_model=None,
    ) -> dict:
        """Custom training loop on historical price data.

        For each episode:
        1. Select a random segment of historical prices
        2. Encode price windows using world_model.encode() to get latent vectors
        3. Step through latent vectors sequentially
        4. Collect rollouts and call agent.learn()

        Args:
            ticker_prices: 1D numpy array of close prices.
            total_timesteps: Total training timesteps.
            world_model: Trained PriceVAE (if not already stored).

        Returns:
            Training history dictionary.
        """
        if world_model is not None:
            self.world_model = world_model

        if self.world_model is None:
            raise ValueError("World model must be provided or set before training")

        self.world_model.eval()
        self.world_model.to(self.device)

        window_size = 500
        if len(ticker_prices) < window_size + 200:
            raise ValueError(
                f"Price array too short ({len(ticker_prices)}), need at least {window_size + 200}"
            )

        self.logger.info(
            f"Training RL agent for {self.ticker} on {len(ticker_prices)} price points, "
            f"{total_timesteps} total timesteps"
        )

        # We'll use stable-baselines3's built-in training with a custom
        # environment that provides latent states

        # Wrap in DummyVecEnv for stable-baselines3
        vec_env = DummyVecEnv([lambda: self.env])

        # Recreate model with vec env
        config = get_config()
        log_dir = os.path.join(config.LOG_DIR, "tensorboard", self.ticker)

        policy_kwargs = {
            "net_arch": [
                dict(pi=[512, 256, 128], vf=[512, 256, 128])
            ],
        }

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device=self.device,
        )

        callback = TrainingCallback()

        # Train the PPO model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        self.training_history["episode_rewards"] = callback.episode_rewards
        self.training_history["episode_lengths"] = callback.episode_lengths

        self.logger.info(f"RL training complete for {self.ticker}")
        return self.training_history

    def predict_action(
        self, latent_vector: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Any]:
        """Predict trading action from latent state.

        Args:
            latent_vector: numpy array of shape (256,)
            deterministic: If True, use greedy action selection.

        Returns:
            (action: int, _state: any)
            Action mapping: 0=LONG, 1=SHORT, 2=EXIT/HOLD
        """
        latent_vector = latent_vector.astype(np.float32).flatten()
        if len(latent_vector) != 256:
            raise ValueError(f"Latent vector must be 256-dim, got {len(latent_vector)}")

        action, _state = self.model.predict(latent_vector, deterministic=deterministic)
        return int(action), _state

    def save(self, filepath: str) -> None:
        """Save PPO model to filepath."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        self.logger.info(f"Saved RL agent to {filepath}")

    def load(self, filepath: str) -> None:
        """Load PPO model from filepath."""
        self.model = PPO.load(filepath, device=self.device)
        self.logger.info(f"Loaded RL agent from {filepath}")


# ============================================================================
# HindsightExperienceReplay
# ============================================================================

class HindsightExperienceReplay:
    """Replays completed trades to generate counterfactual training examples.

    For each losing trade, it asks:
    - What if exit was earlier? (5, 10, 20 ticks earlier)
    - What if exit was later? (5, 10, 20 ticks later)
    - What if no entry was taken?
    """

    def __init__(self, agent: RLAgent, storage=None):
        self.agent = agent
        self.storage = storage
        self.logger = get_logger()

    def replay_trade(self, trade_record: dict) -> List[dict]:
        """Replay a single trade with alternative outcomes.

        Args:
            trade_record: dict with keys: ticker, direction, entry_time, entry_price,
                         exit_time, exit_price, pnl

        Returns:
            List of alternative outcome dictionaries.
        """
        alternatives = []

        entry_price = trade_record.get("entry_price", 0)
        exit_price = trade_record.get("exit_price", 0)
        direction = trade_record.get("direction", "")
        actual_pnl = trade_record.get("pnl", 0)

        if entry_price == 0 or exit_price == 0:
            return alternatives

        # Calculate price movement per tick (approximate as 0.1% of entry)
        tick_size = entry_price * 0.001

        # Alternative exits: earlier
        for ticks_earlier in [5, 10, 20]:
            alt_exit = exit_price
            if direction == "LONG":
                alt_exit = exit_price - (tick_size * ticks_earlier)  # Worse for long
            else:
                alt_exit = exit_price + (tick_size * ticks_earlier)  # Worse for short

            alt_pnl = self._calc_alt_pnl(entry_price, alt_exit, direction)
            alternatives.append({
                "scenario": f"exit_{ticks_earlier}_ticks_earlier",
                "exit_price": alt_exit,
                "pnl": alt_pnl,
                "pnl_diff": alt_pnl - actual_pnl,
            })

        # Alternative exits: later
        for ticks_later in [5, 10, 20]:
            alt_exit = exit_price
            if direction == "LONG":
                alt_exit = exit_price + (tick_size * ticks_later)
            else:
                alt_exit = exit_price - (tick_size * ticks_later)

            alt_pnl = self._calc_alt_pnl(entry_price, alt_exit, direction)
            alternatives.append({
                "scenario": f"exit_{ticks_later}_ticks_later",
                "exit_price": alt_exit,
                "pnl": alt_pnl,
                "pnl_diff": alt_pnl - actual_pnl,
            })

        # Alternative: no entry
        alternatives.append({
            "scenario": "no_entry",
            "exit_price": 0,
            "pnl": 0.0,
            "pnl_diff": -actual_pnl,  # What we saved/lost by not entering
        })

        return alternatives

    def _calc_alt_pnl(self, entry: float, exit_price: float, direction: str) -> float:
        """Calculate alternative P&L as a percentage."""
        if entry == 0:
            return 0.0
        if direction == "LONG":
            return (exit_price - entry) / entry * 100.0
        else:
            return (entry - exit_price) / entry * 100.0

    def replay_session(self, session_trades: List[dict]) -> dict:
        """Replay all trades from a session and aggregate insights.

        Returns summary of alternative outcomes.
        """
        summary = {
            "total_trades": len(session_trades),
            "total_actual_pnl": sum(t.get("pnl", 0) for t in session_trades),
            "best_alt_total_pnl": 0.0,
            "earlier_exit_better": 0,
            "later_exit_better": 0,
            "no_entry_better": 0,
            "alternatives": [],
        }

        total_alt_best = 0.0

        for trade in session_trades:
            alts = self.replay_trade(trade)
            trade_summary = {
                "trade_pnl": trade.get("pnl", 0),
                "alternatives": alts,
            }
            summary["alternatives"].append(trade_summary)

            # Count which scenario was best
            all_outcomes = [{"scenario": "actual", "pnl": trade.get("pnl", 0)}]
            all_outcomes.extend(alts)

            best = max(all_outcomes, key=lambda x: x["pnl"])
            total_alt_best += best["pnl"]

            if best["scenario"].startswith("exit_") and "earlier" in best["scenario"]:
                summary["earlier_exit_better"] += 1
            elif best["scenario"].startswith("exit_") and "later" in best["scenario"]:
                summary["later_exit_better"] += 1
            elif best["scenario"] == "no_entry":
                summary["no_entry_better"] += 1

        summary["best_alt_total_pnl"] = total_alt_best

        self.logger.info(
            f"Hindsight replay: {summary['total_trades']} trades, "
            f"actual PnL: {summary['total_actual_pnl']:.2f}, "
            f"best alt PnL: {total_alt_best:.2f}"
        )
        self.logger.info(
            f"Earlier exit better: {summary['earlier_exit_better']}, "
            f"Later exit better: {summary['later_exit_better']}, "
            f"No entry better: {summary['no_entry_better']}"
        )

        return summary
