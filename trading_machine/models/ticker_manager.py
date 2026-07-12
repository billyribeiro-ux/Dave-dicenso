"""
models/ticker_manager.py — Per-ticker model orchestration.

Each ticker gets its own COMPLETELY SEPARATE PriceVAE and RLAgent.
No shared weights. No transfer learning. The machine learns each
instrument independently.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.world_model import PriceVAE, WorldModelTrainer, prepare_dataloaders
from models.rl_agent import RLAgent
from utils.config import get_config, Config
from utils.logger import get_logger


class TickerManager:
    """Manages world models and RL agents for all tickers.

    Each ticker's models are stored in a completely isolated dictionary entry:
      {
        "world_model": PriceVAE instance,
        "rl_agent": RLAgent instance,
        "trained": bool,
        "version": str,
        "world_model_path": str | None,
        "rl_agent_path": str | None,
      }
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Ticker registry: ticker -> model dict
        self._registry: Dict[str, dict] = {}

        # Initialize entries for all configured tickers
        for ticker in config.TICKERS:
            self._registry[ticker] = {
                "world_model": None,
                "rl_agent": None,
                "trained": False,
                "version": "1.0.0",
                "world_model_path": None,
                "rl_agent_path": None,
            }

        # Ensure model save directory exists
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_ticker(self, ticker: str) -> bool:
        """Create new PriceVAE and RLAgent for a ticker.

        Returns True on success, False if ticker not in registry.
        """
        if ticker not in self._registry:
            self.logger.error(f"Ticker {ticker} not in configured TICKERS list")
            return False

        entry = self._registry[ticker]

        # Create world model
        world_model = PriceVAE(
            input_dim=1,
            latent_dim=self.config.LATENT_DIM,
            sequence_length=self.config.INPUT_WINDOW_TICKS,
            future_length=self.config.FUTURE_WINDOW_TICKS,
        ).to(self.device)

        # Create RL agent (world model set later after training)
        rl_agent = RLAgent(
            ticker=ticker,
            world_model=world_model,
            device=self.device,
        )

        entry["world_model"] = world_model
        entry["rl_agent"] = rl_agent
        entry["trained"] = False

        self.logger.info(f"Initialized models for {ticker} (v{entry['version']})")
        return True

    # ------------------------------------------------------------------
    # World model training
    # ------------------------------------------------------------------

    def train_world_model(
        self, ticker: str, prices_array: np.ndarray, epochs: int = 100
    ) -> dict:
        """Train the PriceVAE for a single ticker.

        Args:
            ticker: Ticker symbol.
            prices_array: 1D numpy array of close prices.
            epochs: Number of training epochs.

        Returns:
            Training history dictionary.
        """
        if ticker not in self._registry:
            raise ValueError(f"Ticker {ticker} not registered")

        entry = self._registry[ticker]
        if entry["world_model"] is None:
            self.initialize_ticker(ticker)

        world_model = entry["world_model"]

        # Prepare dataloaders
        train_loader, val_loader = prepare_dataloaders(
            prices_array,
            batch_size=self.config.BATCH_SIZE,
            sequence_length=self.config.INPUT_WINDOW_TICKS,
            future_length=self.config.FUTURE_WINDOW_TICKS,
            train_split=0.8,
        )

        self.logger.info(
            f"Training world model for {ticker}: "
            f"{len(train_loader.dataset)} train windows, "
            f"{len(val_loader.dataset)} val windows"
        )

        # Train
        checkpoint_dir = os.path.join(
            self.config.MODEL_DIR, ticker
        )
        trainer = WorldModelTrainer(world_model, device=self.device)
        history = trainer.train_full(
            train_loader,
            val_loader,
            epochs=epochs,
            patience=10,
            checkpoint_dir=checkpoint_dir,
        )

        # Save trained model
        version = entry["version"]
        save_path = os.path.join(
            checkpoint_dir, f"world_model_v{version}.pt"
        )
        torch.save(world_model.state_dict(), save_path)
        entry["world_model_path"] = save_path
        entry["trained"] = True

        self.logger.info(
            f"Saved world model for {ticker} v{version} to {save_path}"
        )

        return history

    # ------------------------------------------------------------------
    # RL agent training
    # ------------------------------------------------------------------

    def train_rl_agent(
        self, ticker: str, prices_array: np.ndarray, timesteps: int = 1_000_000
    ) -> dict:
        """Train the RL agent for a single ticker.

        Args:
            ticker: Ticker symbol.
            prices_array: 1D numpy array of close prices.
            timesteps: Total PPO training timesteps.

        Returns:
            Training history dictionary.
        """
        if ticker not in self._registry:
            raise ValueError(f"Ticker {ticker} not registered")

        entry = self._registry[ticker]
        if entry["world_model"] is None:
            raise ValueError(f"World model not initialized for {ticker}. Call initialize_ticker first.")

        if not entry["trained"]:
            self.logger.warning(
                f"World model for {ticker} not marked as trained. "
                "Training anyway with existing model."
            )

        world_model = entry["world_model"]
        world_model.eval()

        rl_agent = entry["rl_agent"]
        if rl_agent is None:
            rl_agent = RLAgent(ticker=ticker, world_model=world_model, device=self.device)
            entry["rl_agent"] = rl_agent

        history = rl_agent.train_on_historical(prices_array, timesteps, world_model)

        # Save trained agent
        checkpoint_dir = os.path.join(self.config.MODEL_DIR, ticker)
        version = entry["version"]
        save_path = os.path.join(checkpoint_dir, f"rl_agent_v{version}.zip")
        rl_agent.save(save_path)
        entry["rl_agent_path"] = save_path

        self.logger.info(f"Saved RL agent for {ticker} v{version} to {save_path}")

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_latent_state(
        self, ticker: str, price_window: np.ndarray
    ) -> np.ndarray:
        """Encode a price window into the 256-dim latent state.

        Args:
            ticker: Ticker symbol.
            price_window: numpy array of shape (500,) — 500 recent close prices.

        Returns:
            numpy array of shape (256,) — the latent state vector.
        """
        if ticker not in self._registry:
            raise ValueError(f"Ticker {ticker} not registered")

        entry = self._registry[ticker]
        world_model = entry["world_model"]
        if world_model is None:
            raise ValueError(f"World model not initialized for {ticker}")

        world_model.eval()
        world_model.to(self.device)

        # Normalize price window
        price_window = price_window.astype(np.float32).flatten()
        if len(price_window) != self.config.INPUT_WINDOW_TICKS:
            raise ValueError(
                f"Price window must be {self.config.INPUT_WINDOW_TICKS} ticks, "
                f"got {len(price_window)}"
            )

        mean = price_window.mean()
        std = price_window.std()
        if std < 1e-8:
            std = 1.0
        normalized = (price_window - mean) / std

        # Convert to tensor
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self.device)  # (1, 500, 1)

        with torch.no_grad():
            mu, logvar, z = world_model.encode(x)

        return z.cpu().numpy().flatten()  # (256,)

    def get_signal(
        self, ticker: str, latent_vector: np.ndarray
    ) -> Tuple[int, float]:
        """Get trading signal from latent state.

        Args:
            ticker: Ticker symbol.
            latent_vector: numpy array of shape (256,)

        Returns:
            (action_int, confidence_score)
            action_int: 0=LONG, 1=SHORT, 2=NEUTRAL
            confidence_score: probability of chosen action
        """
        if ticker not in self._registry:
            raise ValueError(f"Ticker {ticker} not registered")

        entry = self._registry[ticker]
        rl_agent = entry["rl_agent"]
        if rl_agent is None:
            raise ValueError(f"RL agent not initialized for {ticker}")

        action, _ = rl_agent.predict_action(latent_vector, deterministic=True)

        # Get action probabilities for confidence
        latent_vector = latent_vector.astype(np.float32).flatten()
        obs_tensor = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract policy distribution from the PPO model
            obs_tensor = obs_tensor.to(self.device)
            # SB3 PPO stores the policy in self.model.policy
            distribution = rl_agent.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy().flatten()
            confidence = float(probs[action])

        return int(action), confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all_models(self) -> None:
        """Save world model and RL agent for each ticker."""
        for ticker, entry in self._registry.items():
            if entry["world_model"] is not None:
                checkpoint_dir = os.path.join(self.config.MODEL_DIR, ticker)
                os.makedirs(checkpoint_dir, exist_ok=True)
                version = entry["version"]

                wm_path = os.path.join(checkpoint_dir, f"world_model_v{version}.pt")
                torch.save(entry["world_model"].state_dict(), wm_path)
                entry["world_model_path"] = wm_path
                self.logger.info(f"Saved world model for {ticker} to {wm_path}")

            if entry["rl_agent"] is not None:
                checkpoint_dir = os.path.join(self.config.MODEL_DIR, ticker)
                rl_path = os.path.join(checkpoint_dir, f"rl_agent_v{entry['version']}.zip")
                entry["rl_agent"].save(rl_path)
                entry["rl_agent_path"] = rl_path
                self.logger.info(f"Saved RL agent for {ticker} to {rl_path}")

    def load_all_models(self) -> None:
        """Load the latest version of models for each ticker.

        If no model exists for a ticker, marks it as untrained.
        """
        for ticker in self._registry:
            checkpoint_dir = os.path.join(self.config.MODEL_DIR, ticker)
            if not os.path.isdir(checkpoint_dir):
                self._registry[ticker]["trained"] = False
                continue

            # Find latest world model
            wm_files = sorted([
                f for f in os.listdir(checkpoint_dir)
                if f.startswith("world_model_v") and f.endswith(".pt")
            ])
            if wm_files:
                latest_wm = wm_files[-1]
                wm_path = os.path.join(checkpoint_dir, latest_wm)

                entry = self._registry[ticker]
                if entry["world_model"] is None:
                    self.initialize_ticker(ticker)

                entry["world_model"].load_state_dict(
                    torch.load(wm_path, map_location=self.device)
                )
                entry["world_model"].eval()
                entry["world_model_path"] = wm_path

                # Extract version from filename
                version = latest_wm.replace("world_model_v", "").replace(".pt", "")
                entry["version"] = version
                entry["trained"] = True

                self.logger.info(f"Loaded world model for {ticker} v{version} from {wm_path}")

            # Find latest RL agent
            rl_files = sorted([
                f for f in os.listdir(checkpoint_dir)
                if f.startswith("rl_agent_v") and f.endswith(".zip")
            ])
            if rl_files:
                latest_rl = rl_files[-1]
                rl_path = os.path.join(checkpoint_dir, latest_rl)

                entry = self._registry[ticker]
                if entry["rl_agent"] is None:
                    self.initialize_ticker(ticker)

                entry["rl_agent"].load(rl_path)
                entry["rl_agent_path"] = rl_path

                self.logger.info(f"Loaded RL agent for {ticker} from {rl_path}")
            else:
                self._registry[ticker]["trained"] = False

    def get_model_versions(self) -> Dict[str, str]:
        """Return dictionary mapping ticker to current model version."""
        return {
            ticker: entry["version"]
            for ticker, entry in self._registry.items()
        }

    def is_ticker_ready(self, ticker: str) -> bool:
        """Check if a ticker has trained models ready."""
        if ticker not in self._registry:
            return False
        entry = self._registry[ticker]
        return (
            entry["trained"]
            and entry["world_model"] is not None
            and entry["rl_agent"] is not None
        )

    def get_ready_tickers(self) -> List[str]:
        """Return list of tickers with trained models ready."""
        return [t for t in self._registry if self.is_ticker_ready(t)]
