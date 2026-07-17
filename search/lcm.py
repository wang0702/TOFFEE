
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from toffee.config import (
    ALPHA_B,
    RIDGE_LAMBDA,
    D_ACTION,
    D_STATE,
)

log = logging.getLogger(__name__)


class FactoredLinUCB:

    def __init__(
        self,
        d_state: int = D_STATE,
        d_action: int = D_ACTION,
        alpha: float = ALPHA_B,
        lambda_reg: float = RIDGE_LAMBDA,
    ):
        self.d_state = d_state
        self.d_action = d_action
        self.D = (d_state + 1) * d_action
        self.alpha = alpha

        self.A = lambda_reg * np.eye(self.D, dtype=np.float64)
        self.A_inv = (1.0 / lambda_reg) * np.eye(self.D, dtype=np.float64)
        self.b = np.zeros(self.D, dtype=np.float64)

        self.task_count = 0
        self.update_count = 0
        self._lock = threading.Lock()

        log.info("LCM initialised: D=%d, alpha=%.2f", self.D, self.alpha)


    def _phi(self, x_state: np.ndarray, z_action: np.ndarray) -> np.ndarray:
        x_tilde = np.append(x_state, 1.0)
        return np.outer(x_tilde, z_action).ravel()


    def rank_actions(
        self, state_features: np.ndarray, feasible_actions: list,
    ) -> List[Tuple[float, float, object]]:
        x = state_features
        scored = []
        with self._lock:
            theta_hat = self.A_inv @ self.b
            for action in feasible_actions:
                z = action.feature_vector()
                phi = self._phi(x, z)
                mu = float(theta_hat @ phi)
                sigma = float(np.sqrt(np.clip(phi @ self.A_inv @ phi, 0, None)))
                scored.append((mu + self.alpha * sigma, sigma, action))

        scored.sort(key=lambda t: -t[0])
        return scored


    def update(self, x_state: np.ndarray, z_action: np.ndarray, reward: float,
               op_name: str = "") -> None:
        phi = self._phi(x_state, z_action)
        with self._lock:
            self.A += np.outer(phi, phi)
            self.b += reward * phi
            Ainv_phi = self.A_inv @ phi
            denom = 1.0 + float(phi @ Ainv_phi)
            if denom > 1e-12:
                self.A_inv = self.A_inv - np.outer(Ainv_phi, Ainv_phi) / denom
            else:
                self.A_inv = np.linalg.inv(self.A)
            if self.update_count % 200 == 199:
                self.A_inv = np.linalg.inv(self.A)
            self.update_count += 1


    def increment_task_count(self) -> None:
        with self._lock:
            self.task_count += 1
