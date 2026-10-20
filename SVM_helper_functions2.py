import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MinSVD:
    """
    Python port of the MATLAB minSVD(phi, epsilon, Z, E, p, p_max).

    Storage:
      - Z: (n, p_max) columns of phi (phi treated as length-n vector)
      - E: (m, p_max) columns of epsilon (m can be 1 if scalar)
      - p: number of filled columns (0..p_max)
      - SV: smallest singular value of Z_active

    Call update(phi, epsilon) each control-step.
    """
    p_max: int
    eta: float = 0.05

    # internal state
    Z: np.ndarray = field(init=False)
    E: np.ndarray = field(init=False)
    p: int = field(default=0, init=False)
    SV: float = field(default=0.0, init=False)

    # last-step diagnostics (optional but handy mid-loop)
    last_action: str = field(default="init", init=False)  # "append"|"replace"|"skip"
    last_k: Optional[int] = field(default=None, init=False)
    last_S_old: Optional[float] = field(default=None, init=False)
    last_S_max: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.p_max <= 0:
            raise ValueError("p_max must be positive.")

        # Lazily size Z/E on first update because we don't know phi/epsilon dims yet
        self.Z = np.empty((0, self.p_max), dtype=float)
        self.E = np.empty((0, self.p_max), dtype=float)

    def _ensure_dims(self, phi, epsilon):
        # phi can come in as (1,3), (3,), (3,1) -> store as (3,)
        phi = np.asarray(phi, dtype=float).reshape(-1)

        # epsilon can be scalar or vector -> store as (m,)
        epsilon = np.asarray(epsilon, dtype=float)
        if epsilon.ndim == 0:
            epsilon = epsilon.reshape(1)
        else:
            epsilon = epsilon.reshape(-1)

        if self.Z.size == 0:
            n = phi.size
            m = epsilon.size
            self.Z = np.zeros((n, self.p_max), dtype=float)
            self.E = np.zeros((m, self.p_max), dtype=float)
        else:
            if phi.size != self.Z.shape[0]:
                raise ValueError(
                    f"phi dim mismatch: got {phi.size}, expected {self.Z.shape[0]}"
                )
            if epsilon.size != self.E.shape[0]:
                raise ValueError(
                    f"epsilon dim mismatch: got {epsilon.size}, expected {self.E.shape[0]}"
                )

        return phi, epsilon

    @staticmethod
    def value_function(Z_active: np.ndarray) -> float:
        """
        MATLAB: real(sqrt(min(eig(Z*Z')))) == smallest singular value of Z
        """
        if Z_active.size == 0 or Z_active.shape[1] == 0:
            return 0.0
        s = np.linalg.svd(Z_active, compute_uv=False)
        return float(s[-1])

    def sum_components(self) -> np.ndarray:
        """
        Matrix accumulation:
            S = sum_{j=1..p} (phi_j * epsilon_j^T) = Z_active @ E_active.T

        Shapes:
          Z_active: (n, p)
          E_active: (m, p)
          S:        (n, m)
        """
        if self.Z.size == 0 or self.E.size == 0:
            return np.zeros((0, 0), dtype=float)

        n = self.Z.shape[0]
        m = self.E.shape[0]

        if self.p == 0:
            return np.zeros((n, m), dtype=float)

        Z_active = self.Z[:, :self.p]
        E_active = self.E[:, :self.p]
        return Z_active @ E_active.T

    def sum_components_scalar(self) -> float:
        """
        Controller-safe scalar version.

        Returns a Python float only when sum_components() is 1x1.
        - If Z/E not initialized yet or p==0, returns 0.0.
        - If the result is not 1x1, raises ValueError.
        """
        if self.Z.size == 0 or self.E.size == 0:
            return 0.0
        if self.p == 0:
            return 0.0

        out = self.sum_components()  # (n, m)

        if out.size == 0:
            return 0.0
        if out.size != 1:
            raise ValueError(
                f"sum_components is not scalar (shape={out.shape}). "
                "Use sum_phiT_eps() for vector adaptation or define a scalar reduction."
            )

        return float(out.squeeze())

    def sum_phiT_eps(self) -> np.ndarray:
        """
        Vector term for the xi adaptation law when epsilon is scalar:

            v = sum_{j=1..p} (phi_j^T * epsilon_j)

        With storage:
          - Z_active columns are phi_j^T (stored as vectors): shape (n, p)
          - epsilon_j scalar => E has shape (1, p_max)

        Returns:
          v shape (n,)
        """
        if self.Z.size == 0 or self.E.size == 0:
            return np.zeros((0,), dtype=float)

        n = self.Z.shape[0]

        if self.p == 0:
            return np.zeros((n,), dtype=float)

        if self.E.shape[0] != 1:
            raise ValueError(
                f"sum_phiT_eps expects scalar epsilon (E has 1 row), but E has shape {self.E.shape}."
            )

        Z_active = self.Z[:, :self.p]        # (n, p)
        w = self.E[0, :self.p]               # (p,)
        out = Z_active @ w                   # (n,)

        return out

    def update(self, phi, epsilon) -> float:
        """
        Update storage with new (phi, epsilon). Returns current SV.

        Implements:
          - If full (p >= p_max): try replacing one column to maximize value_function
          - Else: append if "interesting enough" by eta test
        """
        phi, epsilon = self._ensure_dims(phi, epsilon)

        self.last_k = None
        self.last_S_old = None
        self.last_S_max = None

        # If we have no points yet, always add first point
        if self.p == 0:
            self.Z[:, 0] = phi
            self.E[:, 0] = epsilon
            self.p = 1
            self.last_action = "append"
            self.SV = self.value_function(self.Z[:, :self.p])
            return self.SV

        # phi_p = last stored phi
        phi_p = self.Z[:, self.p - 1]

        if self.p >= self.p_max:
            # Full: consider replacing each column with phi, pick best
            S_old = self.value_function(self.Z[:, :self.p])
            self.last_S_old = S_old

            best_S = -np.inf
            best_k = 0

            for j in range(self.p):
                col_backup = self.Z[:, j].copy()
                self.Z[:, j] = phi
                Sj = self.value_function(self.Z[:, :self.p])
                self.Z[:, j] = col_backup

                if Sj > best_S:
                    best_S = Sj
                    best_k = j

            self.last_S_max = float(best_S)
            self.last_k = int(best_k)

            if best_S > S_old:
                self.Z[:, best_k] = phi
                self.E[:, best_k] = epsilon
                self.last_action = "replace"
            else:
                self.last_action = "skip"

        else:
            # Not full: append only if interesting enough
            denom = np.linalg.norm(phi)

            # guard against divide-by-zero
            if denom == 0.0:
                interesting = True
            else:
                interesting = (np.linalg.norm(phi - phi_p) ** 2) / denom >= self.eta

            if interesting:
                self.Z[:, self.p] = phi
                self.E[:, self.p] = epsilon
                self.p += 1
                self.last_action = "append"
            else:
                self.last_action = "skip"

        self.SV = self.value_function(self.Z[:, :self.p])
        return self.SV

    def get_state(self) -> Dict[str, Any]:
        """Convenience snapshot for debugging/logging mid-loop."""
        return {
            "p": self.p,
            "p_max": self.p_max,
            "SV": self.SV,
            "last_action": self.last_action,
            "last_k": self.last_k,
            "last_S_old": self.last_S_old,
            "last_S_max": self.last_S_max,
            "Z_active": self.Z[:, :self.p].copy(),
            "E_active": self.E[:, :self.p].copy(),
        }
