"""matrix_surgeon.py
Core logic for the Matrix Surgeon extension game.

The game teaches row operations by explicitly building elementary matrices E and
updating A via left multiplication: A_new = E @ A.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Move:
    """Represents one player move."""

    op_type: str
    row_a: int
    row_b: Optional[int]
    k: Optional[float]
    E: np.ndarray
    before: np.ndarray
    after: np.ndarray

# Elementary matrix constructors

def elementary_swap(n: int, i: int, j: int) -> np.ndarray:
    """Return E for swapping rows i and j (0-based)."""
    if i == j:
        raise ValueError("Swap rows must be different.")
    E = np.eye(n)
    E[[i, j]] = E[[j, i]]
    return E


def elementary_scale(n: int, i: int, k: float) -> np.ndarray:
    """Return E for scaling row i by nonzero k (0-based)."""
    if np.isclose(k, 0.0):
        raise ValueError("Scale factor k must be nonzero.")
    E = np.eye(n)
    E[i, i] = float(k)
    return E


def elementary_add(n: int, src: int, dst: int, k: float) -> np.ndarray:
    """Return E for operation: R_dst <- R_dst + k * R_src (0-based)."""
    if src == dst:
        raise ValueError("Source and destination rows must be different.")
    E = np.eye(n)
    E[dst, src] = float(k)
    return E


def apply_operation(
    A: np.ndarray,
    op_type: str,
    row_a: int,
    row_b: Optional[int] = None,
    k: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one row operation by left multiplication and return (A_new, E)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")

    n = A.shape[0]
    if not (0 <= row_a < n):
        raise ValueError("row_a out of range.")

    if op_type == "swap":
        if row_b is None or not (0 <= row_b < n):
            raise ValueError("swap requires a valid row_b.")
        E = elementary_swap(n, row_a, row_b)
    elif op_type == "scale":
        if k is None:
            raise ValueError("scale requires scalar k.")
        E = elementary_scale(n, row_a, float(k))
    elif op_type == "add":
        if row_b is None or not (0 <= row_b < n):
            raise ValueError("add requires a valid row_b.")
        if k is None:
            raise ValueError("add requires scalar k.")
        E = elementary_add(n, row_a, row_b, float(k))
    else:
        raise ValueError("Unsupported op_type. Use 'swap', 'scale', or 'add'.")

    return E @ A, E


def move_to_text(move: Move) -> str:
    """Return a short human-readable string for a move (1-based row display)."""
    a = move.row_a + 1
    if move.op_type == "swap":
        b = (move.row_b or 0) + 1
        return f"swap R{a} <-> R{b}"
    if move.op_type == "scale":
        return f"scale R{a} by {move.k:g}"
    b = (move.row_b or 0) + 1
    return f"R{b} <- R{b} + ({move.k:g}) * R{a}"

# Game state manager

class MatrixSurgeonGame:
    """Stateful game manager for Matrix Surgeon."""

    def __init__(self, n: int = 3, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.n = int(n)

        self.start_A: np.ndarray = np.eye(self.n)
        self.goal_G: np.ndarray = np.eye(self.n)
        self.current_A: np.ndarray = np.eye(self.n)

        self.history: List[Move] = []
        self.start_time = time.time()

        self.new_puzzle(n=self.n)

    def _random_invertible_matrix(self, n: int) -> np.ndarray:
        """Generate a random full-rank integer matrix."""
        for _ in range(1000):
            A = self.rng.integers(-4, 5, size=(n, n)).astype(float)
            if np.linalg.matrix_rank(A) == n:
                return A
        raise RuntimeError("Failed to generate invertible matrix after many attempts.")

    def new_puzzle(self, n: Optional[int] = None) -> None:
        """Create a fresh puzzle with start matrix A and goal matrix G."""
        if n is not None:
            self.n = int(n)

        self.start_A = self._random_invertible_matrix(self.n)
        self.goal_G = self._random_invertible_matrix(self.n)
        while np.allclose(self.goal_G, self.start_A):
            self.goal_G = self._random_invertible_matrix(self.n)

        self.current_A = self.start_A.copy()
        self.history = []
        self.start_time = time.time()

    def apply(self, op_type: str, row_a: int, row_b: Optional[int] = None, k: Optional[float] = None) -> Move:
        """Apply one player operation and append to move history."""
        before = self.current_A.copy()
        after, E = apply_operation(before, op_type, row_a, row_b=row_b, k=k)

        move = Move(
            op_type=op_type,
            row_a=int(row_a),
            row_b=None if row_b is None else int(row_b),
            k=None if k is None else float(k),
            E=E,
            before=before,
            after=after,
        )

        self.current_A = after
        self.history.append(move)
        return move

    def undo(self) -> Optional[Move]:
        """Undo one move by restoring its 'before' matrix."""
        if not self.history:
            return None
        move = self.history.pop()
        self.current_A = move.before.copy()
        return move

    def moves_used(self) -> int:
        return len(self.history)

    def elapsed_seconds(self) -> float:
        return float(time.time() - self.start_time)

    def solved(self, tol: float = 1e-8) -> bool:
        return bool(np.allclose(self.current_A, self.goal_G, atol=tol))

    def score(self) -> int:
        """Score rewards fewer moves and faster completion."""
        moves_penalty = 70 * self.moves_used()
        time_penalty = 2.0 * self.elapsed_seconds()
        raw = 1000.0 - moves_penalty - time_penalty
        return int(max(0.0, round(raw)))

# Optional local leaderboard

def load_scores(path: str | Path) -> List[dict]:
    """Load local scores from JSON. Returns [] on missing/corrupt file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        p.write_text("[]\n", encoding="utf-8")
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Score file must contain a list.")
    except Exception:
        p.write_text("[]\n", encoding="utf-8")
        return []

    cleaned = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            cleaned.append(
                {
                    "name": str(row["name"]),
                    "score": int(row["score"]),
                    "moves": int(row["moves"]),
                    "seconds": float(row["seconds"]),
                    "n": int(row["n"]),
                    "timestamp": str(row["timestamp"]),
                }
            )
        except Exception:
            continue

    cleaned.sort(key=lambda x: (-x["score"], x["moves"], x["seconds"]))
    return cleaned[:20]


def save_score(
    path: str | Path,
    name: str,
    score: int,
    moves: int,
    seconds: float,
    n: int,
) -> List[dict]:
    """Append one score and keep top 20 entries."""
    board = load_scores(path)
    entry = {
        "name": str(name).strip() if str(name).strip() else "Anonymous",
        "score": int(score),
        "moves": int(moves),
        "seconds": float(seconds),
        "n": int(n),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    board.append(entry)
    board.sort(key=lambda x: (-x["score"], x["moves"], x["seconds"]))
    board = board[:20]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    return board
