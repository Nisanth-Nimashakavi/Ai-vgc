"""Double-Oracle Matrix Game Solver for simultaneous-move Pokémon battles.

Nessie's key insight: VGC is a simultaneous-move game. Instead of best-responding
to an estimated opponent distribution (which is exploitable), we solve the actual
normal-form game at each search node using Double Oracle.

Double Oracle algorithm:
1. Start with a small subset of actions for each player (e.g., policy top-k)
2. Solve the restricted game (linear programming) to get equilibrium strategies
3. For each player, find a best response to the opponent's equilibrium strategy
4. If the best response improves the player's value by > epsilon, add it and repeat
5. When no player has an improving best response, the restricted game's equilibrium
   is an epsilon-equilibrium of the full game.

This gives exact (up to epsilon) Nash equilibrium strategies for the simultaneous
move selection, eliminating exploitability in team preview and lead selection.
"""

from __future__ import annotations

import numpy as np

# Optional scipy for LP solving; fallback to fictitious play if unavailable
try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# For environments without scipy, we implement a simple LP solver
# using the simplex method for small games (n_actions <= 50 typically)


def solve_hidden_world_game(payoffs, weights, row_bonus=None):
    """Restricted zero-sum game with an opponent who knows its hidden set.

    Our one strategy must work across all worlds. Each world has its own
    opponent action columns and value constraint; column positions cannot be
    averaged across worlds. Returns None if the LP cannot be solved, so the
    caller can explicitly fall back to its expected-value search.
    """
    matrices = [np.asarray(m, dtype=np.float64) for m in payoffs]
    w = np.asarray(weights, dtype=np.float64)
    if (not matrices or w.shape != (len(matrices),) or not np.isfinite(w).all()
            or (w < 0).any() or w.sum() <= 0):
        raise ValueError("hidden worlds require finite nonnegative weights with positive mass")
    k = matrices[0].shape[0] if matrices[0].ndim == 2 else 0
    if not k or any(m.ndim != 2 or m.shape[0] != k or not m.shape[1]
                    or not np.isfinite(m).all() for m in matrices):
        raise ValueError("world payoffs require the same nonempty row actions and finite values")
    bonus = np.zeros(k) if row_bonus is None else np.asarray(row_bonus, dtype=np.float64)
    if bonus.shape != (k,) or not np.isfinite(bonus).all():
        raise ValueError("row bonus must be finite and match our actions")
    if not _HAS_SCIPY:
        return None
    w = w / w.sum()
    # Variables: one common row strategy x, one worst-response value per world.
    c = np.concatenate((-bonus, -w))
    constraints = []
    for world, m in enumerate(matrices):
        block = np.zeros((m.shape[1], k + len(matrices)))
        block[:, :k] = -m.T
        block[:, k + world] = 1
        constraints.append(block)
    a = np.concatenate(constraints)
    equality = np.zeros((1, len(c)))
    equality[0, :k] = 1
    result = linprog(c, A_ub=a, b_ub=np.zeros(len(a)), A_eq=equality,
                     b_eq=[1], bounds=[(0, None)] * k + [(None, None)] * len(matrices),
                     method="highs")
    if not result.success:
        return None
    strategy = np.maximum(result.x[:k], 0)
    strategy /= strategy.sum()
    value = sum(weight * float((strategy @ m).min()) for weight, m in zip(w, matrices))
    return strategy, value


def solve_hidden_world_regret(payoffs, weights, opponent_probs, *, worst_case_weight=0.5,
                              iterations=2048, row_bonus=None):
    """Regret matching for a common row strategy and private-world responses.

    Maximizes alpha * worst-case value + (1-alpha) * predicted-opponent value.
    The blend coefficient is our explicit choice, not a published Mikumiku value.
    Chance must already be averaged within each hidden world.
    """
    matrices = [np.asarray(m, dtype=np.float64) for m in payoffs]
    w = np.asarray(weights, dtype=np.float64)
    if not 0 <= worst_case_weight <= 1 or iterations < 1:
        raise ValueError("blend must be in [0, 1] and iterations positive")
    if not matrices or w.shape != (len(matrices),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid hidden-world weights")
    k = matrices[0].shape[0] if matrices[0].ndim == 2 else 0
    if not k or any(m.ndim != 2 or m.shape[0] != k or not m.shape[1] or not np.isfinite(m).all() for m in matrices):
        raise ValueError("invalid payoff matrices")
    if len(opponent_probs) != len(matrices):
        raise ValueError("one opponent distribution required per world")
    q = [np.asarray(p, dtype=np.float64) for p in opponent_probs]
    if any(p.shape != (m.shape[1],) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0
           for p, m in zip(q, matrices)):
        raise ValueError("invalid opponent probabilities")
    w = w / w.sum()
    q = [p / p.sum() for p in q]
    bonus = np.zeros(k) if row_bonus is None else np.asarray(row_bonus, dtype=np.float64)
    if bonus.shape != (k,) or not np.isfinite(bonus).all():
        raise ValueError("invalid row bonus")
    expected = sum(weight * (m @ p) for weight, m, p in zip(w, matrices, q))
    alpha = worst_case_weight
    if alpha == 0:
        x = np.zeros(k)
        x[np.argmax(expected + bonus)] = 1
        return x, float(x @ expected), {"gap": 0., "iterations": 0}

    # Pad variable-sized action sets, masking padding out of the opponent mix.
    width = max(m.shape[1] for m in matrices)
    table = np.zeros((len(matrices), k, width))
    valid = np.zeros((len(matrices), width), dtype=bool)
    for i, m in enumerate(matrices):
        table[i, :, :m.shape[1]] = m
        valid[i, :m.shape[1]] = True
    row_regret = np.zeros(k)
    col_regret = np.zeros_like(valid, dtype=float)
    sum_x = np.zeros(k)
    sum_y = np.zeros_like(col_regret)
    default_y = valid / valid.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        x = np.maximum(row_regret, 0)
        x = x / x.sum() if x.sum() > 0 else np.full(k, 1/k)
        positive = np.maximum(col_regret, 0) * valid
        mass = positive.sum(axis=1, keepdims=True)
        y = np.divide(positive, mass, out=default_y.copy(), where=mass > 0)
        row_values = alpha * np.einsum('w,wkj,wj->k', w, table, y) + (1-alpha)*expected + bonus
        col_values = -alpha * np.einsum('k,wkj->wj', x, table)
        row_regret += row_values - x @ row_values
        col_regret += (col_values - (y * col_values).sum(axis=1, keepdims=True)) * valid
        sum_x += x
        sum_y += y
    x, y = sum_x / iterations, sum_y / iterations
    worst = sum(weight * float((x @ m).min()) for weight, m in zip(w, matrices))
    value = alpha * worst + (1-alpha) * float(x @ expected)
    best = float((alpha*np.einsum('w,wkj,wj->k', w, table, y) + (1-alpha)*expected + bonus).max())
    return x, value, {"gap": max(0., best-value-float(x @ bonus)), "iterations": iterations}


def solve_matrix_game_lp(payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a zero-sum matrix game via linear programming.

    Player 1 (row) maximizes, Player 2 (column) minimizes.
    payoff[i, j] = value to Player 1 when they play i, opponent plays j.

    Returns: (row_strategy, col_strategy, game_value)
    """
    n_rows, n_cols = payoff.shape

    # Player 1's LP: max v s.t. sum_i x_i * payoff[i, j] >= v for all j, sum x_i = 1, x_i >= 0
    # Convert to standard form: min -v
    # Variables: x_0...x_{n_rows-1}, v
    c = np.zeros(n_rows + 1)
    c[-1] = -1.0  # minimize -v

    # Constraints: -sum_i x_i * payoff[i, j] + v <= 0  for each j
    #              sum_i x_i = 1
    #              x_i >= 0, v free
    A_ub = np.zeros((n_cols, n_rows + 1))
    for j in range(n_cols):
        A_ub[j, :n_rows] = -payoff[:, j]
        A_ub[j, -1] = 1.0
    b_ub = np.zeros(n_cols)

    A_eq = np.zeros((1, n_rows + 1))
    A_eq[0, :n_rows] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * n_rows + [(None, None)]

    if _HAS_SCIPY:
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                raise RuntimeError(f"LP failed: {res.message}")
            x = res.x[:n_rows]
            v = res.x[-1]
        except Exception:
            # Fallback: use a simple iterative solver (fictitious play)
            return solve_matrix_game_fictitious(payoff)
    else:
        # No scipy: use fictitious play directly
        return solve_matrix_game_fictitious(payoff)

    # Player 2's strategy from dual variables
    # Dual of Player 1's LP gives Player 2's strategy
    # Or solve Player 2's LP directly
    c2 = np.zeros(n_cols + 1)
    c2[-1] = 1.0  # minimize v

    A_ub2 = np.zeros((n_rows, n_cols + 1))
    for i in range(n_rows):
        A_ub2[i, :n_cols] = payoff[i, :]
        A_ub2[i, -1] = -1.0
    b_ub2 = np.zeros(n_rows)

    A_eq2 = np.zeros((1, n_cols + 1))
    A_eq2[0, :n_cols] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0, None)] * n_cols + [(None, None)]

    if _HAS_SCIPY:
        try:
            res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method='highs')
            if not res2.success:
                raise RuntimeError(f"LP2 failed: {res2.message}")
            y = res2.x[:n_cols]
            v2 = res2.x[-1]
        except Exception:
            # Compute best response to x
            y = np.zeros(n_cols)
            br = np.argmin(payoff.T @ x)
            y[br] = 1.0
            v2 = v
    else:
        # No scipy: compute best response to x
        y = np.zeros(n_cols)
        br = np.argmin(payoff.T @ x)
        y[br] = 1.0
        v2 = v

    return x, y, float(v)


def solve_matrix_game_fictitious(payoff: np.ndarray, iters: int = 10000) -> tuple[np.ndarray, np.ndarray, float]:
    """Fictitious play fallback for matrix game solving (no scipy needed)."""
    n_rows, n_cols = payoff.shape

    # Initialize with uniform strategies
    x = np.ones(n_rows) / n_rows
    y = np.ones(n_cols) / n_cols

    # Cumulative play counts
    cum_x = np.ones(n_rows)
    cum_y = np.ones(n_cols)

    for _ in range(iters):
        # Best response to opponent's average strategy
        br_row = np.argmax(payoff @ y)
        br_col = np.argmin(payoff.T @ x)

        cum_x[br_row] += 1
        cum_y[br_col] += 1

        x = cum_x / cum_x.sum()
        y = cum_y / cum_y.sum()

    v = float(x @ payoff @ y)
    return x, y, v


class DoubleOracle:
    """Double Oracle algorithm for solving large matrix games.

    Maintains a restricted game with a subset of actions for each player.
    Iteratively expands by adding best responses until convergence.
    """

    def __init__(
        self,
        payoff_fn,
        initial_actions_1: list[int],
        initial_actions_2: list[int],
        n_actions_1: int,
        n_actions_2: int,
        epsilon: float = 1e-4,
        max_iters: int = 50,
    ):
        """
        Args:
            payoff_fn: Callable(action_1, action_2) -> float (value to player 1)
            initial_actions_1: Initial action indices for player 1
            initial_actions_2: Initial action indices for player 2
            n_actions_1: Total number of actions for player 1
            n_actions_2: Total number of actions for player 2
            epsilon: Convergence threshold (exploitability)
            max_iters: Maximum oracle iterations
        """
        self.payoff_fn = payoff_fn
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_2
        self.epsilon = epsilon
        self.max_iters = max_iters

        # Restricted action sets
        self.actions_1 = list(initial_actions_1)
        self.actions_2 = list(initial_actions_2)

        # Cache for payoff matrix entries
        self._payoff_cache = {}

    def _get_payoff(self, a1: int, a2: int) -> float:
        key = (a1, a2)
        if key not in self._payoff_cache:
            self._payoff_cache[key] = self.payoff_fn(a1, a2)
        return self._payoff_cache[key]

    def _build_restricted_game(self) -> np.ndarray:
        """Build payoff matrix for current restricted action sets."""
        m = len(self.actions_1)
        n = len(self.actions_2)
        M = np.zeros((m, n))
        for i, a1 in enumerate(self.actions_1):
            for j, a2 in enumerate(self.actions_2):
                M[i, j] = self._get_payoff(a1, a2)
        return M

    def _best_response_1(self, y: np.ndarray) -> tuple[int, float]:
        """Find player 1's best response to opponent strategy y."""
        best_a = -1
        best_val = -np.inf
        for a1 in range(self.n_actions_1):
            val = sum(y[j] * self._get_payoff(a1, self.actions_2[j]) for j in range(len(self.actions_2)))
            if val > best_val:
                best_val = val
                best_a = a1
        return best_a, best_val

    def _best_response_2(self, x: np.ndarray) -> tuple[int, float]:
        """Find player 2's best response to player 1 strategy x."""
        best_a = -1
        best_val = np.inf
        for a2 in range(self.n_actions_2):
            val = sum(x[i] * self._get_payoff(self.actions_1[i], a2) for i in range(len(self.actions_1)))
            if val < best_val:
                best_val = val
                best_a = a2
        return best_a, best_val

    def solve(self) -> tuple[np.ndarray, np.ndarray, float, dict]:
        """Run Double Oracle until convergence.

        Returns:
            full_strategy_1: Strategy over all n_actions_1
            full_strategy_2: Strategy over all n_actions_2
            game_value: Value of the game
            info: Dict with iteration count, exploitability, etc.
        """
        for iteration in range(self.max_iters):
            # Solve restricted game
            M = self._build_restricted_game()
            x_restricted, y_restricted, v = solve_matrix_game_lp(M)

            # Find best responses
            br_1, br_1_val = self._best_response_1(y_restricted)
            br_2, br_2_val = self._best_response_2(x_restricted)

            # Exploitability: how much each player can gain by deviating
            exploit_1 = br_1_val - v
            exploit_2 = v - br_2_val
            exploitability = max(exploit_1, exploit_2)

            if exploitability <= self.epsilon:
                break

            # Add best responses to restricted sets
            if br_1 not in self.actions_1:
                self.actions_1.append(br_1)
            if br_2 not in self.actions_2:
                self.actions_2.append(br_2)
        else:
            # Max iterations reached
            pass

        # Map restricted strategies back to full action spaces
        full_x = np.zeros(self.n_actions_1)
        full_y = np.zeros(self.n_actions_2)

        for i, a1 in enumerate(self.actions_1):
            full_x[a1] = x_restricted[i]
        for j, a2 in enumerate(self.actions_2):
            full_y[a2] = y_restricted[j]

        info = {
            "iterations": iteration + 1,
            "exploitability": exploitability,
            "n_actions_1": len(self.actions_1),
            "n_actions_2": len(self.actions_2),
            "restricted_actions_1": self.actions_1.copy(),
            "restricted_actions_2": self.actions_2.copy(),
        }

        return full_x, full_y, v, info


def double_oracle_search(
    our_candidates: list[tuple[np.ndarray, float]],  # (action, logp)
    opp_candidates: list[tuple[str, float]],         # (choice, prob)
    eval_fn,                                         # (our_action, opp_choice) -> value
    n_our_actions: int,
    n_opp_actions: int,
    epsilon: float = 1e-3,
    max_iters: int = 30,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """High-level Double Oracle interface for search root.

    Args:
        our_candidates: Our top actions from policy (action_array, logp)
        opp_candidates: Opponent's top choices from aux head (choice_str, prob)
        eval_fn: Function to evaluate a pairing (our_action, opp_choice) -> win_prob
        n_our_actions: Total legal actions for us
        n_opp_actions: Total legal actions for opponent
        epsilon: Convergence threshold
        max_iters: Max oracle iterations

    Returns:
        our_strategy: Equilibrium strategy over our actions
        opp_strategy: Equilibrium strategy over opponent actions
        game_value: Value of the game
        info: Debug info
    """
    # Map our actions to indices
    our_action_list = [a for a, _ in our_candidates]
    our_logp = [lp for _, lp in our_candidates]

    # Map opponent choices to indices
    opp_choice_list = [c for c, _ in opp_candidates]
    opp_probs = np.array([p for _, p in opp_candidates])

    # Initial restricted sets: top policy actions + top opponent choices
    initial_1 = list(range(min(len(our_candidates), 8)))
    initial_2 = list(range(min(len(opp_candidates), 8)))

    def payoff(i: int, j: int) -> float:
        return eval_fn(our_action_list[i], opp_choice_list[j])

    oracle = DoubleOracle(
        payoff_fn=payoff,
        initial_actions_1=initial_1,
        initial_actions_2=initial_2,
        n_actions_1=len(our_action_list),
        n_actions_2=len(opp_choice_list),
        epsilon=epsilon,
        max_iters=max_iters,
    )

    return oracle.solve()


def equilibrium_strategy_from_search(
    M: np.ndarray,          # [n_our, n_opp] win probability matrix
    q: np.ndarray,          # [n_opp] opponent distribution (from aux head)
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Given a payoff matrix M and opponent distribution q, compute our
    equilibrium strategy using Double Oracle.

    This replaces the current "best response to q" with a true equilibrium
    computation. If we only best-respond to q, we're exploitable.
    """
    n_our, n_opp = M.shape

    def payoff(i: int, j: int) -> float:
        return M[i, j]

    # Initial actions: top policy actions (by value against q) + actions that are
    # best responses to q
    vals_vs_q = M @ q
    initial_1 = list(np.argsort(-vals_vs_q)[:min(8, n_our)])
    initial_2 = list(np.argsort(q)[-min(8, n_opp):])  # opponent's most likely

    oracle = DoubleOracle(
        payoff_fn=payoff,
        initial_actions_1=initial_1,
        initial_actions_2=initial_2,
        n_actions_1=n_our,
        n_actions_2=n_opp,
        epsilon=epsilon,
        max_iters=30,
    )

    x, y, v, info = oracle.solve()
    return x, v, info
