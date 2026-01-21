from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from one_hot import (
    comp_to_one_hot,
    element_to_one_hot,
    featurize_target,
    one_hot_to_comp,
    one_hot_to_element,
    step_to_one_hot,
)


DEFAULT_CATION_SET: List[str] = [
    "Mg",
    "Ca",
    "Sc",
    "Ti",
    "Cu",
    "Sr",
    "Y",
    "Zr",
    "Hf",
    "Bi",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Co",
    "Ni",
    "Fe",
    "Mn",
]

# Discrete fractions required by the user.
DEFAULT_FRACTIONS: List[str] = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35"]


def _format_fraction(units: int) -> str:
    # units are in 0.05 increments, i.e. units/20
    return f"{units / 20:.2f}"


def _fractions_to_units(fractions: Sequence[str]) -> List[int]:
    # Parse and map to integer units of 0.05.
    # We require exact multiples of 0.05.
    out: List[int] = []
    for f in fractions:
        val = float(f)
        units = int(round(val * 20))
        out.append(units)
    return out


def _possible_sums(units: Sequence[int], k: int, max_total: int) -> set[int]:
    # DP over k picks with repetition.
    sums = {0}
    for _ in range(k):
        next_sums: set[int] = set()
        for s in sums:
            for u in units:
                t = s + u
                if t <= max_total:
                    next_sums.add(t)
        sums = next_sums
    return sums


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float


class ABCDEOOHEnv:
    """Constrained 5-step environment for generating compositions of the form A B C D E + OOH.

    Requirements implemented:
    - exactly 5 cations (one per step)
    - cations are distinct, chosen from a user-provided cation set
    - each cation fraction is chosen from a discrete set
    - the 5 cation fractions sum to 1.0 (internally enforced in integer 0.05 units)

    Notes:
    - This environment appends `anion_formula` only when computing the terminal formula.
    - Reward is terminal-only, provided via reward_fn(terminal_formula).
    """

    def __init__(
        self,
        *,
        cation_set: Sequence[str] = DEFAULT_CATION_SET,
        fraction_set: Sequence[str] = DEFAULT_FRACTIONS,
        anion_formula: str = "O2H1",
        max_steps: int = 5,
        reward_fn: Callable[[str], float] | None = None,
    ) -> None:
        if max_steps != 5:
            raise ValueError("This environment is designed for exactly 5 cation steps (max_steps=5).")

        self.cation_set = list(cation_set)
        self.fraction_set = list(fraction_set)
        self.anion_formula = anion_formula
        self.max_steps = max_steps
        self.reward_fn = reward_fn or (lambda _formula: 0.0)

        self._allowed_units = _fractions_to_units(self.fraction_set)
        self._total_units = 20
        # Precompute which sums are achievable with k remaining picks.
        self._possible_sums_by_k: List[set[int]] = [
            _possible_sums(self._allowed_units, k, self._total_units) for k in range(self.max_steps + 1)
        ]

        self.state: str = ""
        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._selected: set[str] = set()
        self._used_units: int = 0

    def initialize(self) -> None:
        self.state = ""
        self.counter = 0
        self.path = []
        self._selected = set()
        self._used_units = 0

    @property
    def remaining_units(self) -> int:
        return self._total_units - self._used_units

    @property
    def terminal_formula(self) -> str:
        return f"{self.state}{self.anion_formula}" if self.counter == self.max_steps else ""

    def cation_fractions(self) -> Dict[str, float]:
        """Return current cation fractions parsed from the internal state string.

        The returned dict sums to 1.0 only at terminal.
        """
        if not self.state:
            return {}
        parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", self.state)
        out: Dict[str, float] = {}
        for el, frac in parts:
            out[el] = out.get(el, 0.0) + float(frac)
        return out

    def terminal_cation_fractions(self) -> Dict[str, float]:
        """Return terminal cation fractions; raises if called before terminal."""
        if self.counter != self.max_steps:
            raise RuntimeError("terminal_cation_fractions() called before episode termination")
        return self.cation_fractions()

    def _allowed_fraction_units_now(self) -> List[int]:
        steps_left = self.max_steps - self.counter
        remaining = self.remaining_units

        allowed: List[int] = []
        for u in self._allowed_units:
            if u > remaining:
                continue
            rem_after = remaining - u
            if rem_after in self._possible_sums_by_k[steps_left - 1]:
                allowed.append(u)
        return allowed

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return all valid actions given current partial composition."""
        if self.counter >= self.max_steps:
            return []

        elems = [e for e in self.cation_set if e not in self._selected]
        units = self._allowed_fraction_units_now()

        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        for elem in elems:
            elem_oh = tuple(element_to_one_hot([elem], element_set_override=self.cation_set)[0].tolist())
            for u in units:
                comp = _format_fraction(u)
                comp_oh = tuple(comp_to_one_hot([comp], comp_set_override=self.fraction_set)[0].tolist())
                actions.append((elem_oh, comp_oh))
        return actions

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        actions = self.allowed_actions()
        if not actions:
            raise RuntimeError("No valid actions available; check fraction set and constraints.")
        return random.choice(actions)

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        elem_oh, comp_oh = action
        elem = one_hot_to_element([elem_oh], element_set_override=self.cation_set)[0]
        comp_str = one_hot_to_comp([comp_oh], comp_set_override=self.fraction_set)[0]

        # Validate (defensive; generation should already mask)
        if elem in self._selected:
            raise ValueError(f"Repeated element '{elem}' is not allowed.")

        comp_units = int(round(float(comp_str) * 20))
        if comp_units not in self._allowed_units:
            raise ValueError(f"Composition '{comp_str}' is not in allowed set {self.fraction_set}.")

        steps_left = self.max_steps - self.counter
        remaining = self.remaining_units
        if comp_units > remaining:
            raise ValueError("Action exceeds remaining fraction budget.")
        if (remaining - comp_units) not in self._possible_sums_by_k[steps_left - 1]:
            raise ValueError("Action makes it impossible to reach total fraction 1.0 with remaining steps.")

        old_state = self.state

        # Apply action
        if self.counter == 0:
            self.state = f"{elem}{comp_str}"
        else:
            self.state = f"{self.state}{elem}{comp_str}"

        self._selected.add(elem)
        self._used_units += comp_units
        self.counter += 1

        reward = 0.0
        if self.counter == self.max_steps:
            if self._used_units != self._total_units:
                # Should be impossible due to masking; keep a guard.
                reward = -1e6
            else:
                reward = float(self.reward_fn(self.terminal_formula))

        s_material = featurize_target(old_state)
        s_step = step_to_one_hot([self.counter])[0]
        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=elem_oh,
                action_comp_onehot=comp_oh,
                reward=reward,
            )
        )
