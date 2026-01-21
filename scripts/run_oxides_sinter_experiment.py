#!/usr/bin/env python3
"""Reproduce the paper's "minimize sintering temperature" experiment for 5-step oxide generation.

This script mirrors the key logic in the sinter notebooks:
- generate random episodes (4 cation steps + 1 oxygen step)
- compute Monte-Carlo discounted returns as Q-targets
- train DQN (Q network) on the dataset
- train DCN (constraint network) to predict electronegativity-validity
- generate new episodes using DQN-ranked actions, optionally filtered by DCN

It is designed to be run on a cluster (CPU or GPU) without Jupyter.

Example:
  python scripts/run_oxides_sinter_experiment.py \
    --rf-model rf_models/rf_sinter_predict_no_imputation_no_precursors.joblib \
    --out runs/oxides_sinter_repro \
    --num-random-eps 2000 \
    --dqn-epochs 50 \
    --dcn-epochs 30 \
    --num-gen-eps 500 \
    --use-dcn \
    --en-threshold 0.4 \
    --stochastic-top-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from pymatgen.core.composition import Composition
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from constraints.checkers import check_electronegativity
from model import DCN_pytorch, DQN_pytorch
from one_hot import (
    comp_set,
    comp_to_one_hot,
    element_set,
    element_to_one_hot,
    featurize_target,
    one_hot_to_comp,
    one_hot_to_element,
    step_to_one_hot,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rf_model(path: str) -> object:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RF model not found at '{path}'. Provide the correct file via --rf-model."
        )
    return joblib.load(path)


def predict_sinter_temperature(
    rf_model: object, feature_calculator, formula: str
) -> float:
    """Predict sintering temperature for a formula using a pre-trained sklearn regressor."""
    try:
        chemical = Composition(formula)
        features = feature_calculator.featurize(chemical)
        features = np.asarray(features, dtype=float).reshape(1, -1)
        pred = float(rf_model.predict(features)[0])
    except Exception:
        # Keep it simple and safe: if featurization fails, treat as bad.
        pred = 1000.0
    return pred


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float
    en_terminal: float | None


class OxideSinterEnv:
    """5-step environment: 4 cation-add steps + 1 oxygen-add step.

    State is a material formula string being constructed.
    Reward only at terminal: - predicted sintering temperature.
    Constraint label en_terminal at terminal: 1.0 if electronegativity constraint passes, else 0.0.

    Notes:
      - We treat intermediate rewards as 0 (as in the notebooks).
      - We store features of OLD state (before applying action), same as env_constrained.py.
    """

    def __init__(
        self,
        rf_model: object,
        feature_calculator,
        max_steps: int = 5,
        force_nonzero_cation: bool = True,
    ) -> None:
        self.rf_model = rf_model
        self.feature_calculator = feature_calculator
        self.max_steps = max_steps
        self.force_nonzero_cation = force_nonzero_cation

        self.state: str = ""
        self.counter: int = 0
        self.path: List[EpisodeStep] = []

    def initialize(self) -> None:
        self.state = ""
        self.counter = 0
        self.path = []

    def _terminal_reward(self) -> float:
        sinter_t = predict_sinter_temperature(
            self.rf_model, self.feature_calculator, self.state
        )
        return -sinter_t

    def _terminal_en(self) -> float:
        try:
            chemical = Composition(self.state)
            return 1.0 if check_electronegativity(chemical) else 0.0
        except Exception:
            return 0.0

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        elem_oh, comp_oh = action
        elem = one_hot_to_element([elem_oh])[0]
        comp = one_hot_to_comp([comp_oh])[0]

        old_state = self.state

        # Apply action: append elem+comp unless comp is "0".
        if comp != "0":
            if self.counter == 0:
                self.state = f"{elem}{comp}"
            else:
                self.state = f"{self.state}{elem}{comp}"

        # increment step counter (1..max_steps)
        self.counter += 1

        # Reward/constraint only at terminal
        reward = self._terminal_reward() if self.counter == self.max_steps else 0.0
        en = self._terminal_en() if self.counter == self.max_steps else None

        # Record SAR(C): use features of old_state
        s_material = featurize_target(old_state)
        s_step = step_to_one_hot([self.counter])[0]
        step_rec = EpisodeStep(
            state_material_features=s_material,
            state_step_onehot=s_step,
            action_elem_onehot=elem_oh,
            action_comp_onehot=comp_oh,
            reward=reward,
            en_terminal=en,
        )
        self.path.append(step_rec)


def _cation_elements() -> List[str]:
    return [e for e in element_set if e != "O"]


def random_action(
    *,
    oxide_step: bool,
    force_nonzero_cation: bool,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Sample a random action.

    - For oxide_step=True, force element O and comp in comp_set[1:] (non-zero).
    - Else choose a random cation and optionally force non-zero comp.
    """

    if oxide_step:
        elem = "O"
        comp_choices = comp_set[1:]
    else:
        elem = random.choice(_cation_elements())
        comp_choices = comp_set[1:] if force_nonzero_cation else comp_set

    comp = random.choice(comp_choices)
    elem_oh = tuple(element_to_one_hot([elem])[0].tolist())
    comp_oh = tuple(comp_to_one_hot([comp])[0].tolist())
    return elem_oh, comp_oh


def generate_random_episode(env: OxideSinterEnv) -> List[EpisodeStep]:
    env.initialize()
    for i in range(env.max_steps):
        oxide_step = i == env.max_steps - 1
        action = random_action(
            oxide_step=oxide_step,
            force_nonzero_cation=env.force_nonzero_cation,
        )
        env.step(action)
    return env.path


def extract_mc_q_targets(
    episode: List[EpisodeStep], gamma: float
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[float], List[float]]:
    """Compute Monte-Carlo discounted returns and propagate terminal en label backward.

    Returns:
      inputs: list of (s_material, s_step, a_elem, a_comp) arrays
      q_targets: list of floats (same length as episode)
      en_targets: list of floats (same length; terminal en copied to all steps)
    """

    q_targets: List[float] = []
    en_targets: List[float] = []
    inputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    G = 0.0
    en_terminal = float(episode[-1].en_terminal if episode[-1].en_terminal is not None else 0.0)

    for step in reversed(episode):
        G = float(step.reward) + gamma * G
        q_targets.append(G)
        en_targets.append(en_terminal)
        inputs.append(
            (
                np.asarray(step.state_material_features, dtype=float),
                np.asarray(step.state_step_onehot, dtype=float),
                np.asarray(step.action_elem_onehot, dtype=float),
                np.asarray(step.action_comp_onehot, dtype=float),
            )
        )

    # We built in reverse order; flip back to chronological order.
    inputs.reverse()
    q_targets.reverse()
    en_targets.reverse()
    return inputs, q_targets, en_targets


def build_dataset(
    episodes: List[List[EpisodeStep]],
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_material_list: List[np.ndarray] = []
    s_step_list: List[np.ndarray] = []
    a_elem_list: List[np.ndarray] = []
    a_comp_list: List[np.ndarray] = []
    q_list: List[float] = []
    en_list: List[float] = []

    for ep in episodes:
        inputs, q_targets, en_targets = extract_mc_q_targets(ep, gamma=gamma)
        for (s_mat, s_step, a_elem, a_comp), q, en in zip(inputs, q_targets, en_targets):
            s_material_list.append(s_mat)
            s_step_list.append(s_step)
            a_elem_list.append(a_elem)
            a_comp_list.append(a_comp)
            q_list.append(float(q))
            en_list.append(float(en))

    s_material = np.stack(s_material_list)
    s_step = np.stack(s_step_list)
    a_elem = np.stack(a_elem_list)
    a_comp = np.stack(a_comp_list)
    q_targets = np.asarray(q_list, dtype=float).reshape(-1, 1)
    en_targets = np.asarray(en_list, dtype=float).reshape(-1, 1)

    return s_material, s_step, a_elem, a_comp, q_targets, en_targets


def train_regressor_network(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    lr: float,
    epochs: int,
    loss_name: str,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_name == "smoothl1":
        loss_fn = torch.nn.SmoothL1Loss()
    elif loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    train_losses: List[float] = []
    valid_losses: List[float] = []

    for _ in tqdm(range(epochs), desc=f"train({model.__class__.__name__})"):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            s_mat, s_step, a_elem, a_comp, y = (x.to(device) for x in batch)
            pred = model(s_material=s_mat, s_step=s_step, a_elem=a_elem, a_comp=a_comp)
            loss = loss_fn(pred.float(), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * s_mat.shape[0]
            n += int(s_mat.shape[0])
        train_losses.append(total / max(1, n))

        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in valid_loader:
                s_mat, s_step, a_elem, a_comp, y = (x.to(device) for x in batch)
                pred = model(
                    s_material=s_mat, s_step=s_step, a_elem=a_elem, a_comp=a_comp
                )
                loss = loss_fn(pred.float(), y.float())
                total += float(loss.detach().cpu()) * s_mat.shape[0]
                n += int(s_mat.shape[0])
        valid_losses.append(total / max(1, n))

    return train_losses, valid_losses


def split_indices(n: int, valid_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(round(n * (1.0 - valid_frac)))
    return idx[:split], idx[split:]


def ranked_actions_for_state(
    *,
    dqn: DQN_pytorch,
    scaler: StandardScaler,
    state_material: str,
    step: int,
    device: torch.device,
    oxide_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a_elem_ranked, a_comp_ranked, q_ranked) for a state."""

    s_mat = np.asarray(featurize_target(state_material), dtype=float).reshape(1, -1)
    s_mat = scaler.transform(s_mat).reshape(-1)
    s_step = np.asarray(step_to_one_hot([step])[0], dtype=float)

    if oxide_only:
        elems = ["O"]
        comps = comp_set[1:]
    else:
        elems = _cation_elements()
        comps = comp_set[1:]  # keep non-zero for cation steps

    a_elem_list = []
    a_comp_list = []
    for e in elems:
        e_oh = element_to_one_hot([e])[0]
        for c in comps:
            c_oh = comp_to_one_hot([c])[0]
            a_elem_list.append(e_oh)
            a_comp_list.append(c_oh)

    a_elem = np.stack(a_elem_list).astype(np.float32)
    a_comp = np.stack(a_comp_list).astype(np.float32)

    n_actions = a_elem.shape[0]
    s_mat_batch = np.repeat(s_mat.reshape(1, -1), n_actions, axis=0).astype(np.float32)
    s_step_batch = np.repeat(s_step.reshape(1, -1), n_actions, axis=0).astype(np.float32)

    dqn.eval()
    with torch.no_grad():
        q = dqn(
            s_material=torch.from_numpy(s_mat_batch).to(device),
            s_step=torch.from_numpy(s_step_batch).to(device),
            a_elem=torch.from_numpy(a_elem).to(device),
            a_comp=torch.from_numpy(a_comp).to(device),
        ).detach().cpu().numpy().reshape(-1)

    order = np.argsort(-q)  # descending
    return a_elem[order], a_comp[order], q[order]


def choose_action_dqn(
    *,
    dqn: DQN_pytorch,
    scaler: StandardScaler,
    state_material: str,
    step: int,
    stochastic_top_frac: float,
    device: torch.device,
    oxide_only: bool,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    a_elem_ranked, a_comp_ranked, _ = ranked_actions_for_state(
        dqn=dqn,
        scaler=scaler,
        state_material=state_material,
        step=step,
        device=device,
        oxide_only=oxide_only,
    )

    if stochastic_top_frac and stochastic_top_frac > 0:
        k = max(1, int(round(len(a_elem_ranked) * stochastic_top_frac)))
        idx = np.random.randint(0, k)
    else:
        idx = 0

    return tuple(a_elem_ranked[idx].tolist()), tuple(a_comp_ranked[idx].tolist())


def choose_action_constrained(
    *,
    dqn: DQN_pytorch,
    dcn: DCN_pytorch,
    scaler: StandardScaler,
    state_material: str,
    step: int,
    en_threshold: float,
    stochastic_top_frac: float,
    device: torch.device,
    oxide_only: bool,
    max_tries: int = 200,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    a_elem_ranked, a_comp_ranked, _ = ranked_actions_for_state(
        dqn=dqn,
        scaler=scaler,
        state_material=state_material,
        step=step,
        device=device,
        oxide_only=oxide_only,
    )

    # Build state tensors once
    s_mat = np.asarray(featurize_target(state_material), dtype=float).reshape(1, -1)
    s_mat = scaler.transform(s_mat).astype(np.float32)
    s_step = np.asarray(step_to_one_hot([step])[0], dtype=float).reshape(1, -1).astype(np.float32)

    dcn.eval()
    tries = 0
    while True:
        tries += 1
        if stochastic_top_frac and stochastic_top_frac > 0:
            k = max(1, int(round(len(a_elem_ranked) * stochastic_top_frac)))
            idx = np.random.randint(0, k)
        else:
            idx = 0

        a_elem = a_elem_ranked[idx].reshape(1, -1).astype(np.float32)
        a_comp = a_comp_ranked[idx].reshape(1, -1).astype(np.float32)

        with torch.no_grad():
            en_pred = dcn(
                s_material=torch.from_numpy(s_mat).to(device),
                s_step=torch.from_numpy(s_step).to(device),
                a_elem=torch.from_numpy(a_elem).to(device),
                a_comp=torch.from_numpy(a_comp).to(device),
            ).detach().cpu().numpy().reshape(-1)[0]

        if float(en_pred) >= en_threshold:
            return tuple(a_elem_ranked[idx].tolist()), tuple(a_comp_ranked[idx].tolist())

        if tries >= max_tries:
            # Give up and return the best DQN action.
            return tuple(a_elem_ranked[0].tolist()), tuple(a_comp_ranked[0].tolist())


def generate_episode_with_policy(
    *,
    env: OxideSinterEnv,
    dqn: DQN_pytorch,
    dcn: DCN_pytorch | None,
    scaler: StandardScaler,
    epsilon: float,
    stochastic_top_frac: float,
    en_threshold: float,
    device: torch.device,
) -> Tuple[List[EpisodeStep], str]:
    env.initialize()

    for i in range(env.max_steps):
        oxide_only = i == env.max_steps - 1
        step_num = env.counter + 1

        if np.random.uniform(0, 1) < epsilon:
            action = random_action(
                oxide_step=oxide_only,
                force_nonzero_cation=env.force_nonzero_cation,
            )
        else:
            if dcn is None:
                action = choose_action_dqn(
                    dqn=dqn,
                    scaler=scaler,
                    state_material=env.state,
                    step=step_num,
                    stochastic_top_frac=stochastic_top_frac,
                    device=device,
                    oxide_only=oxide_only,
                )
            else:
                action = choose_action_constrained(
                    dqn=dqn,
                    dcn=dcn,
                    scaler=scaler,
                    state_material=env.state,
                    step=step_num,
                    en_threshold=en_threshold,
                    stochastic_top_frac=stochastic_top_frac,
                    device=device,
                    oxide_only=oxide_only,
                )

        env.step(action)

    return env.path, env.state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-model", required=True, help="Path to sinter RF .joblib")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--valid-frac", type=float, default=0.2)

    parser.add_argument("--num-random-eps", type=int, default=5000)

    parser.add_argument("--dqn-epochs", type=int, default=50)
    parser.add_argument("--dqn-lr", type=float, default=1e-3)
    parser.add_argument("--dqn-batch", type=int, default=1024)

    parser.add_argument("--use-dcn", action="store_true")
    parser.add_argument("--dcn-epochs", type=int, default=30)
    parser.add_argument("--dcn-lr", type=float, default=1e-3)
    parser.add_argument("--dcn-batch", type=int, default=1024)

    parser.add_argument("--num-gen-eps", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--stochastic-top-frac", type=float, default=0.2)
    parser.add_argument("--en-threshold", type=float, default=0.4)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    # Matminer featurizer used everywhere in this repo.
    from one_hot import feature_calculators as matminer_feature_calculators

    rf_model = load_rf_model(args.rf_model)

    env = OxideSinterEnv(
        rf_model=rf_model,
        feature_calculator=matminer_feature_calculators,
        max_steps=args.max_steps,
        force_nonzero_cation=True,
    )

    # 1) Generate random episodes (offline dataset)
    episodes: List[List[EpisodeStep]] = []
    for _ in tqdm(range(args.num_random_eps), desc="generate_random_episodes"):
        episodes.append(generate_random_episode(env))

    s_mat, s_step, a_elem, a_comp, q_targets, en_targets = build_dataset(
        episodes, gamma=args.gamma
    )

    # Save a lightweight snapshot of the raw random terminal rewards
    final_rewards = q_targets[args.max_steps - 1 :: args.max_steps].reshape(-1)

    # 2) Fit scaler on s_material and scale
    scaler = StandardScaler()
    scaler.fit(s_mat)
    s_mat_scaled = scaler.transform(s_mat).astype(np.float32)

    # Save scaler for reproducibility
    joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"), compress=True)

    # 3) Train/valid split
    train_idx, valid_idx = split_indices(len(s_mat_scaled), args.valid_frac, args.seed)

    def make_loader(
        idx: np.ndarray, batch_size: int, y: np.ndarray
    ) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(s_mat_scaled[idx]).float(),
            torch.from_numpy(s_step[idx]).float(),
            torch.from_numpy(a_elem[idx]).float(),
            torch.from_numpy(a_comp[idx]).float(),
            torch.from_numpy(y[idx]).float(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # 4) Train DQN
    dqn = DQN_pytorch().to(device)
    dqn_train = make_loader(train_idx, args.dqn_batch, q_targets)
    dqn_valid = make_loader(valid_idx, args.dqn_batch, q_targets)

    dqn_train_losses, dqn_valid_losses = train_regressor_network(
        model=dqn,
        train_loader=dqn_train,
        valid_loader=dqn_valid,
        lr=args.dqn_lr,
        epochs=args.dqn_epochs,
        loss_name="smoothl1",
        device=device,
    )

    torch.save(dqn, os.path.join(args.out, "dqn.pt"))

    dcn: DCN_pytorch | None = None
    dcn_train_losses: List[float] = []
    dcn_valid_losses: List[float] = []

    if args.use_dcn:
        dcn = DCN_pytorch().to(device)
        dcn_train = make_loader(train_idx, args.dcn_batch, en_targets)
        dcn_valid = make_loader(valid_idx, args.dcn_batch, en_targets)

        dcn_train_losses, dcn_valid_losses = train_regressor_network(
            model=dcn,
            train_loader=dcn_train,
            valid_loader=dcn_valid,
            lr=args.dcn_lr,
            epochs=args.dcn_epochs,
            loss_name="mse",
            device=device,
        )

        torch.save(dcn, os.path.join(args.out, "dcn.pt"))

    # 5) Generate new oxides with the learned policy
    gen_rows: List[Tuple[str, float, float]] = []

    for _ in tqdm(range(args.num_gen_eps), desc="generate_with_policy"):
        _, final_compound = generate_episode_with_policy(
            env=env,
            dqn=dqn,
            dcn=dcn,
            scaler=scaler,
            epsilon=args.epsilon,
            stochastic_top_frac=args.stochastic_top_frac,
            en_threshold=args.en_threshold,
            device=device,
        )
        sinter = predict_sinter_temperature(
            rf_model, matminer_feature_calculators, final_compound
        )
        # Validity label purely from checker (not dcn)
        try:
            en_ok = 1.0 if check_electronegativity(Composition(final_compound)) else 0.0
        except Exception:
            en_ok = 0.0
        gen_rows.append((final_compound, float(sinter), float(en_ok)))

    # 6) Save outputs
    with open(os.path.join(args.out, "generated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "pred_sinter_T", "en_ok"])
        for row in sorted(gen_rows, key=lambda r: r[1]):
            w.writerow(row)

    summary = {
        "seed": args.seed,
        "device": str(device),
        "rf_model": args.rf_model,
        "num_random_eps": args.num_random_eps,
        "num_gen_eps": args.num_gen_eps,
        "gamma": args.gamma,
        "max_steps": args.max_steps,
        "dqn": {
            "epochs": args.dqn_epochs,
            "lr": args.dqn_lr,
            "batch": args.dqn_batch,
            "train_loss_last": dqn_train_losses[-1] if dqn_train_losses else None,
            "valid_loss_last": dqn_valid_losses[-1] if dqn_valid_losses else None,
        },
        "dcn": {
            "enabled": bool(args.use_dcn),
            "epochs": args.dcn_epochs if args.use_dcn else 0,
            "lr": args.dcn_lr if args.use_dcn else 0.0,
            "batch": args.dcn_batch if args.use_dcn else 0,
            "train_loss_last": dcn_train_losses[-1] if dcn_train_losses else None,
            "valid_loss_last": dcn_valid_losses[-1] if dcn_valid_losses else None,
        },
        "random_final_reward": {
            "mean": float(np.mean(final_rewards)),
            "std": float(np.std(final_rewards)),
            "min": float(np.min(final_rewards)),
            "max": float(np.max(final_rewards)),
        },
        "generated_pred_sinter_T": {
            "mean": float(np.mean([r[1] for r in gen_rows])),
            "std": float(np.std([r[1] for r in gen_rows])),
            "min": float(np.min([r[1] for r in gen_rows])),
            "max": float(np.max([r[1] for r in gen_rows])),
        },
        "generated_en_ok_fraction": float(np.mean([r[2] for r in gen_rows])),
        "policy": {
            "epsilon": args.epsilon,
            "stochastic_top_frac": args.stochastic_top_frac,
            "en_threshold": args.en_threshold,
        },
    }

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Wrote: {os.path.join(args.out, 'generated.csv')}")
    print(f"Wrote: {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
