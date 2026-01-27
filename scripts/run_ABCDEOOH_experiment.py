#!/usr/bin/env python3
"""Train a DQN to generate constrained ABCDEOOH-like compositions.

Implements the user's constraint family:
- 5 distinct cations chosen from a provided list
- each cation fraction in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}
- cation fractions sum to 1.0
- terminal formula appends a fixed anion string (default: O2H1, i.e. "OOH")

Like the original repo notebooks, this is an OFFLINE workflow:
1) generate random episodes under the constraint mask
2) compute Monte-Carlo returns as Q targets
3) fit DQN by supervised regression on (s, a) -> Q
4) generate new candidates by greedily selecting actions that maximize predicted Q

Example:
  python scripts/run_ABCDEOOH_experiment.py \
    --out runs/abcde_ooh \
    --num-random-eps 5000 \
    --dqn-epochs 50 \
    --num-gen-eps 500 \
    --reward-mode none

If you want to reuse the RF sintering predictor (mostly for plumbing / demo), use:
  --reward-mode sinter-rf --rf-model rf_models/optimal_sinter_RF.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
from pymatgen.core.composition import Composition
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from env_oxyhydroxide import ABCDEOOHEnv, DEFAULT_CATION_SET, DEFAULT_FRACTIONS
from constraints.primary_phase import check_primary_phase
from model import DQN_pytorch
from one_hot import feature_calculators, featurize_target, step_to_one_hot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def predict_sinter_temperature(rf_model: object, formula: str) -> float:
    try:
        chemical = Composition(formula)
        features = feature_calculators.featurize(chemical)
        features = np.asarray(features, dtype=float).reshape(1, -1)
        return float(rf_model.predict(features)[0])
    except Exception:
        return 1000.0


def extract_mc_q_targets(episode, gamma: float):
    inputs = []
    q_targets: List[float] = []

    G = 0.0
    for step in reversed(episode):
        G = float(step.reward) + gamma * G
        q_targets.append(G)
        inputs.append(
            (
                np.asarray(step.state_material_features, dtype=float),
                np.asarray(step.state_step_onehot, dtype=float),
                np.asarray(step.action_elem_onehot, dtype=float),
                np.asarray(step.action_comp_onehot, dtype=float),
            )
        )

    inputs.reverse()
    q_targets.reverse()
    return inputs, q_targets


def train_dqn(
    *,
    dqn: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    dqn.train()
    opt = torch.optim.Adam(dqn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in tqdm(range(epochs), desc="DQN epochs"):
        for s_mat, s_step, a_elem, a_comp, y in loader:
            s_mat = s_mat.to(device)
            s_step = s_step.to(device)
            a_elem = a_elem.to(device)
            a_comp = a_comp.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = dqn(s_mat, s_step, a_elem, a_comp)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()


def choose_action(
    *,
    dqn: torch.nn.Module,
    device: torch.device,
    s_material: np.ndarray,
    s_step: np.ndarray,
    allowed_actions: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    stochastic_top_frac: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not allowed_actions:
        raise RuntimeError("No allowed actions.")

    a_elem = np.asarray([a[0] for a in allowed_actions], dtype=float)
    a_comp = np.asarray([a[1] for a in allowed_actions], dtype=float)

    s_mat_batch = np.repeat(s_material.reshape(1, -1), repeats=len(allowed_actions), axis=0)
    s_step_batch = np.repeat(s_step.reshape(1, -1), repeats=len(allowed_actions), axis=0)

    with torch.no_grad():
        q = dqn(
            torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
            torch.tensor(s_step_batch, dtype=torch.float32, device=device),
            torch.tensor(a_elem, dtype=torch.float32, device=device),
            torch.tensor(a_comp, dtype=torch.float32, device=device),
        ).reshape(-1)

    q_np = q.detach().cpu().numpy()
    order = np.argsort(-q_np)

    if stochastic_top_frac <= 0.0:
        return allowed_actions[int(order[0])]

    k = max(1, int(round(stochastic_top_frac * len(allowed_actions))))
    topk = order[:k]
    idx = int(np.random.choice(topk))
    return allowed_actions[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--num-random-eps",
        type=int,
        default=5000,
        help=(
            "Number of random episodes used to build the offline training dataset. "
            "If --primary-phase-filter is 'buffer' or 'both', this is the number of ACCEPTED "
            "(constraint-valid) episodes to collect."
        ),
    )
    parser.add_argument(
        "--max-random-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of random episode attempts to reach --num-random-eps accepted episodes "
            "when buffer filtering is enabled. If unset, defaults to num_random_eps*200."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--dqn-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--num-gen-eps",
        type=int,
        default=500,
        help=(
            "Number of generated episodes (candidates). If --primary-phase-filter is 'generated' or 'both', "
            "this is the number of ACCEPTED (constraint-valid) candidates to write to generated.csv."
        ),
    )
    parser.add_argument(
        "--max-gen-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of generation attempts to reach --num-gen-eps accepted candidates when generated "
            "filtering is enabled. If unset, defaults to num_gen_eps*200."
        ),
    )
    parser.add_argument("--stochastic-top-frac", type=float, default=0.0)

    parser.add_argument("--anion-formula", type=str, default="O2H1")

    parser.add_argument(
        "--use-saved-random-dataset",
        action="store_true",
        help=(
            "If set, load offline random dataset from 'random_dataset.npz' in --out "
            "and skip regenerating random episodes (no DeepMD/RF calls for data gen)."
        ),
    )

    parser.add_argument(
        "--primary-phase-filter",
        choices=["none", "buffer", "generated", "both"],
        default="none",
        help=(
            "Apply Ni/NiFe/NiFeCo/CoFe primary-phase constraints either when building "
            "the offline buffer ('buffer'), when writing generated.csv ('generated'), "
            "both, or not at all ('none')."
        ),
    )

    parser.add_argument(
        "--reward-mode",
        choices=["none", "sinter-rf", "dp"],
        default="none",
        help="Reward at terminal. 'sinter-rf' uses - predicted sintering temperature.",
    )
    parser.add_argument("--rf-model", type=str, default="")

    # DeepMD / overpotential reward config
    parser.add_argument("--dp-poscar", type=str, default="POSCAR")
    parser.add_argument(
        "--dp-model",
        action="append",
        default=[],
        help="Path to a DeepMD .pt checkpoint. Repeat 5 times for ensemble.",
    )
    parser.add_argument("--dp-n-random-configs", type=int, default=10)
    parser.add_argument("--dp-ads-height", type=float, default=1.9)
    parser.add_argument("--dp-ads-dz", type=float, default=1.0)
    parser.add_argument(
        "--dp-objective",
        choices=["mean_minus_kstd", "mean_plus_kstd"],
        default="mean_minus_kstd",
        help="Objective minimized by DP reward. mean_minus_kstd encourages exploration (lower mean or higher std).",
    )
    parser.add_argument(
        "--dp-uncertainty",
        choices=["models", "configs", "total"],
        default="models",
        help="How to compute std: across 5 model means ('models'), across random configs ('configs'), or pooled ('total').",
    )
    parser.add_argument(
        "--dp-k",
        type=float,
        default=1.0,
        help="Weight k for std term in objective (objective = mean ± k*std).",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rf_model = None
    if args.reward_mode == "sinter-rf":
        if not args.rf_model:
            raise SystemExit("--reward-mode sinter-rf requires --rf-model")
        rf_model = joblib.load(args.rf_model)

    dp_predictor = None
    # Cache DP evaluations: key -> {"mean": float, "std": float, "objective": float}
    dp_cache = {}
    if args.reward_mode == "dp":
        from dp_predictor import DPConfig, DeepMDOverpotentialPredictor, objective_from_mean_std

        model_files = tuple(args.dp_model) if args.dp_model else ()
        if not model_files:
            raise SystemExit("--reward-mode dp requires at least one --dp-model (repeat for ensemble)")

        cfg = DPConfig(
            base_poscar=args.dp_poscar,
            model_files=model_files,
            n_random_configs=args.dp_n_random_configs,
            ads_height=args.dp_ads_height,
            ads_dz=args.dp_ads_dz,
            seed=args.seed,
        )
        dp_predictor = DeepMDOverpotentialPredictor(cfg)

    def reward_fn(formula: str) -> float:
        if args.reward_mode == "none":
            return 0.0
        if args.reward_mode == "sinter-rf":
            assert rf_model is not None
            return -predict_sinter_temperature(rf_model, formula)
        if args.reward_mode == "dp":
            # NOTE: formula includes anion suffix; DP predictor needs ONLY metal-site fractions.
            # We'll extract fractions from env.state when reward is computed (see env binding below).
            raise RuntimeError("DP reward is bound via env-aware closure")
        raise RuntimeError("Unknown reward mode")

    # Environment uses overrides implicitly through its own sets.
    # Bind reward in a way that can access env.cation fractions for DP mode.
    env = ABCDEOOHEnv(
        cation_set=DEFAULT_CATION_SET,
        fraction_set=DEFAULT_FRACTIONS,
        anion_formula=args.anion_formula,
        reward_fn=reward_fn,
    )

    # Used by dp_reward_fn to avoid wasting DeepMD on episodes we will discard.
    # Values: "random" | "generate".
    current_phase = "random"

    if args.reward_mode == "dp":
        assert dp_predictor is not None
        from dp_predictor import objective_from_mean_std

        def dp_reward_fn(_terminal_formula: str) -> float:
            comp = env.terminal_cation_fractions()

            # If this episode will be filtered out of the replay buffer anyway,
            # skip expensive DeepMD evaluation entirely.
            skip_for_constraints = False
            if args.primary_phase_filter in {"buffer", "both"}:
                skip_for_constraints = True
            elif current_phase == "generate" and args.primary_phase_filter in {"generated", "both"}:
                skip_for_constraints = True

            if skip_for_constraints:
                ok, label = check_primary_phase(comp)
                if not ok:
                    print(
                        f"[DP-Reward] skipped (primary-phase={label or 'none'}) comp={comp}",
                        flush=True,
                    )
                    return 0.0

            key = tuple(sorted((k, float(v)) for k, v in comp.items()))
            if key in dp_cache:
                entry = dp_cache[key]
                mean = entry["mean"]
                std = entry["std"]
            else:
                mean, std, _ = dp_predictor.predict_overpotential(
                    comp,
                    uncertainty=args.dp_uncertainty,
                    return_per_model=False,
                )
                obj = objective_from_mean_std(mean, std, mode=args.dp_objective, k=args.dp_k)
                dp_cache[key] = {"mean": mean, "std": std, "objective": obj}

            # If cache was hit, objective might not yet be stored (older runs); ensure it is.
            if "objective" not in dp_cache[key]:
                obj = objective_from_mean_std(mean, std, mode=args.dp_objective, k=args.dp_k)
                dp_cache[key]["objective"] = obj
            else:
                obj = dp_cache[key]["objective"]

            # Log raw DP outputs for debugging.
            print(
                f"[DP-Reward] comp={comp} mean={mean:.6f} std={std:.6f} "
                f"objective={obj:.6f} reward={-obj:.6f}",
                flush=True,
            )

            # RL maximizes reward; we want to minimize objective.
            return -float(obj)

        env.reward_fn = dp_reward_fn

    # 1) Build or load offline random dataset for DQN training.
    if args.use_saved_random_dataset:
        ds_path = os.path.join(args.out, "random_dataset.npz")
        if not os.path.exists(ds_path):
            raise SystemExit(
                f"--use-saved-random-dataset was set but '{ds_path}' does not exist. "
                "Run once without this flag to generate it."
            )
        data = np.load(ds_path)
        s_mat = data["s_mat"]
        s_step = data["s_step"]
        a_elem = data["a_elem"]
        a_comp = data["a_comp"]
        y = data["y"]
    else:
        all_inputs = []
        all_q = []

        need_buffer_filter = args.primary_phase_filter in {"buffer", "both"}
        target_eps = int(args.num_random_eps)
        max_attempts = (
            int(args.max_random_attempts)
            if args.max_random_attempts is not None
            else (target_eps * 200 if need_buffer_filter else target_eps)
        )

        accepted_eps = 0
        attempts = 0
        pbar = tqdm(total=target_eps, desc="Random episodes (accepted)")
        while accepted_eps < target_eps and attempts < max_attempts:
            attempts += 1
            env.initialize()
            for _step in range(env.max_steps):
                env.step(env.sample_random_action())

            # Optional primary-phase filter for buffer construction.
            if need_buffer_filter:
                comp = env.terminal_cation_fractions()
                ok, _label = check_primary_phase(comp)
                if not ok:
                    continue

            episode = env.path
            inputs, q_targets = extract_mc_q_targets(episode, args.gamma)
            all_inputs.extend(inputs)
            all_q.extend(q_targets)
            accepted_eps += 1
            pbar.update(1)

        pbar.close()

        if accepted_eps < target_eps:
            raise SystemExit(
                "Could not collect enough constraint-valid random episodes. "
                f"Accepted {accepted_eps}/{target_eps} after {attempts} attempts. "
                "Increase --max-random-attempts and/or relax constraints."
            )

        if need_buffer_filter:
            print(
                f"[Buffer] Accepted {accepted_eps}/{attempts} episodes "
                f"(acceptance={accepted_eps / max(1, attempts):.4f}); "
                f"training rows={len(all_inputs)} (~{len(all_inputs) / accepted_eps:.1f} per ep)",
                flush=True,
            )

        # Build arrays from collected inputs/targets.
        s_mat = np.stack([x[0] for x in all_inputs], axis=0)
        s_step = np.stack([x[1] for x in all_inputs], axis=0)
        a_elem = np.stack([x[2] for x in all_inputs], axis=0)
        a_comp = np.stack([x[3] for x in all_inputs], axis=0)
        y = np.asarray(all_q, dtype=float).reshape(-1, 1)

        # Save offline dataset so future runs can skip random episode generation / DP calls.
        np.savez(
            os.path.join(args.out, "random_dataset.npz"),
            s_mat=s_mat,
            s_step=s_step,
            a_elem=a_elem,
            a_comp=a_comp,
            y=y,
        )

    scaler = StandardScaler()
    s_mat_scaled = scaler.fit_transform(s_mat)

    with open(os.path.join(args.out, "scaler.json"), "w") as f:
        json.dump(
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
            },
            f,
        )

    ds = TensorDataset(
        torch.tensor(s_mat_scaled, dtype=torch.float32),
        torch.tensor(s_step, dtype=torch.float32),
        torch.tensor(a_elem, dtype=torch.float32),
        torch.tensor(a_comp, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 3) Train DQN
    dqn = DQN_pytorch(num_elem=len(DEFAULT_CATION_SET), num_comp=len(DEFAULT_FRACTIONS)).to(device)
    train_dqn(dqn=dqn, loader=loader, device=device, epochs=args.dqn_epochs, lr=args.lr)

    torch.save(dqn.state_dict(), os.path.join(args.out, "dqn.pt"))

    # 4) Generate new candidates using DQN policy
    dqn.eval()
    rows = []

    need_generated_filter = args.primary_phase_filter in {"generated", "both"}
    target_gen = int(args.num_gen_eps)
    max_gen_attempts = (
        int(args.max_gen_attempts)
        if args.max_gen_attempts is not None
        else (target_gen * 200 if need_generated_filter else target_gen)
    )

    current_phase = "generate"
    accepted_gen = 0
    gen_attempts = 0
    pbar = tqdm(total=target_gen, desc="Generate (accepted)")
    while accepted_gen < target_gen and gen_attempts < max_gen_attempts:
        gen_attempts += 1
        env.initialize()
        for _step in range(env.max_steps):
            # State for decision is features of *current* state string.
            s_material = np.asarray(featurize_target(env.state), dtype=float)
            s_material = scaler.transform(s_material.reshape(1, -1)).reshape(-1)
            # step_to_one_hot is fixed (5 steps)
            s_step_vec = step_to_one_hot([env.counter + 1])[0]

            allowed = env.allowed_actions()
            action = choose_action(
                dqn=dqn,
                device=device,
                s_material=s_material,
                s_step=np.asarray(s_step_vec, dtype=float),
                allowed_actions=allowed,
                stochastic_top_frac=args.stochastic_top_frac,
            )
            env.step(action)

        comp = env.terminal_cation_fractions()
        ok, label = check_primary_phase(comp)
        if need_generated_filter and not ok:
            continue

        formula = env.terminal_formula
        reward = float(env.path[-1].reward)

        dp_mean = ""
        dp_std = ""
        dp_mean_minus_std = ""
        primary_ok = bool(ok)
        primary_label = label or ""

        if args.reward_mode == "dp":
            assert dp_predictor is not None
            key = tuple(sorted((k, float(v)) for k, v in comp.items()))
            entry = dp_cache.get(key)
            if entry is None:
                # If this composition was not seen during random episodes, evaluate now.
                mean, std, _ = dp_predictor.predict_overpotential(
                    comp,
                    uncertainty=args.dp_uncertainty,
                    return_per_model=False,
                )
                obj = objective_from_mean_std(mean, std, mode=args.dp_objective, k=args.dp_k)
                entry = {"mean": mean, "std": std, "objective": obj}
                dp_cache[key] = entry
            dp_mean = float(entry["mean"])
            dp_std = float(entry["std"])
            dp_mean_minus_std = float(dp_mean) - float(dp_std)

        rows.append(
            {
                "formula": formula,
                "reward": reward,
                "dp_mean": dp_mean,
                "dp_std": dp_std,
                "dp_mean_minus_std": dp_mean_minus_std,
                "primary_ok": primary_ok,
                "primary_label": primary_label,
            }
        )
        accepted_gen += 1
        pbar.update(1)

    pbar.close()
    if accepted_gen < target_gen:
        raise SystemExit(
            "Could not generate enough constraint-valid candidates. "
            f"Accepted {accepted_gen}/{target_gen} after {gen_attempts} attempts. "
            "Increase --max-gen-attempts and/or relax constraints."
        )

    if need_generated_filter:
        print(
            f"[Generate] Accepted {accepted_gen}/{gen_attempts} candidates "
            f"(acceptance={accepted_gen / max(1, gen_attempts):.4f})",
            flush=True,
        )

    # For DP reward mode, sort candidates by increasing dp_mean - dp_std (best first).
    if args.reward_mode == "dp":
        rows.sort(key=lambda r: r["dp_mean_minus_std"])

    with open(os.path.join(args.out, "generated.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "formula",
                "reward",
                "dp_mean",
                "dp_std",
                "dp_mean_minus_std",
                "primary_ok",
                "primary_label",
            ],
        )
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
