# Trace: RL_materials_generation

<!-- concepts: deep-RL materials design, DQN reproduction, surrogate reward models -->

## Reproducing sinter / calcine / sinter+calcine DQN runs

**Task:** Reproduce the sintering, calcination, and combined sinter+calcine DQN
experiments from Karpovich et al. (npj Comput Mater 2024). User runs on a separate GPU machine.

**Key findings**
- The missing `optimal_calcine_RF.joblib` (required by `env_constrained.py:39`, loaded
  unconditionally at import) was found in the parent repo as
  `../../deep-rl-inorganic/PGN/optimal_calcine_RF.joblib.zip`. Unzipped into `rf_models/`.
  Without it, *every* run crashes at import — including sinter-only.
- `deep-rl-inorganic` is the parent paper repo; its `DQN/` is a git submodule pointing to
  this exact repo (eltonpan/RL_materials_generation). PGN shares the same rf_models/roost_models folders.
- **Gotcha:** `generate_random_ep()` (random-init seed phase) uses the *module-level* `env`
  in env_constrained, which is built from `configs.py`, NOT the driver's tasks. Must sync
  `env_constrained.env.tasks` to the CLI tasks or the seed data optimizes the wrong objective.
- Output dirs are created with `os.mkdir` (fails if top-level dir absent) → switching to
  `os.makedirs(exist_ok=True)`.
- Sinter/calcine rewards are raw temperatures (~1000s K); combined reward = -(sinter+calcine),
  equal-weighted, comparable scales. ROOST property tasks remain blocked (no checkpoints).

**Changes made to `oxides_driver.py`**
- Replaced hardcoded inline config + RUN_ID with argparse (`--tasks`, `--run-id`, `--gpu`).
- `--gpu` no longer hardcodes index "2"; respects shell `CUDA_VISIBLE_DEVICES` if omitted.
- Sync module-level random-init env objective to `--tasks`.

### EARS — Session Start (2026-06-22 10:28)
<!-- concepts: RL materials generation, model benchmarking, generated-candidate comparison -->
- Task: Help compare RL_material_generation (DQN) sinter/calcine outputs against rl_matdesign's generated.csv candidates.
- Why: User wants to benchmark two RL materials-generation approaches on the same sinter/calcine property tasks.

### EARS — Session Start (2026-06-29 18:09)
<!-- concepts: policy-gradient RL, stack-RNN generator, PGN reproduction -->
- Task: Reproduce sinter / calcine / sinter+calcine with a policy-based method, mirroring the DQN CLI repro.
- Why: Benchmark a policy-gradient agent against the DQN agent on the same surrogate-reward materials-design tasks.

## Reproducing sinter / calcine / sinter+calcine PGN (policy-gradient) runs
<!-- concepts: policy-gradient RL, stack-RNN generator, PGN reproduction -->

**Decision:** User chose the *paper's actual PGN* (stack-augmented RNN, ReLeaSE-style
REINFORCE) over a from-scratch REINFORCE on the DQN env. Work lives in the sibling
repo `../../deep-rl-inorganic/PGN/` (NOT this repo — separate git repo; DQN/ there is a
submodule pointing here).

**Setup done (mac, no GPU — code prep only; training runs on user's GPU machine):**
- Symlinked `roost_models`, `constraints`, `metrics.py` from this repo into PGN/.
  `metrics.py` needs `ElM2D` (EMD); scripts also import `rdkit`. Use the `dqn` conda env.
- Unzipped `optimal_{sinter,calcine}_RF.joblib` in PGN/ (loaded from cwd root, not rf_models/).
- **`pretrain_unbiased.py` (new):** headless port of the unbiased-RNN notebook. Produces
  `checkpoints/unbiased/unbiased_10000` — the hard prerequisite for every RL run.
  Dropped the notebook's per-1000-epoch Roost/EMD eval so pretraining is self-contained.
- **`train_RL_models_hyperparameter.py` (sinter/calcine):** added argparse
  (`--tasks`/`--runs`/`--weight`/`--gpu`); already cwd-relative.
- **`train_RL_models_hyperparameter_multiobjective.py` (combined):** was NOT repro-ready —
  hardcoded `/home/jupyter/...` paths, broken `from pymatgen import Composition`, and
  **only a sinter+bulk_mod reward branch (no sinter+calcine!)**. Added the
  `sinter_temp`+`calcine_temp` reward branch, rewrote all paths cwd-relative, fixed the
  import, added CLI + `__main__`.
- Both RL scripts: guarded `predict_formation_energy` (form_e is reported-only) so a
  missing Roost checkpoint doesn't discard valid sinter/calcine compounds.
- **`run_pgn_reproductions.sh` (new):** pretrain-if-needed -> sinter -> calcine -> combined,
  mirroring DQN's `run_reproductions.sh`. Plus `README_reproduction.md`.

**Gotcha for comparison:** PGN normalizes each property to [0,1] (min/max dict in script)
and mixes with a charge/EN-balance reward (weight 0.5); DQN used raw negative temps. So
compare the two methods on *generated compounds' predicted temperatures*, not raw rewards.
