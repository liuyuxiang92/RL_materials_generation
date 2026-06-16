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
