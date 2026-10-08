"""Lesson 17 definition of done (optional lesson).

Run:  pytest tests/test_17.py -q
Expected: 7 passed.
"""

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import lesson_module

exp = lesson_module("experiments", "seed_everything", "RunConfig", "run_experiment", "compare", "repeat_with_seeds", lesson="Lesson 17")
MLP = lesson_module("model", "MLP", lesson="Lesson 4").MLP

fast = pytest.mark.fast
QUICK = dict(epochs=40, n_orders=800)


@fast
def test_seed_everything_makes_init_and_shuffle_reproducible():
    g1 = exp.seed_everything(123); m1 = MLP(4, (8,)); order1 = [b[0].tolist() for b in DataLoader(TensorDataset(torch.arange(16)), batch_size=4, shuffle=True, generator=g1)]
    g2 = exp.seed_everything(123); m2 = MLP(4, (8,)); order2 = [b[0].tolist() for b in DataLoader(TensorDataset(torch.arange(16)), batch_size=4, shuffle=True, generator=g2)]
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters())), "same seed, same initial weights"
    assert order1 == order2, "a seeded generator gives the same shuffle order"
    exp.seed_everything(124); m3 = MLP(4, (8,))
    assert not torch.equal(next(m1.parameters()), next(m3.parameters()))


@fast
def test_deterministic_flag_is_set_and_restored():
    exp.seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.benchmark
    exp.seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.benchmark
    exp.seed_everything(0, deterministic=True)


@fast
def test_config_round_trip_and_fingerprint(tmp_path):
    cfg = exp.RunConfig(hidden=(16, 16), lr=0.005, seed=3)
    cfg.to_json(tmp_path / "run.json")
    assert exp.RunConfig.from_json(tmp_path / "run.json") == cfg, "JSON round trip must give back an equal config, tuples included"
    fp = cfg.fingerprint()
    assert len(fp) == 8 and fp == exp.RunConfig(hidden=(16, 16), lr=0.005, seed=3).fingerprint(), "the same config always has the same fingerprint"
    assert fp != replace(cfg, lr=0.01).fingerprint() and fp != replace(cfg, seed=4).fingerprint(), "any field change changes the fingerprint"


@fast
def test_run_experiment_is_reproducible_and_logged(tmp_path):
    cfg = exp.RunConfig(hidden=(16,), **QUICK)
    log = tmp_path / "experiments.jsonl"
    a = exp.run_experiment(cfg, log)
    b = exp.run_experiment(cfg, log)
    assert a["best_val"] == b["best_val"] and a["best_epoch"] == b["best_epoch"], "same config, same result, to the last digit"
    assert a["fingerprint"] == cfg.fingerprint() and {"seconds", "timestamp", "commit", "lr", "hidden"} <= set(a)
    assert len(log.read_text().strip().splitlines()) == 2, "one JSON line per run"
    assert exp.load_log(log).shape[0] == 2


@fast
def test_a_different_seed_is_a_different_run():
    cfg = exp.RunConfig(hidden=(16,), **QUICK)
    a = exp.run_experiment(cfg)
    b = exp.run_experiment(replace(cfg, seed=1))
    assert a["fingerprint"] != b["fingerprint"] and a["best_val"] != b["best_val"]


@fast
def test_compare_groups_by_config_not_seed(tmp_path):
    log = tmp_path / "experiments.jsonl"
    for seed in (0, 1):
        exp.run_experiment(exp.RunConfig(hidden=(16,), seed=seed, **QUICK), log)
        exp.run_experiment(exp.RunConfig(model="linear", seed=seed, **QUICK), log)
    table = exp.compare(exp.load_log(log))
    assert len(table) == 2 and set(table["runs"]) == {2}, "two configs, two seeds each"
    assert {"mean_val", "std_val", "model"} <= set(table.columns)
    assert table.iloc[0]["mean_val"] <= table.iloc[1]["mean_val"], "sorted best first"
    assert table.loc[table["model"] == "mlp", "mean_val"].item() < table.loc[table["model"] == "linear", "mean_val"].item()


@fast
def test_repeat_with_seeds_reports_spread():
    out = exp.repeat_with_seeds(exp.RunConfig(hidden=(16,), **QUICK), seeds=[0, 1, 2])
    assert len(out["values"]) == 3 and out["std"] > 0
    assert abs(out["mean"] - sum(out["values"]) / 3) < 1e-4
    assert abs(out["std"] - torch.tensor(out["values"], dtype=torch.float64).std().item()) < 1e-4, "sample standard deviation (ddof=1)"
