"""Hugging Face Trainer integration, end to end with a tiny randomly initialised BERT (no downloads)."""

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from trainscope.integrations.hf_trainer import TrainScopeCallback  # noqa: E402
from trainscope_server.db import open_store  # noqa: E402
from trainscope_server.detectors import RunData, run_detectors  # noqa: E402


class TinyDS(torch.utils.data.Dataset):
    def __init__(self, n=64, seq=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(5, 100, (n, seq), generator=g)
        self.y = (self.x[:, 0] > 50).long()  # learnable rule

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input_ids": self.x[i], "attention_mask": torch.ones_like(self.x[i]), "labels": self.y[i]}


@pytest.mark.slow
def test_trainer_callback_records_run(tmp_path):
    cfg = transformers.BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=64, max_position_embeddings=32, num_labels=2)
    model = transformers.BertForSequenceClassification(cfg)
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "out"), max_steps=30, per_device_train_batch_size=8, learning_rate=1e-3,
        warmup_steps=5, logging_steps=5, eval_strategy="steps", eval_steps=10, save_strategy="no",
        report_to=[], use_cpu=True, disable_tqdm=True, gradient_accumulation_steps=2, max_grad_norm=1.0)
    db = str(tmp_path / "hf.duckdb")
    cb = TrainScopeCallback(project="hf-test", grad_every=5, act_every=10, db=db, spool_dir=str(tmp_path / "sp"),
                            system_interval=None, poll_commands=False)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=TinyDS(), eval_dataset=TinyDS(32, seed=1),
                                   callbacks=[cb])
    trainer.train()

    store = open_store(db)
    run = store.get_run(cb.run.id)
    assert run["status"] == "finished"
    assert run["config"]["grad_accum"] == 2 and run["config"]["grad_clip"] == 1.0
    names = set(store.metric_names(cb.run.id))
    # Trainer scalars, per-step bookkeeping and sampled layer stats all arrive
    assert {"loss", "lr", "eval/loss", "step_time", "grad/norm", "grad/global_norm", "weight/update_ratio",
            "grad/carryover_ratio", "act/mean", "model/train_mode"} <= names
    steps = store.metrics_df(cb.run.id, names=["step_time"])["step"]
    assert len(steps) >= 25, "step bookkeeping must run every optimizer step, not only on logging steps"
    # gradient accumulation of 2 must not look like a missing zero_grad
    findings = run_detectors(RunData.from_store(store, cb.run.id))
    assert not any(f.detector == "grad_carryover" for f in findings), [f.summary for f in findings]
