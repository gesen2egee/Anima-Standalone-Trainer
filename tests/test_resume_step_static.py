from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_train_network() -> str:
    return (ROOT / "train_network.py").read_text(encoding="utf-8")


def test_resume_step_from_state_updates_global_step_and_progress_bar():
    source = read_train_network()

    assert "resume_step = steps_from_state or 0" in source
    assert "global_step = resume_step" in source
    assert "current_step.value = global_step" in source
    assert "initial=global_step" in source
    assert "total=args.max_train_steps" in source


def test_sample_at_first_uses_resume_global_step():
    source = read_train_network()

    assert "self.sample_images(accelerator, args, 0, global_step" in source
    assert "accelerator.log({}, step=global_step)" in source
