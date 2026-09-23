import sys
from pathlib import Path

import torch

MAM_ROOT = Path(__file__).resolve().parents[2] / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_model import ConditionalREGNNConfig
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


def tiny_student(num_experts=2):
    backbone = ConditionalREGNNConfig(
        audio_dim=4, speaker_emotion_dim=3, speaker_3dmm_dim=5,
        output_dim=25, hidden_dim=16, temporal_layers=1,
        temporal_heads=4, graph_dim=8, graph_layers=1,
        graph_heads=2, dropout=0.0, max_seq_len=6,
    )
    return ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=backbone, num_experts=num_experts, expert_embedding_dim=4,
    ))


def inputs(batch=2):
    torch.manual_seed(3)
    return (
        torch.randn(batch, 6, 4),
        torch.randn(batch, 6, 3),
        torch.randn(batch, 6, 5),
        torch.tensor([6, 4][:batch]),
    )


def test_generate_encodes_once_and_returns_all_experts(monkeypatch):
    model = tiny_student(2).eval()
    calls = 0
    original = model.backbone.encode_conditions

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backbone, "encode_conditions", counted)
    with torch.no_grad():
        output = model.generate(*inputs())
    assert calls == 1
    assert output["prediction"].shape == (2, 2, 6, 25)
    assert output["valid_mask"].shape == (2, 2, 6)
    assert torch.equal(output["valid_mask"][0, 0], output["valid_mask"][0, 1])


def test_expert_embedding_receives_gradient_and_targets_are_not_averaged():
    model = tiny_student(2).train()
    source = inputs(batch=1)
    teacher = torch.zeros(1, 2, 6, 25)
    teacher[:, 0, :, :15] = 0.1
    teacher[:, 1, :, :15] = 0.9
    output = model.generate(*source)["prediction"]
    loss = (output - teacher).square().mean()
    loss.backward()
    gradient = model.expert_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    # This loss must differ from fitting every slot to the teacher mean.
    mean_loss = (output - teacher.mean(1, keepdim=True)).square().mean()
    assert not torch.allclose(loss.detach(), mean_loss.detach())


def test_student_checkpoint_has_no_teacher_modules_or_paths():
    model = tiny_student(2)
    names = list(model.state_dict())
    assert names
    assert not any("teacher" in name for name in names)


def test_microfit_recovers_two_different_teacher_targets_for_same_input():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    model = tiny_student(2).train()
    source = inputs(batch=1)
    teacher = torch.zeros(1, 2, 6, 25)
    teacher[:, 0, :, :15] = 0.1
    teacher[:, 1, :, :15] = 0.9
    teacher[:, 0, :, 17] = 1.0
    teacher[:, 1, :, 18] = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        prediction = model.generate(*source)["prediction"]
        loss = (prediction - teacher).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model.generate(*source)["prediction"]
    assert (prediction - teacher).square().mean() < 2e-4
    assert (prediction[:, 0] - prediction[:, 1]).abs().mean() > 0.4
