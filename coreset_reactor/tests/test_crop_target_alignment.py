import torch
import pytest

from coreset_reactor.crop_target_alignment import align_targets_then_crop


class RecordingProcessor:
    def forward(self, predictions, targets):
        self.length = predictions[0].shape[1]
        return [torch.stack(targets[0])]


def test_equal_length_gt_keeps_source_time_window():
    processor = RecordingProcessor()
    gt = torch.arange(12.).view(12, 1).expand(-1, 25)
    pred = torch.zeros(10, 4, 25)
    out = align_targets_then_crop(processor, pred, [gt], source_length=12,
                                  crop_start=4, device=torch.device('cpu'),
                                  seed=1234, clip_id='example')
    assert processor.length == 12
    torch.testing.assert_close(out[0], gt[4:8])


def test_uncropped_input_is_identity_and_rng_is_restored():
    processor = RecordingProcessor()
    gt = torch.rand(5, 25)
    state = torch.random.get_rng_state().clone()
    out = align_targets_then_crop(processor, torch.zeros(10, 5, 25), [gt],
                                  source_length=5, crop_start=0,
                                  device=torch.device('cpu'), seed=1234, clip_id='short')
    torch.testing.assert_close(out[0], gt)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('start,total', [(-1, 12), (9, 12), (0, 0)])
def test_invalid_crop_rejected(start, total):
    with pytest.raises(ValueError, match='Crop'):
        align_targets_then_crop(RecordingProcessor(), torch.zeros(10, 4, 25), [],
                                source_length=total, crop_start=start,
                                device=torch.device('cpu'), seed=1, clip_id='bad')
