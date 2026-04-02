import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.export_onnx import (
    FrameGraph,
    export,
    palette_tensor,
    random_inputs,
    randomize_zero_init,
    run_onnx,
)
from scripts.model import Denoiser, group_norm, n_actions, sample, snap

width = 8
context = 4
batch = 2


def make_denoiser() -> Denoiser:
    torch.manual_seed(0)
    denoiser = Denoiser(width, context)
    randomize_zero_init(denoiser)
    return denoiser.eval()


def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    frames = torch.rand(batch, 3 * context, 64, 64, generator=generator) * 2 - 1
    actions = torch.randint(0, n_actions, (batch, context), generator=generator)
    noise = torch.randn(batch, 3 * (context + 1), 64, 64, generator=generator)
    return frames, actions, noise


def test_shapes() -> None:
    denoiser = make_denoiser()
    frames, actions, noise = make_batch()

    with torch.inference_mode():
        denoised = denoiser(noise[:, :3], torch.ones(batch), frames, actions, torch.zeros(batch))
        frame = sample(denoiser, frames, actions, noise)
        snapped = snap(frame, palette_tensor())

    assert denoised.shape == frame.shape == snapped.shape == (batch, 3, 64, 64)


def test_group_norm_matches_torch() -> None:
    x = torch.randn(batch, 32, 8, 8)
    torch.testing.assert_close(group_norm(x, 8), F.group_norm(x, 8))


def test_denoiser_returns_input_as_sigma_goes_to_zero() -> None:
    denoiser = make_denoiser()
    frames, actions, noise = make_batch()
    x = noise[:, :3]

    with torch.inference_mode():
        denoised = denoiser(x, torch.full((batch,), 1e-6), frames, actions, torch.zeros(batch))

    torch.testing.assert_close(denoised, x, atol=1e-4, rtol=0)


def test_one_step_is_one_denoiser_call() -> None:
    denoiser = make_denoiser()
    frames, actions, noise = make_batch()
    sigma_max = 5.0

    with torch.inference_mode():
        frame = sample(denoiser, frames, actions, noise, steps=1, sigma_max=sigma_max)
        denoised = denoiser(sigma_max * noise[:, :3], torch.full((batch,), sigma_max), frames, actions, torch.zeros(batch))

    assert torch.equal(frame, denoised)


def test_snap() -> None:
    palette = palette_tensor()
    image = palette.T[None, :, None, :]  # shape: (1, 3, 1, colors)
    assert torch.equal(snap(image, palette), image)

    # Colors are at least 2 apart, so an offset shorter than 1 keeps each pixel nearest its own color
    offset = torch.tensor([0.4, -0.3, 0.2])[None, :, None, None]
    assert torch.equal(snap(image + offset, palette), image)


def test_onnx_parity_before_snap() -> None:
    graph = FrameGraph(make_denoiser(), steps=3, palette=None).eval()
    generator = torch.Generator().manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.onnx"
        export(graph, random_inputs(context, generator), path)
        inputs = random_inputs(context, generator)
        actual = run_onnx(path, inputs)

    with torch.inference_mode():
        expected = graph(*inputs).numpy()
    assert np.abs(actual - expected).max() < 1e-4
