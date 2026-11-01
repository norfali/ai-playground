"""Exercise the MLX boundary with local fixtures and fake runtime modules only."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_lab import generation
from rag_lab.generation import LocalGenerator

# Keep the real loader body available while conftest blocks all accidental loads.
load_runtime = generation._load_model.__wrapped__


@pytest.fixture
def directory(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}))
    (path / "model.safetensors").write_bytes(b"fake weights; never read by inference")
    return path


def test_local_load_uses_resolved_path_and_respects_smaller_window(directory, monkeypatch):
    calls = []
    fake_tokenizer = SimpleNamespace(model_max_length=1024)
    monkeypatch.setattr(
        generation, "_load_model", lambda path: calls.append(path) or (object(), fake_tokenizer)
    )
    generator = LocalGenerator(str(directory))
    generator.load()
    assert generator.context_window == 1024
    assert generator.tokenizer is fake_tokenizer
    assert calls == [str(directory.resolve())]


@pytest.mark.parametrize("config", ["[]", "{broken", '{"text_config": []}', "{}"])
def test_bad_or_unknown_model_configuration(directory, monkeypatch, config):
    (directory / "config.json").write_text(config)
    monkeypatch.setattr(generation, "_load_model", lambda path: (object(), object()))
    with pytest.raises(ValueError):
        LocalGenerator(str(directory)).load()


def test_missing_weights_are_rejected_before_runtime_import(directory):
    (directory / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="No local safetensors"):
        LocalGenerator(str(directory)).load()


def test_loader_passes_only_local_path_and_disables_remote_tokenizer_code(directory, monkeypatch):
    calls = []
    module = ModuleType("mlx_lm")
    module.load = lambda *args, **kwargs: calls.append((args, kwargs)) or ("model", "tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setattr(generation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(generation.platform, "machine", lambda: "arm64")
    assert load_runtime(str(directory)) == ("model", "tokenizer")
    assert calls == [
        (
            (str(directory),),
            {
                "tokenizer_config": {
                    "local_files_only": True,
                    "trust_remote_code": False,
                }
            },
        )
    ]


def test_missing_runtime_is_actionable(directory, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    monkeypatch.setattr(generation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(generation.platform, "machine", lambda: "arm64")
    with pytest.raises(RuntimeError, match="--extra llm"):
        load_runtime(str(directory))


def test_unsupported_platform_does_not_import_runtime(directory, monkeypatch):
    monkeypatch.setattr(generation.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        load_runtime(str(directory))


def test_generation_uses_exact_prepared_tokens_and_greedy_sampling(monkeypatch):
    calls = []
    module = ModuleType("mlx_lm")
    module.generate = lambda *args, **kwargs: calls.append((args, kwargs)) or "  Answer [S1]  "
    sampler_module = ModuleType("mlx_lm.sample_utils")
    sampler_module.make_sampler = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sampler_module)
    model = LocalGenerator("/unused")
    model.model, model.tokenizer = "model", "tokenizer"
    prepared = generation.PreparedGeneration(generation.assemble_context([]), [1, 9, 4], 1000, 512)
    assert model.generate(prepared) == "Answer [S1]"
    assert calls == [
        (
            ("model", "tokenizer"),
            {
                "prompt": [1, 9, 4],
                "max_tokens": 512,
                "sampler": {"temp": 0.0},
                "verbose": False,
            },
        )
    ]
