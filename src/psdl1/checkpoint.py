"""Versioned, atomic local checkpoints with integrity and resume state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .schema import SCHEMA_VERSION
from .model import MODEL_INPUT_VERSION

CHECKPOINT_VERSION = "dl1_checkpoint_v1"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    return json_hash({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("checkpoint state requires finite tensors")
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_config(config):
    required = {"grid", "roles", "layers", "backbone", "use_unit_size", "unknown_vocabulary", "model_kwargs"}
    if not isinstance(config, dict) or set(config) != required or type(config["grid"]) is not int or config["grid"] <= 0 or config["unknown_vocabulary"] != "reject":
        raise ValueError("incompatible model config/fingerprint")
    for name in ("roles", "layers"):
        vocab = config[name]
        if not isinstance(vocab, dict) or not vocab or any(not isinstance(k, str) or not k.strip() or type(v) is not int for k, v in vocab.items()) or sorted(vocab.values()) != list(range(1, len(vocab) + 1)):
            raise ValueError("vocabulary requires contiguous training-only IDs")
    if type(config["use_unit_size"]) is not bool or not isinstance(config["backbone"], str) or not config["backbone"] or not isinstance(config["model_kwargs"], dict):
        raise ValueError("invalid model configuration")
    if set(config["model_kwargs"]) - {"dim", "out_dim", "heads", "blocks"}:
        raise ValueError("unknown architecture fields")
    return json_hash({"schema_version": SCHEMA_VERSION, "model_input_version": MODEL_INPUT_VERSION, "config": config})


def capture_rng(generator=None):
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(), "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "generator": None if generator is None else generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator=None):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["kind"], np.array(ns["keys"], dtype=np.uint32), ns["position"], ns["has_gauss"], ns["cached_gaussian"]))
    if state["generator"] is not None:
        if generator is None:
            raise ValueError("resume requires sampler generator")
        generator.set_state(state["generator"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored on this device")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, config, *, optimizer=None, generator=None, epoch=0, history=None, metadata=None):
    fingerprint = validate_config(config)
    state = model.state_dict()
    saved = {"format_version": CHECKPOINT_VERSION, "schema_version": SCHEMA_VERSION,
             "model_input_version": MODEL_INPUT_VERSION, "fingerprint": fingerprint,
             "config": config, "state_dict": state, "weights_hash": state_hash(state),
             "optimizer": None if optimizer is None else optimizer.state_dict(), "rng": capture_rng(generator),
             "epoch": epoch, "history": list(history or []), "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(saved, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return saved


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != CHECKPOINT_VERSION or saved.get("schema_version") != SCHEMA_VERSION or saved.get("model_input_version") != MODEL_INPUT_VERSION:
        raise ValueError("incompatible checkpoint/schema version")
    required = {"config", "state_dict", "optimizer", "rng", "epoch", "history", "metadata", "fingerprint", "weights_hash"}
    if not required <= saved.keys() or not isinstance(saved["state_dict"], dict) or not isinstance(saved["metadata"], dict):
        raise ValueError("incomplete checkpoint: required fields are missing or malformed")
    if saved.get("fingerprint") != validate_config(saved["config"]):
        raise ValueError("incompatible checkpoint fingerprint")
    if saved.get("weights_hash") != state_hash(saved["state_dict"]):
        raise ValueError("checkpoint weights hash mismatch")
    return saved


VOCAB_KEYS = {"roles": "role.weight", "layers": "layer.weight"}


def extend_vocab(parent, names):
    """부모 번호는 그대로 두고 부모에 없는 이름만 정렬해 뒤에 붙인다. 부모에만 있는 이름도 남긴다."""
    vocab = dict(parent)
    for name in sorted(set(names) - vocab.keys()):
        vocab[name] = len(vocab) + 1
    return vocab


def check_parent_compatible(parent_config, config):
    for field in ("backbone", "use_unit_size", "grid", "model_kwargs", "unknown_vocabulary"):
        if parent_config[field] != config[field]:
            raise ValueError(f"init-from parent differs: {field}")
    for field in VOCAB_KEYS:
        if any(config[field].get(name) != number for name, number in parent_config[field].items()):
            raise ValueError(f"init-from parent differs: {field}")


def warm_start_state(parent_state, parent_config, config, current_state):
    """부모 state 사본에서 어휘 임베딩만 확장한다. 새 행은 current_state(현재 모델)의 행 그대로이고 RNG 는 쓰지 않는다."""
    check_parent_compatible(parent_config, config)
    if parent_state.keys() != current_state.keys():
        raise ValueError("init-from parent differs: state keys")
    state = {key: value.clone() for key, value in parent_state.items()}
    for field, key in VOCAB_KEYS.items():
        old, new = parent_state[key], current_state[key]
        if old.shape[0] != len(parent_config[field]) + 1 or new.shape[0] != len(config[field]) + 1 or old.shape[1:] != new.shape[1:]:
            raise ValueError(f"init-from parent differs: {field}")
        grown = new.detach().clone()
        grown[:old.shape[0]] = old
        state[key] = grown
    return state


def init_from_checkpoint(path, model, *, expected_config, saved=None):
    """부모 가중치만 싣는다(optimizer·RNG 는 건드리지 않는다). saved 가 있으면 이미 읽은 부모를 쓴다."""
    saved = read_checkpoint(path) if saved is None else saved
    state = warm_start_state(saved["state_dict"], saved["config"], expected_config, model.state_dict())
    model.load_state_dict(state, strict=True)
    warm = {}
    for field in VOCAB_KEYS:
        parent_names = saved["config"][field]
        warm[field] = {"parent": len(parent_names), "total": len(expected_config[field]),
                       "new_names": sorted(set(expected_config[field]) - parent_names.keys())}
    return {"saved": saved, "warm_start": warm}


def resume_checkpoint(path, model, optimizer, generator, *, expected_config=None, expected_input_hash=None, expected_contract_hash=None, expected_code_hash=None):
    saved = read_checkpoint(path)
    if expected_config is not None and saved["fingerprint"] != validate_config(expected_config):
        raise ValueError("resume config differs")
    if expected_input_hash is not None and saved["metadata"].get("input_hash") != expected_input_hash:
        raise ValueError("resume input hash differs")
    if expected_contract_hash is not None and saved["metadata"].get("contract_hash") != expected_contract_hash:
        raise ValueError("resume training/split/relation contract differs")
    if expected_code_hash is not None and saved["metadata"].get("code_hash") != expected_code_hash:
        raise ValueError("resume code hash differs")
    if saved["optimizer"] is None:
        raise ValueError("checkpoint has no optimizer for resume")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"], generator)
    return saved
