import socket

import pytest
import torch

from psdl1.model import build_backbone


def test_network_calls_fail_under_fixture():
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.invalid", 443))


def test_tiny_and_local_file_load_without_network(tmp_path):
    net = build_backbone("tiny", in_chans=4)
    file = tmp_path / "local.pt"
    torch.save(net.state_dict(), file)
    loaded = build_backbone("tiny", in_chans=4, weights_file=str(file))
    x = torch.zeros(1, 4, 16, 16)
    assert torch.equal(net(x), loaded(x))


def test_missing_weights_file_has_no_online_fallback(tmp_path):
    with pytest.raises(ValueError, match="local weights file"):
        build_backbone("tiny", weights_file=str(tmp_path / "missing.pt"))


def test_timm_local_file_loads_pretrained_false_without_network(tmp_path, monkeypatch):
    timm = pytest.importorskip("timm")
    original = timm.create_model
    flags = []
    def checked(name, **kwargs):
        flags.append(kwargs["pretrained"])
        assert kwargs["pretrained"] is False
        return original(name, **kwargs)
    monkeypatch.setattr(timm, "create_model", checked)
    net = build_backbone("convnextv2_atto", in_chans=4)
    file = tmp_path / "convnext.pt"
    torch.save(net.state_dict(), file)
    loaded = build_backbone("convnextv2_atto", in_chans=4, weights_file=str(file))
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), loaded.parameters()))
    assert flags == [False, False]


def test_bad_local_weights_format_and_nonfinite_values(tmp_path):
    file = tmp_path / "bad.pt"
    torch.save({"bad": torch.tensor(float("nan"))}, file)
    with pytest.raises(ValueError, match="finite state"):
        build_backbone("tiny", weights_file=str(file))


def test_n4_public_three_channel_classifier_state_is_not_silently_converted(tmp_path):
    timm = pytest.importorskip("timm")
    from psdl1.model import ProductEncoder
    public_shape = timm.create_model("convnextv2_atto", pretrained=False)
    file = tmp_path / "public-shaped-state.pt"
    torch.save(public_shape.state_dict(), file)
    with pytest.raises(RuntimeError, match="size mismatch|Unexpected key"):
        ProductEncoder(2, 2, backbone="convnextv2_atto", weights_file=str(file))
