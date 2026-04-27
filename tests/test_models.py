"""
Unit tests for ECGCNN and ECGCNNWithKAN.
All tests use CPU-only tensors and dummy inputs — no data files or GPU needed.
"""
import os
import pytest
import torch


@pytest.fixture(scope="module")
def dummy_batch():
    return torch.randn(2, 1, 256)


@pytest.fixture(scope="module")
def cnn_model():
    from src.models.cnn import ECGCNN
    model = ECGCNN(num_classes=5)
    model.eval()
    return model


@pytest.fixture(scope="module")
def kan_model():
    from src.models.cnn_kan import ECGCNNWithKAN
    model = ECGCNNWithKAN(num_classes=5)
    model.eval()
    return model


class TestECGCNN:
    def test_forward_shape(self, cnn_model, dummy_batch):
        with torch.no_grad():
            out = cnn_model(dummy_batch)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_feature_shape(self, cnn_model, dummy_batch):
        with torch.no_grad():
            feat = cnn_model.get_features(dummy_batch)
        assert feat.shape == (2, 256), f"Expected (2,256), got {feat.shape}"

    def test_parameter_count_positive(self, cnn_model):
        assert cnn_model.count_parameters() > 0


class TestECGCNNWithKAN:
    def test_forward_shape(self, kan_model, dummy_batch):
        with torch.no_grad():
            out = kan_model(dummy_batch)
        assert out.shape == (2, 5), f"Expected (2,5), got {out.shape}"

    def test_feature_shape(self, kan_model, dummy_batch):
        with torch.no_grad():
            feat = kan_model.get_features(dummy_batch)
        assert feat.shape == (2, 256)

    def test_parameter_breakdown(self, kan_model):
        bd = kan_model.parameter_breakdown()
        assert "encoder" in bd and "kan_head" in bd
        assert bd["total"] == bd["encoder"] + bd["kan_head"]
        assert bd["total"] > 0


class TestCheckpointLoading:
    @pytest.mark.skipif(
        not os.path.exists("checkpoints/cnn_mlp.pth"),
        reason="cnn_mlp.pth not present"
    )
    def test_load_cnn_mlp(self):
        from src.models.cnn import ECGCNN
        model = ECGCNN(num_classes=5)
        ckpt = torch.load("checkpoints/cnn_mlp.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 256))
        assert out.shape == (1, 5)

    @pytest.mark.skipif(
        not os.path.exists("checkpoints/cnn_kan.pth"),
        reason="cnn_kan.pth not present"
    )
    def test_load_cnn_kan(self):
        from src.models.cnn_kan import ECGCNNWithKAN
        model = ECGCNNWithKAN(num_classes=5)
        ckpt = torch.load("checkpoints/cnn_kan.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 256))
        assert out.shape == (1, 5)
