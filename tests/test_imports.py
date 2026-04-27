"""
Smoke tests — verify all src modules import cleanly.
Catches missing dependencies and broken relative imports fast.
"""


def test_import_cnn():
    from src.models.cnn import ECGCNN, ConvBlock
    assert ECGCNN is not None
    assert ConvBlock is not None


def test_import_kan_head():
    from src.models.kan_head import KANHead, KANLayer
    assert KANHead is not None
    assert KANLayer is not None


def test_import_cnn_kan():
    from src.models.cnn_kan import ECGCNNWithKAN
    assert ECGCNNWithKAN is not None


def test_import_data_loader():
    from src.data.loader import create_dataloaders
    assert create_dataloaders is not None


def test_import_train():
    from src.training.train import train
    assert train is not None


def test_import_evaluate():
    from src.training.evaluate import evaluate
    assert evaluate is not None
