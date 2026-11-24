import numpy as np
import pytest
import torch

from eurosat.data import random_flip_rotate
from eurosat.models import DeepMLP, SpectralCNN, build, count_parameters
from eurosat.train import evaluate


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_models_output_one_logit_per_class(name):
    model = build(name, num_classes=10) if name == "cnn" else DeepMLP(hidden_sizes=(64, 32))
    assert model(torch.rand(2, 13, 64, 64)).shape == (2, 10)


def test_cnn_is_much_smaller_than_mlp():
    assert count_parameters(SpectralCNN()) * 100 < count_parameters(DeepMLP())


def test_augmentation_preserves_shape_and_values():
    x = torch.rand(13, 64, 64)
    y = random_flip_rotate(x)
    assert y.shape == x.shape
    assert torch.isclose(y.sum(), x.sum())


def test_evaluate_uses_probabilities():
    labels = np.array([0, 1, 2, 2])
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5]])
    m = evaluate(probs, labels)
    assert m["accuracy"] == 1.0
    assert 0.99 <= m["roc_auc_macro_ovr"] <= 1.0
