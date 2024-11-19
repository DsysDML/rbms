import pytest
import torch

from rbms.custom_fn import log2cosh, one_hot


def test_one_hot_happy_path():
    # Arrange
    x = torch.tensor([[0, 1, 2], [1, 0, 2]])
    num_classes = 3
    expected_output = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    # Act
    result = one_hot(x, num_classes)

    # Assert
    assert torch.equal(result, expected_output)


def test_one_hot_auto_num_classes():
    # Arrange
    x = torch.tensor([[0, 1, 2], [1, 0, 2]])
    expected_output = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    # Act
    result = one_hot(x)

    # Assert
    assert torch.equal(result, expected_output)


def test_one_hot_invalid_input():
    # Arrange
    x = torch.tensor([[0, 1, 2], [1, 0, 3]])
    num_classes = 3

    # Act & Assert
    with pytest.raises(IndexError):
        one_hot(x, num_classes)


def test_log2cosh():
    # Arrange
    x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    expected_output = torch.log(2 * torch.cosh(x))

    # Act
    result = log2cosh(x)

    # Assert
    assert torch.allclose(result, expected_output, atol=1e-4)


def test_log2cosh_empty_input():
    # Arrange
    x = torch.tensor([])
    expected_output = torch.tensor([])

    # Act
    result = log2cosh(x)

    # Assert
    assert torch.equal(result, expected_output)


def test_log2cosh_large_values():
    # Arrange
    x = torch.tensor([1e6, -1e6])
    expected_output = torch.tensor([1e6, 1e6])

    # Act
    result = log2cosh(x)

    # Assert
    assert torch.allclose(result, expected_output, atol=1e-4)
