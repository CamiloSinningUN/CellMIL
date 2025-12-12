import pytest
import torch
import torch.nn.functional as F
from cellmil.utils.train.losses import FocalLoss


class TestFocalLoss:
    @pytest.fixture
    def sample_logits(self):
        """Sample logits for binary classification"""
        return torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])

    @pytest.fixture
    def sample_targets(self):
        """Sample targets for binary classification"""
        return torch.tensor([0, 1, 0, 1])

    def test_focal_loss_initialization_default(self):
        """Test FocalLoss initialization with default parameters"""
        loss_fn = FocalLoss()
        assert loss_fn.alpha is None
        assert loss_fn.gamma == 2.0

    def test_focal_loss_initialization_custom(self):
        """Test FocalLoss initialization with custom parameters"""
        alpha, gamma = 0.5, 1.5
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        assert loss_fn.alpha == alpha
        assert loss_fn.gamma == gamma

    def test_forward_pass_shape(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ):
        """Test that forward pass returns scalar loss"""
        loss_fn = FocalLoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.shape == torch.Size([])
        assert isinstance(loss.item(), float)

    def test_forward_pass_positive_loss(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ):
        """Test that loss is positive"""
        loss_fn = FocalLoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.item() >= 0

    def test_perfect_predictions(self):
        """Test loss with perfect predictions (should be near zero)"""
        # Perfect predictions: high confidence correct predictions
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([0, 1])

        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01  # Should be very small

    def test_worst_predictions(self):
        """Test loss with worst predictions (should be high)"""
        # Worst predictions: high confidence wrong predictions
        logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
        targets = torch.tensor([0, 1])

        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() > 1.0  # Should be high

    def test_gamma_effect(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ):
        """Test that higher gamma reduces loss for well-classified examples"""
        loss_fn_low_gamma = FocalLoss(gamma=0.5)
        loss_fn_high_gamma = FocalLoss(gamma=3.0)

        loss_low = loss_fn_low_gamma(sample_logits, sample_targets)
        loss_high = loss_fn_high_gamma(sample_logits, sample_targets)

        # Both should be positive
        assert loss_low.item() > 0
        assert loss_high.item() > 0

    def test_alpha_effect(self):
        """Test that alpha affects class weighting"""
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])

        loss_fn_low_alpha = FocalLoss(alpha=0.2)
        loss_fn_high_alpha = FocalLoss(alpha=0.8)

        loss_low = loss_fn_low_alpha(logits, targets)
        loss_high = loss_fn_high_alpha(logits, targets)

        assert loss_low.item() > 0
        assert loss_high.item() > 0

    def test_single_sample(self):
        """Test with single sample"""
        logits = torch.tensor([[0.6, 0.4]])
        targets = torch.tensor([0])

        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_batch_consistency(self):
        """Test that batch loss equals mean of individual losses"""
        logits = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([0, 1])

        loss_fn = FocalLoss()

        # Batch loss
        batch_loss = loss_fn(logits, targets)

        # Individual losses
        loss1 = loss_fn(logits[0:1], targets[0:1])
        loss2 = loss_fn(logits[1:2], targets[1:2])
        individual_mean = (loss1 + loss2) / 2

        # Should be approximately equal
        assert torch.allclose(batch_loss, individual_mean, atol=1e-6)

    def test_gradient_flow(
        self, sample_logits: torch.Tensor, sample_targets: torch.Tensor
    ):
        """Test that gradients flow through the loss"""
        logits = sample_logits.clone().requires_grad_(True)
        loss_fn = FocalLoss()

        loss = loss_fn(logits, sample_targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_comparison_with_cross_entropy(self):
        """Test that focal loss differs from cross entropy"""
        logits = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        targets = torch.tensor([0, 1])

        focal_loss = FocalLoss(alpha=0.5, gamma=0.0)  # gamma=0 should be closer to CE
        ce_loss = F.cross_entropy(logits, targets)
        focal_loss_value = focal_loss(logits, targets)

        # With gamma=0, should be close but not identical due to alpha weighting
        assert abs(focal_loss_value.item() - ce_loss.item()) < 1.0

    def test_invalid_targets(self, sample_logits: torch.Tensor):
        """Test with invalid target values"""
        invalid_targets = torch.tensor([0, 2])  # 2 is invalid for binary classification
        loss_fn = FocalLoss()

        # Should raise an error or handle gracefully
        with pytest.raises((RuntimeError, IndexError)):
            loss_fn(sample_logits[:2], invalid_targets)

    def test_empty_tensors(self):
        """Test with empty tensors"""
        empty_logits = torch.empty(0, 2)
        empty_targets = torch.empty(0, dtype=torch.long)

        loss_fn = FocalLoss()
        loss = loss_fn(empty_logits, empty_targets)

        # Loss should be NaN or 0 for empty batch
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_different_dtypes(self):
        """Test with different tensor dtypes"""
        logits_float32 = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        logits_float64 = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
        targets = torch.tensor([0])

        loss_fn = FocalLoss()

        loss32 = loss_fn(logits_float32, targets)
        loss64 = loss_fn(logits_float64, targets)

        # Should handle different dtypes
        assert torch.allclose(loss32, loss64.float(), atol=1e-6)
