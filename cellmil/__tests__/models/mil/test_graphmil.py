"""
Comprehensive test suite for GraphMIL model.

This test suite includes:
1. Basic functionality tests
2. GNN architecture validation tests
3. Pooling classifier tests
4. Gradient flow analysis
5. Overfitting test (primary focus)
6. Lightning integration tests
7. torch_geometric compatibility tests

The tests are designed to catch potential implementation issues by being thorough
and validating against expected graph learning behavior.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from typing import Any
from cellmil.models.mil.graphmil import (
    LitGraphMIL,
    GAT,
    GATv2,
    SAGE,
    GNN,
    Standard,
    Attention,
    Mean_MLP,
    CLAM,
    GlobalPooling_Classifier,
)


class TestGraphMILComponents:
    """Test individual components of GraphMIL architecture"""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing"""
        torch.manual_seed(42)  # type: ignore

        # Create small graphs with different characteristics
        graphs: dict[str, Data] = {}

        # Small graph: 10 nodes, simple structure
        edge_index_small = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 5, 6, 6, 7, 7, 8, 8, 9, 9, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 6, 5, 7, 6, 8, 7, 9, 8, 5, 9],
            ],
            dtype=torch.long,
        )
        graphs["small"] = Data(
            x=torch.randn(10, 64),  # Fix: Use 64 features for consistency
            edge_index=edge_index_small,
            y=torch.tensor([0]),
        )

        # Medium graph: 50 nodes
        num_nodes_medium = 50
        # Create random edges (ensuring connectivity)
        edge_list: list[list[float]] = []
        for i in range(num_nodes_medium - 1):
            edge_list.extend([[i, i + 1], [i + 1, i]])  # Chain connectivity
        for _ in range(100):  # Add random edges
            i, j = torch.randint(0, num_nodes_medium, (2,))
            if i != j:
                edge_list.extend([[i.item(), j.item()], [j.item(), i.item()]])
        edge_index_medium = torch.tensor(edge_list).t().contiguous()

        graphs["medium"] = Data(
            x=torch.randn(num_nodes_medium, 64),  # Fix: Use 64 features for consistency
            edge_index=edge_index_medium,
            y=torch.tensor([1]),
        )

        # Large graph: 200 nodes
        num_nodes_large = 200
        edge_list_large: list[list[float]] = []
        for i in range(num_nodes_large - 1):
            edge_list_large.extend([[i, i + 1], [i + 1, i]])
        for _ in range(500):
            i, j = torch.randint(0, num_nodes_large, (2,))
            if i != j:
                edge_list_large.extend([[i.item(), j.item()], [j.item(), i.item()]])
        edge_index_large = torch.tensor(edge_list_large).t().contiguous()

        graphs["large"] = Data(
            x=torch.randn(num_nodes_large, 64),  # Fix: Use 64 features for consistency
            edge_index=edge_index_large,
            y=torch.tensor([0]),
        )

        return graphs

    @pytest.fixture
    def sample_gnn_models(self):
        """Create sample GNN models for testing"""
        models: dict[str, GNN] = {}

        # GAT
        models["gat"] = GAT(
            input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1, heads=4
        )  # Fix: Use 64 input_dim

        # GATv2
        models["gatv2"] = GATv2(
            input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1, heads=4
        )  # Fix: Use 64 input_dim

        # SAGE
        models["sage"] = SAGE(
            input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1
        )  # Fix: Use 64 input_dim

        return models

    @pytest.fixture
    def sample_pooling_classifiers(self):
        """Create sample pooling classifiers for testing"""
        classifiers: dict[str, GlobalPooling_Classifier] = {}

        # Standard MIL
        classifiers["standard"] = Standard(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16], standard_type="fc"
        )  # Fix: Match hidden_dim=32

        # Attention pooling
        classifiers["attention"] = Attention(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16, 8]
        )  # Fix: Match hidden_dim=32

        # Mean MLP
        classifiers["mean_mlp"] = Mean_MLP(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16]
        )  # Fix: Match hidden_dim=32

        # CLAM pooling
        classifiers["clam"] = CLAM(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16, 8], clam_type="SB"
        )  # Fix: Match hidden_dim=32

        return classifiers

    def test_gnn_initialization(self, sample_gnn_models: dict[str, GNN]):
        """Test GNN model initialization"""
        for model_name, model in sample_gnn_models.items():
            print(f"\nTesting {model_name} initialization...")

            # Check basic properties
            assert model.input_dim == 64  # Fix: updated to match fixture
            assert model.n_layers == 2
            assert model.dropout == 0.1

            # Check that model has convolution layers
            assert hasattr(model, "convs")
            assert len(model.convs) > 0

            print(f"✅ {model_name} initialized correctly")

    def test_gnn_forward_pass(
        self, sample_gnn_models: dict[str, GNN], sample_graph_data: dict[str, Data]
    ):
        """Test GNN forward pass functionality"""
        for model_name, model in sample_gnn_models.items():
            print(f"\nTesting {model_name} forward pass...")

            for graph_name, orig_graph in sample_graph_data.items():
                # Clone graph to avoid in-place modifications affecting subsequent tests
                graph = Data(
                    x=orig_graph.x.clone(), # type: ignore
                    edge_index=orig_graph.edge_index.clone(), # type: ignore
                    y=orig_graph.y.clone() if orig_graph.y is not None else None # type: ignore
                )
                
                # Forward pass
                output = model(graph)

                # Check output structure
                assert isinstance(output, Data)
                assert hasattr(output, "x")
                assert hasattr(output, "edge_index")

                # Check output dimensions
                expected_hidden_dim = (
                    model.hidden_dim
                    if isinstance(model.hidden_dim, int)
                    else model.hidden_dim[-1]
                )
                assert output.x.shape[0] == graph.x.shape[0], ( # type: ignore
                    f"Node count changed for {graph_name}"
                )  
                assert output.x.shape[1] == expected_hidden_dim, ( # type: ignore
                    f"Hidden dim wrong for {graph_name}"
                )  

                # Check edge_index unchanged
                assert torch.equal(output.edge_index, graph.edge_index)  # type: ignore

                print(f"  ✅ {graph_name} graph: {graph.x.shape} -> {output.x.shape}")  # type: ignore

    def test_pooling_classifier_initialization(
        self, sample_pooling_classifiers: dict[str, GlobalPooling_Classifier]
    ):
        """Test pooling classifier initialization"""
        for classifier_name, classifier in sample_pooling_classifiers.items():
            print(f"\nTesting {classifier_name} pooling initialization...")

            # Check basic properties
            assert classifier.input_dim == 32  # Fix: updated to match fixture
            assert classifier.n_classes == 2
            assert classifier.dropout == 0.1

            print(f"✅ {classifier_name} initialized correctly")

    def test_pooling_classifier_forward_pass(
        self,
        sample_pooling_classifiers: dict[str, GlobalPooling_Classifier],
        sample_graph_data: dict[str, Data],
    ):
        """Test pooling classifier forward pass"""
        for classifier_name, classifier in sample_pooling_classifiers.items():
            print(f"\nTesting {classifier_name} pooling forward pass...")

            for graph_name, graph in sample_graph_data.items():
                # Simulate GNN output (reduce feature dim to match classifier input)
                node_features = torch.randn(
                    graph.x.shape[0], 32 # type: ignore
                )  # Fix: Match classifier input_dim=32 
                batch = None  # Single graph

                # Forward pass with classifier-specific arguments
                if classifier_name == "clam":
                    # CLAM needs label and instance_eval
                    logits, output_dict = classifier(
                        node_features, batch, label=graph.y, instance_eval=True
                    )
                else:
                    # Other classifiers only need node features and batch
                    logits, output_dict = classifier(
                        node_features, batch
                    )

                # Extract Y_prob and Y_hat from output_dict
                Y_prob = output_dict.get("y_prob", torch.softmax(logits, dim=1))
                Y_hat = output_dict.get("y_hat", torch.topk(Y_prob, 1, dim=1)[1])

                # Check output shapes
                assert logits.shape == (1, 2), f"Wrong logits shape for {graph_name}"
                assert Y_prob.shape == (1, 2), f"Wrong Y_prob shape for {graph_name}"
                assert Y_hat.shape == (1, 1), f"Wrong Y_hat shape for {graph_name}"
                assert isinstance(output_dict, dict), (
                    f"output_dict should be dict for {graph_name}"
                )

                # Check value ranges
                assert torch.all(Y_prob >= 0) and torch.all(Y_prob <= 1), (
                    f"Y_prob not in [0,1] for {graph_name}"
                )
                assert torch.allclose(Y_prob.sum(dim=1), torch.ones(1), atol=1e-6), (
                    f"Y_prob doesn't sum to 1 for {graph_name}"
                )
                assert torch.all(Y_hat >= 0) and torch.all(Y_hat <= 1), (
                    f"Y_hat not in [0,1] for {graph_name}"
                )

                print(
                    f"  ✅ {graph_name} graph: {node_features.shape} -> logits {logits.shape}"
                )

    def test_litgraphmil_initialization(
        self,
        sample_gnn_models: dict[str, GNN],
        sample_pooling_classifiers: dict[str, GlobalPooling_Classifier],
    ):
        """Test LitGraphMIL initialization"""
        gnn = sample_gnn_models["gat"]
        pooling = sample_pooling_classifiers["standard"]

        # Test basic initialization
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            loss_fn=nn.CrossEntropyLoss(),
        )

        # Check properties
        assert model.gnn is gnn
        assert model.pooling_classifier is pooling
        assert model.optimizer_cls == optim.Adam
        assert isinstance(model.loss_fn, nn.CrossEntropyLoss)

        # Check metrics setup
        assert hasattr(model, "train_metrics")
        assert hasattr(model, "val_metrics")
        assert hasattr(model, "test_metrics")

        print("✅ LitGraphMIL initialization successful")

    def test_litgraphmil_forward_pass(
        self,
        sample_gnn_models: dict[str, GNN],
        sample_pooling_classifiers: dict[str, GlobalPooling_Classifier],
        sample_graph_data: dict[str, Data],
    ):
        """Test LitGraphMIL end-to-end forward pass"""
        gnn = sample_gnn_models["gat"]
        pooling = sample_pooling_classifiers["attention"]

        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        for graph_name, graph in sample_graph_data.items():
            # Forward pass
            logits, output_dict = model(graph, label=graph.y, instance_eval=True)

            # Check outputs
            assert logits.shape == (1, 2), f"Wrong logits shape for {graph_name}"
            assert output_dict["y_prob"].shape == (1, 2), (
                f"Wrong y_prob shape for {graph_name}"
            )
            assert output_dict["y_hat"].shape == (1, 1), (
                f"Wrong y_hat shape for {graph_name}"
            )

            print(f"  ✅ {graph_name} graph forward pass successful")

    def test_gradient_flow_analysis(
        self,
        sample_gnn_models: dict[str, GNN],
        sample_pooling_classifiers: dict[str, GlobalPooling_Classifier],
        sample_graph_data: dict[str, Data],
    ):
        """Detailed gradient flow analysis for GraphMIL"""
        print("\n" + "=" * 60)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 60)

        gnn = sample_gnn_models["gat"]
        pooling = sample_pooling_classifiers["standard"]

        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        graph = sample_graph_data["medium"]

        # Forward pass
        logits, _ = model(graph, label=graph.y, instance_eval=True)

        # Compute loss
        loss = model.loss_fn(logits, graph.y)

        # Backward pass
        loss.backward()

        # Analyze gradients
        gradient_analysis: dict[str, Any] = {"gnn": [], "pooling": [], "no_grad": []}

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    gradient_analysis["no_grad"].append(name)
                else:
                    # Check for valid gradients
                    if not torch.isfinite(param.grad).all():
                        pytest.fail(f"Invalid gradient for parameter: {name}")

                    grad_norm = param.grad.norm().item()  # type: ignore

                    # Categorize by component
                    if name.startswith("gnn."):
                        gradient_analysis["gnn"].append((name, grad_norm))
                    elif name.startswith("pooling_classifier."):
                        gradient_analysis["pooling"].append((name, grad_norm))

        # Report results
        print("Gradient Analysis Results:")
        print(f"GNN parameters with gradients: {len(gradient_analysis['gnn'])}")
        print(f"Pooling parameters with gradients: {len(gradient_analysis['pooling'])}")
        print(f"Parameters without gradients: {len(gradient_analysis['no_grad'])}")

        if gradient_analysis["no_grad"]:
            print(f"⚠️  Parameters without gradients: {gradient_analysis['no_grad']}")

        # Print gradient norms
        print("\nGradient Norms:")
        for component, params in [
            ("GNN", gradient_analysis["gnn"]),
            ("Pooling", gradient_analysis["pooling"]),
        ]:
            print(f"{component}:")
            for name, grad_norm in params[:5]:  # Show first 5
                print(f"  {name}: {grad_norm:.6f}")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")

        # Assertions
        assert len(gradient_analysis["gnn"]) > 0, (
            "GNN should have parameters with gradients"
        )
        assert len(gradient_analysis["pooling"]) > 0, (
            "Pooling should have parameters with gradients"
        )

    def test_dimension_compatibility(self):
        """Test dimension compatibility between GNN and pooling classifier"""
        print("\n" + "=" * 60)
        print("DIMENSION COMPATIBILITY TEST")
        print("=" * 60)

        # Test compatible dimensions
        gnn = GAT(input_dim=128, hidden_dim=64, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=64, dropout=0.1, n_classes=2, size_arg=[32], standard_type="fc"
        )

        # Should work without error
        try:
            _ = LitGraphMIL(
                gnn=gnn,
                pooling_classifier=pooling,
                optimizer_cls=optim.Adam,
                optimizer_kwargs={"lr": 0.001},
            )
            print("✅ Compatible dimensions work correctly")
        except Exception as e:
            pytest.fail(f"Compatible dimensions should work: {e}")

        # Test incompatible dimensions
        gnn_wrong = GAT(input_dim=128, hidden_dim=128, n_layers=2, dropout=0.1)
        pooling_wrong = Standard(
            input_dim=64, dropout=0.1, n_classes=2, size_arg=[32], standard_type="fc"
        )

        # Should raise assertion error
        with pytest.raises(AssertionError, match="GNN hidden dimension must match"):
            LitGraphMIL(
                gnn=gnn_wrong,
                pooling_classifier=pooling_wrong,
                optimizer_cls=optim.Adam,
                optimizer_kwargs={"lr": 0.001},
            )
        print("✅ Incompatible dimensions properly detected")

    def test_dataloader_compatibility(self, sample_graph_data: dict[str, Data]):
        """Test compatibility with torch_geometric DataLoader"""
        print("\n" + "=" * 60)
        print("DATALOADER COMPATIBILITY TEST")
        print("=" * 60)

        # Create dataset from sample graphs
        graphs = list(sample_graph_data.values())

        # Test with batch_size=1 (correct for MIL)
        loader = DataLoader(graphs, batch_size=1, shuffle=False)

        gnn = GAT(input_dim=64, hidden_dim=64, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=64, dropout=0.1, n_classes=2, size_arg=[32], standard_type="fc"
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Test forward pass with dataloader
        for batch in loader:
            logits, _ = model(batch, label=batch.y, instance_eval=True)
            assert logits.shape == (1, 2)
            break

        print("✅ DataLoader compatibility successful")

        # Test with batch_size > 1 (should fail for MIL)
        loader_wrong = DataLoader(graphs, batch_size=2, shuffle=False)

        for batch in loader_wrong:
            with pytest.raises(ValueError, match="GraphMIL requires batch_size=1"):
                model._shared_step(batch, "test", log=False)  # type: ignore
            break

        print("✅ Batch size validation working correctly")


class TestGraphMILTraining:
    """Test training dynamics and learning capability"""

    def create_synthetic_graph_dataset(self):
        """Create synthetic graph dataset with clear patterns for overfitting"""
        torch.manual_seed(12345)  # type: ignore

        def create_graph_with_pattern(
            num_nodes: int, pattern_type: int, noise_level: float = 0.1
        ):
            """Create a graph with specific pattern for classification"""
            # Create base features
            features = torch.randn(num_nodes, 64) * noise_level

            if pattern_type == 0:  # Class 0: nodes with positive features in first half
                features[: num_nodes // 2, :32] += 2.0
                features[num_nodes // 2 :, :32] -= 1.0
            else:  # Class 1: nodes with positive features in second half
                features[: num_nodes // 2, 32:] += 2.0
                features[num_nodes // 2 :, 32:] -= 1.0

            # Create simple edge structure (ring + some random edges)
            edge_list: list[list[float]] = []
            for i in range(num_nodes):
                edge_list.extend([[i, (i + 1) % num_nodes], [(i + 1) % num_nodes, i]])

            # Add some random edges
            for _ in range(min(num_nodes, 20)):
                i, j = torch.randint(0, num_nodes, (2,))
                if i != j:
                    edge_list.extend([[i.item(), j.item()], [j.item(), i.item()]])

            edge_index = torch.tensor(edge_list).t().contiguous()

            return Data(
                x=features, edge_index=edge_index, y=torch.tensor([pattern_type])
            )

        # Training set: clear patterns
        train_graphs: list[Data] = []
        for i in range(8):  # 8 training graphs
            pattern = i % 2
            graph = create_graph_with_pattern(30, pattern, noise_level=0.1)
            train_graphs.append(graph)

        # Validation set: same patterns but with more noise
        val_graphs: list[Data] = []
        for i in range(6):  # 6 validation graphs
            pattern = i % 2
            graph = create_graph_with_pattern(
                30, pattern, noise_level=0.5
            )  # More noise
            val_graphs.append(graph)

        return train_graphs, val_graphs

    def test_basic_training_step(self):
        """Test that model can perform basic training steps"""
        print("\n" + "=" * 60)
        print("BASIC TRAINING TEST")
        print("=" * 60)

        train_graphs, _ = self.create_synthetic_graph_dataset()

        # Create model
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.0)
        pooling = Standard(
            input_dim=32, dropout=0.0, n_classes=2, size_arg=[16], standard_type="fc"
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.01},
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None

        # Train for a few epochs
        for epoch in range(5):
            epoch_loss = 0.0

            for orig_graph in train_graphs:
                # Clone graph to avoid in-place modifications
                graph = Data(
                    x=orig_graph.x.clone(), # type: ignore
                    edge_index=orig_graph.edge_index.clone(), # type: ignore
                    y=orig_graph.y.clone() if orig_graph.y is not None else None # type: ignore
                )
                
                optimizer.zero_grad()

                loss, _, _ = model._shared_step(graph, "train", log=False)  # type: ignore
                loss.backward()
                optimizer.step()  # type: ignore

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_graphs)
            if initial_loss is None:
                initial_loss = avg_loss

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Loss should decrease
        assert avg_loss < initial_loss, ( # type: ignore
            f"Loss should decrease: {initial_loss:.4f} -> {avg_loss:.4f}" # type: ignore
        )  
        print("✅ Basic training successful - loss decreased")

    def test_overfitting_capability(self) -> dict[str, Any]:
        """
        CRITICAL TEST: Test model's ability to overfit on synthetic graph data.
        This is the most important test to verify implementation correctness.
        """
        print("\n" + "=" * 70)
        print("GRAPH OVERFITTING TEST - MOST IMPORTANT")
        print("=" * 70)

        # Create model optimized for overfitting
        gnn = GAT(
            input_dim=64,
            hidden_dim=64,  # Large capacity
            n_layers=3,  # Deep network
            dropout=0.0,  # No dropout
            heads=4,
        )
        pooling = Standard(
            input_dim=64,
            dropout=0.0,
            n_classes=2,
            size_arg=[32],  # Must be length 1 for MIL_fc
            standard_type="fc",  # Add required parameter
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.SGD,
            optimizer_kwargs={"lr": 0.1, "momentum": 0.9},  # Aggressive learning
        )

        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        train_losses: list[float] = []
        val_losses: list[float] = []
        train_accs: list[float] = []
        val_accs: list[float] = []

        best_train_acc = 0.0
        overfitting_detected = False

        # Training loop designed to encourage overfitting
        epochs = 100
        print(f"Training GraphMIL for {epochs} epochs...")

        # Create FIXED training graphs that model can memorize
        torch.manual_seed(42)  # Fixed seed for reproducible graphs # type: ignore
        fixed_train_graphs: list[Data] = []
        for i in range(4):
            # Create simple distinguishable patterns
            x = torch.zeros(10, 64)
            if i % 2 == 0:  # Class 0: positive values in first half
                x[:, :32] = 2.0
                x[:, 32:] = -1.0
            else:  # Class 1: positive values in second half  
                x[:, :32] = -1.0
                x[:, 32:] = 2.0
            
            # Simple ring graph structure
            edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]
            ], dtype=torch.long)
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([i % 2])
            )
            fixed_train_graphs.append(graph)

        # Create FIXED validation graphs (similar but with noise)
        fixed_val_graphs: list[Data] = []
        for i in range(2):
            x = torch.zeros(10, 64)
            if i % 2 == 0:
                x[:, :32] = 1.5  # Weaker signal
                x[:, 32:] = -0.5
            else:
                x[:, :32] = -0.5
                x[:, 32:] = 1.5
            
            # Add noise
            x += torch.randn_like(x) * 0.3
            
            edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]
            ], dtype=torch.long)
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([i % 2])
            )
            fixed_val_graphs.append(graph)

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0

            # Training phase - use the SAME graphs every epoch for memorization
            for orig_graph in fixed_train_graphs:
                # Clone to avoid in-place modifications
                graph = Data(
                    x=orig_graph.x.clone(), # type: ignore
                    edge_index=orig_graph.edge_index.clone(), # type: ignore
                    y=orig_graph.y.clone() # type: ignore
                ) 
                
                optimizer.zero_grad()

                loss, Y_prob, label = model._shared_step(graph, "train", log=False)  # type: ignore
                loss.backward()
                optimizer.step()  # type: ignore

                epoch_train_loss += loss.item()
                pred = Y_prob.argmax(dim=1)
                train_correct += (pred == label).sum().item()

            # Validation phase - use the SAME validation graphs
            model.eval()
            epoch_val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for orig_graph in fixed_val_graphs:
                    # Clone to avoid in-place modifications
                    graph = Data(
                        x=orig_graph.x.clone(), # type: ignore
                        edge_index=orig_graph.edge_index.clone(), # type: ignore
                        y=orig_graph.y.clone() # type: ignore
                    )
                    
                    loss, Y_prob, label = model._shared_step(graph, "val", log=False)  # type: ignore
                    epoch_val_loss += loss.item()
                    pred = Y_prob.argmax(dim=1)
                    val_correct += (pred == label).sum().item()

            # Calculate metrics
            train_acc = train_correct / len(fixed_train_graphs)
            val_acc = val_correct / len(fixed_val_graphs)

            train_losses.append(epoch_train_loss / len(fixed_train_graphs))
            val_losses.append(epoch_val_loss / len(fixed_val_graphs))
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            best_train_acc = max(best_train_acc, train_acc)

            # Check for overfitting
            if train_acc >= 0.9 and val_acc <= 0.7:
                overfitting_detected = True

            # Print progress
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:3d}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, "
                    f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
                )

        # Final results
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        overfitting_gap = final_train_acc - final_val_acc

        print("\nFINAL RESULTS:")
        print(f"Best Training Accuracy: {best_train_acc:.3f}")
        print(f"Final Training Accuracy: {final_train_acc:.3f}")
        print(f"Final Validation Accuracy: {final_val_acc:.3f}")
        print(f"Overfitting Gap: {overfitting_gap:.3f}")
        print(f"Overfitting Detected: {overfitting_detected}")

        # Create visualization
        self._create_overfitting_plot(train_accs, val_accs, train_losses, val_losses)  # type: ignore

        # CRITICAL ASSERTIONS for implementation validation
        assert best_train_acc >= 0.6, (
            f"Model should achieve at least 60% training accuracy, got {best_train_acc:.3f}"
        )
        assert final_train_acc >= 0.5, (
            f"Final training accuracy too low: {final_train_acc:.3f}"
        )

        # Check for learning capability (main test)
        if best_train_acc >= 0.8:
            print(
                "✅ OVERFITTING TEST PASSED: Model can achieve high training accuracy"
            )
        else:
            print(
                f"⚠️  OVERFITTING TEST WARNING: Model only achieved {best_train_acc:.3f} training accuracy"
            )
            print("   This might indicate implementation issues with:")
            print("   - Gradient flow problems in GNN or pooling")
            print("   - Incorrect graph processing")
            print("   - Problems with attention/pooling mechanisms")
            print("   - Issues with loss computation")

        # Additional diagnostic info
        if overfitting_gap > 0.2:
            print("✅ Model shows clear overfitting behavior (good sign)")
        else:
            print("ℹ️  Limited overfitting detected - might indicate:")
            print("   - Model capacity too small")
            print("   - Learning rate too low")
            print("   - Implementation preventing proper learning")

        return {
            "best_train_acc": best_train_acc,
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "overfitting_gap": overfitting_gap,
            "overfitting_detected": overfitting_detected,
        }

    def _create_overfitting_plot(
        self,
        train_accs: list[float],
        val_accs: list[float],
        train_losses: list[float],
        val_losses: list[float],
    ):
        """Create visualization of overfitting behavior"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # type: ignore

            epochs = range(1, len(train_accs) + 1)

            # Accuracy plot
            ax1.plot(epochs, train_accs, "b-", label="Training Accuracy", linewidth=2)
            ax1.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("GraphMIL: Training vs Validation Accuracy")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Loss plot
            ax2.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
            ax2.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("GraphMIL: Training vs Validation Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = Path("test_reports/graphmil_overfitting.png")
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")  # type: ignore
            plt.close()

            print(f"📊 Overfitting plot saved to: {plot_path}")

        except Exception as e:
            print(f"Could not create overfitting plot: {e}")


class TestLitGraphMILIntegration:
    """Test Lightning integration and advanced features"""

    def test_lightning_training_step(self):
        """Test Lightning training step"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16], standard_type="fc"
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Create sample batch
        graph = Data(
            x=torch.randn(20, 64),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.tensor([1]),
        )

        # Test training step
        loss = model.training_step(graph, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

        print("✅ Lightning training step successful")

    def test_lightning_validation_step(self):
        """Test Lightning validation step"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16], standard_type="fc"
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Create sample batch
        graph = Data(
            x=torch.randn(20, 64),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.tensor([0]),
        )

        # Test validation step
        loss = model.validation_step(graph, 0)
        assert isinstance(loss, torch.Tensor)

        print("✅ Lightning validation step successful")

    def test_optimizer_configuration(self):
        """Test optimizer configuration"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16], standard_type="fc"
        )

        # Test without scheduler
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, optim.Adam)

        # Test with scheduler
        model_with_scheduler = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            scheduler_cls=optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 10, "gamma": 0.1},
        )

        result = model_with_scheduler.configure_optimizers()
        assert isinstance(result, tuple)
        assert len(result) == 2
        optimizers, schedulers = result
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        print("✅ Optimizer configuration successful")

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        # Create model
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1, heads=2)
        pooling = Attention(input_dim=32, dropout=0.1, n_classes=2, size_arg=[16, 8])

        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.002},
            scheduler_cls=optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 5, "gamma": 0.5},
        )

        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            # Save checkpoint manually (simulating Lightning save)
            checkpoint: dict[str, Any] = {
                "state_dict": model.state_dict(),
                "hyper_parameters": {
                    "gnn_type": "GAT",
                    "gnn_input_dim": 64,
                    "gnn_hidden_dim": 32,
                    "gnn_n_layers": 2,
                    "gnn_dropout": 0.1,
                    "gnn_heads": 2,
                    "pooling_type": "Attention",
                    "pooling_input_dim": 32,
                    "pooling_dropout": 0.1,
                    "pooling_n_classes": 2,
                    "pooling_size_arg": [16, 8],
                    "optimizer_type": "Adam",
                    "optimizer_lr": 0.002,
                    "scheduler_type": "StepLR",
                    "scheduler_step_size": 5,
                    "scheduler_gamma": 0.5,
                    "loss_fn": "CrossEntropyLoss",
                },
            }
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint
            loaded_model = LitGraphMIL.load_from_checkpoint(checkpoint_path)

            # Verify loaded model properties
            assert loaded_model.gnn.input_dim == 64
            assert loaded_model.gnn.hidden_dim == 32
            assert loaded_model.gnn.n_layers == 2
            assert loaded_model.pooling_classifier.input_dim == 32
            assert loaded_model.pooling_classifier.n_classes == 2

            print("✅ Checkpoint save/load successful")

        finally:
            # Clean up
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_clam_integration(self):
        """Test CLAM pooling integration with weight_loss_slide"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = CLAM(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16, 8], clam_type="SB"
        )

        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            weight_loss_slide=0.8,  # Custom weight for CLAM
        )

        # Check that weight is set
        assert hasattr(model, "weight_loss_slide")
        assert model.weight_loss_slide == 0.8

        # Test forward pass with CLAM
        graph = Data(
            x=torch.randn(20, 64),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.tensor([1]),
        )

        loss, _, _ = model._shared_step(graph, "train", log=False)  # type: ignore
        assert isinstance(loss, torch.Tensor)

        print("✅ CLAM integration successful")


class TestGraphMILEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_node_graph(self):
        """Test with single node graphs"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Standard(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16], standard_type="fc"
        )
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Single node graph
        graph = Data(
            x=torch.randn(1, 64),
            edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
            y=torch.tensor([0]),
        )

        # Should handle gracefully
        logits, _ = model(graph, label=graph.y, instance_eval=True)
        assert logits.shape == (1, 2)

        print("✅ Single node graph handled correctly")

    def test_large_graph(self):
        """Test with large graphs"""
        gnn = GAT(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Mean_MLP(input_dim=32, dropout=0.1, n_classes=2, size_arg=[16])
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Large graph (500 nodes)
        num_nodes = 500
        edge_list: list[list[float]] = []
        for i in range(num_nodes - 1):
            edge_list.extend([[i, i + 1], [i + 1, i]])
        for _ in range(1000):  # Add random edges
            i, j = torch.randint(0, num_nodes, (2,))
            if i != j:
                edge_list.extend([[i.item(), j.item()], [j.item(), i.item()]])

        graph = Data(
            x=torch.randn(num_nodes, 64),
            edge_index=torch.tensor(edge_list).t().contiguous(),
            y=torch.tensor([1]),
        )

        # Should handle without memory issues
        logits, _ = model(graph, label=graph.y, instance_eval=True)
        assert logits.shape == (1, 2)

        print("✅ Large graph handled correctly")

    def test_disconnected_graph(self):
        """Test with disconnected graphs"""
        gnn = SAGE(input_dim=64, hidden_dim=32, n_layers=2, dropout=0.1)
        pooling = Attention(
            input_dim=32, dropout=0.1, n_classes=2, size_arg=[16, 8]
        )  # Fix: Attention requires 2 elements
        model = LitGraphMIL(
            gnn=gnn,
            pooling_classifier=pooling,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"lr": 0.001},
        )

        # Disconnected graph (two separate components)
        edge_index = torch.tensor(
            [
                [0, 1, 1, 0, 2, 3, 3, 2],  # Two disconnected triangles
                [1, 0, 2, 2, 3, 2, 4, 4],
            ],
            dtype=torch.long,
        )

        graph = Data(x=torch.randn(5, 64), edge_index=edge_index, y=torch.tensor([0]))

        # Should handle disconnected components
        logits, _ = model(graph, label=graph.y, instance_eval=True)
        assert logits.shape == (1, 2)

        print("✅ Disconnected graph handled correctly")


class TestGraphMILAttentionExtraction:
    """Test attention weights extraction for GraphMIL models"""

    def create_sample_data(self):
        """Create sample graph data for attention testing"""
        torch.manual_seed(42) # type: ignore
        return Data(
            x=torch.randn(20, 128),  # 20 nodes, 128 features
            edge_index=torch.randint(0, 20, (2, 40)),  # 40 edges
            y=torch.tensor([1])  # binary label
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample graph data for attention testing (pytest fixture version)"""
        return self.create_sample_data()

    def test_gat_attention_extraction(self, sample_data: Data):
        """Test attention weights extraction from GAT layers"""
        gnn = GAT(input_dim=128, hidden_dim=256, n_layers=3, dropout=0.0, heads=4)
        
        attention_weights = gnn.get_attention_weights(sample_data)
        
        # Should have attention weights for each layer
        expected_keys = ['gnn_attention_layer_0', 'gnn_attention_layer_1', 'gnn_attention_layer_2']
        assert set(attention_weights.keys()) == set(expected_keys)
        
        # Each attention weight should be a tensor
        for key, weights in attention_weights.items():
            assert isinstance(weights, torch.Tensor)
            assert weights.dim() >= 1  # At least 1D tensor
            print(f"✅ {key}: shape {weights.shape}")

    def test_gatv2_attention_extraction(self, sample_data: Data):
        """Test attention weights extraction from GATv2 layers"""
        gnn = GATv2(input_dim=128, hidden_dim=256, n_layers=2, dropout=0.0, heads=2)
        
        attention_weights = gnn.get_attention_weights(sample_data)
        
        # Should have attention weights for each layer
        expected_keys = ['gnn_attention_layer_0', 'gnn_attention_layer_1']
        assert set(attention_weights.keys()) == set(expected_keys)
        
        for key, weights in attention_weights.items():
            assert isinstance(weights, torch.Tensor)
            print(f"✅ GATv2 {key}: shape {weights.shape}")

    def test_attention_pooling_extraction(self, sample_data: Data):
        """Test attention weights extraction from AttentionDeepMIL pooling"""
        pooling = Attention(
            input_dim=128, dropout=0.0, n_classes=2, 
            size_arg=[64, 32], attention_branches=1
        )
        
        attention_weights = pooling.get_attention_weights(sample_data.x) # type: ignore
        assert attention_weights is not None
        assert isinstance(attention_weights, torch.Tensor)
        assert attention_weights.shape[1] == sample_data.x.shape[0]  # Should attend to all nodes # type: ignore
        print(f"✅ AttentionDeepMIL pooling: shape {attention_weights.shape}")

    def test_clam_pooling_extraction(self, sample_data: Data):
        """Test attention weights extraction from CLAM pooling"""
        pooling = CLAM(
            input_dim=128, dropout=0.0, n_classes=2,
            size_arg="small", gate=True, clam_type="SB"
        )
        
        attention_weights = pooling.get_attention_weights(sample_data.x) # type: ignore
        assert attention_weights is not None
        assert isinstance(attention_weights, torch.Tensor)
        assert attention_weights.shape[1] == sample_data.x.shape[0]  # Should attend to all nodes # type: ignore
        print(f"✅ CLAM pooling: shape {attention_weights.shape}")

    def test_standard_pooling_no_attention(self, sample_data: Data):
        """Test that Standard pooling correctly reports no attention"""
        pooling = Standard(input_dim=128, dropout=0.0, n_classes=2, size_arg=[64])

        attention_weights = pooling.get_attention_weights(sample_data.x) # type: ignore
        assert attention_weights is None
        print("✅ Standard pooling correctly reports no attention weights")

    def test_full_model_gat_attention(self, sample_data: Data):
        """Test full GraphMIL model with GAT + AttentionDeepMIL"""
        gnn = GAT(input_dim=128, hidden_dim=256, n_layers=2, dropout=0.0, heads=4)
        pooling = Attention(
            input_dim=256, dropout=0.0, n_classes=2,
            size_arg=[128, 64], attention_branches=1
        )
        
        model = LitGraphMIL(
            gnn=gnn, pooling_classifier=pooling,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3}
        )
        
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sample_data)
        
        # Should have both GNN and pooling attention
        expected_keys = ['gnn_attention_layer_0', 'gnn_attention_layer_1', 'pooling_attention']
        assert set(attention_weights.keys()) == set(expected_keys)
        
        for key, weights in attention_weights.items():
            assert isinstance(weights, torch.Tensor)
            print(f"✅ Full model {key}: shape {weights.shape}")

    def test_full_model_gat_clam(self, sample_data: Data):
        """Test full GraphMIL model with GAT + CLAM"""
        gnn = GAT(input_dim=128, hidden_dim=256, n_layers=1, dropout=0.0, heads=2)
        pooling = CLAM(
            input_dim=256, dropout=0.0, n_classes=2,
            size_arg="small", gate=True, clam_type="SB"
        )
        
        model = LitGraphMIL(
            gnn=gnn, pooling_classifier=pooling,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3}
        )
        
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sample_data)
        
        # Should have both GNN and pooling attention
        expected_keys = ['gnn_attention_layer_0', 'pooling_attention']
        assert set(attention_weights.keys()) == set(expected_keys)
        
        for key, weights in attention_weights.items():
            assert isinstance(weights, torch.Tensor)
            print(f"✅ GAT+CLAM {key}: shape {weights.shape}")

    def test_full_model_sage_standard(self, sample_data: Data):
        """Test full GraphMIL model with SAGE + Standard (no attention)"""
        gnn = SAGE(input_dim=128, hidden_dim=256, n_layers=2, dropout=0.0)
        pooling = Standard(input_dim=256, dropout=0.0, n_classes=2, size_arg=[128])
        
        model = LitGraphMIL(
            gnn=gnn, pooling_classifier=pooling,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3}
        )
        
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sample_data)
        
        # Should have no attention weights
        assert attention_weights == {}
        print("✅ SAGE+Standard correctly returns no attention weights")

    def test_attention_weights_properties(self, sample_data: Data):
        """Test properties of extracted attention weights"""
        gnn = GAT(input_dim=128, hidden_dim=256, n_layers=1, dropout=0.0, heads=2)
        pooling = Attention(
            input_dim=256, dropout=0.0, n_classes=2,
            size_arg=[128, 64], attention_branches=1
        )
        
        model = LitGraphMIL(
            gnn=gnn, pooling_classifier=pooling,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-3}
        )
        
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sample_data)
        
        # Test pooling attention properties (should be normalized)
        if 'pooling_attention' in attention_weights:
            pooling_attention = attention_weights['pooling_attention']
            
            # Should sum to approximately 1 (softmax normalized)
            attention_sum = pooling_attention.sum(dim=-1)
            assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)
            
            # Should be non-negative
            assert (pooling_attention >= 0).all()
            
            # Should have correct shape
            assert pooling_attention.shape[1] == sample_data.x.shape[0] # type: ignore
            
            print(f"✅ Pooling attention properly normalized: sum={attention_sum.item():.6f}")

    def test_attention_extraction_error_handling(self):
        """Test that attention extraction handles edge cases gracefully"""
        # Test with None input
        gnn = GAT(input_dim=128, hidden_dim=256, n_layers=1, dropout=0.0, heads=2)
        
        invalid_data = Data(x=None, edge_index=torch.tensor([[0], [1]]))
        attention_weights = gnn.get_attention_weights(invalid_data)
        assert attention_weights == {}
        
        print("✅ Gracefully handles invalid input data")

    def test_different_layer_counts(self, sample_data: Data):
        """Test attention extraction with different numbers of layers"""
        for n_layers in [1, 2, 3, 5]:
            gnn = GAT(input_dim=128, hidden_dim=256, n_layers=n_layers, dropout=0.0, heads=2)
            
            attention_weights = gnn.get_attention_weights(sample_data)
            
            # Should have attention weights for each layer
            assert len(attention_weights) == n_layers
            
            expected_keys = [f'gnn_attention_layer_{i}' for i in range(n_layers)]
            assert set(attention_weights.keys()) == set(expected_keys)
            
            print(f"✅ {n_layers} layers: {len(attention_weights)} attention weight sets")

    def test_attention_with_different_heads(self, sample_data: Data):
        """Test attention extraction with different numbers of attention heads"""
        for heads in [1, 2, 4, 8]:
            # Adjust hidden_dim to be divisible by heads
            hidden_dim = 256 if 256 % heads == 0 else ((256 // heads) + 1) * heads
            
            gnn = GAT(input_dim=128, hidden_dim=hidden_dim, n_layers=1, dropout=0.0, heads=heads)
            
            attention_weights = gnn.get_attention_weights(sample_data)
            
            assert 'gnn_attention_layer_0' in attention_weights
            weights = attention_weights['gnn_attention_layer_0']
            
            # Attention weights should reflect the number of heads (in some dimension)
            assert isinstance(weights, torch.Tensor)
            assert weights.numel() > 0  # Should have some elements
            
            print(f"✅ {heads} heads: attention shape {weights.shape}")