"""ReMoE routing for iteration-aware Block-ELL dispatch."""

from .remoe_router import (
    ReMoERouter,
    compute_l1_loss,
    update_lambda,
    init_lambda,
)

from .routed_mlp import (
    RoutedMLP,
    RoutedMLPWithGates,
)

__all__ = [
    # remoe_router.py
    "ReMoERouter",
    "compute_l1_loss",
    "update_lambda",
    "init_lambda",
    # routed_mlp.py
    "RoutedMLP",
    "RoutedMLPWithGates",
]
