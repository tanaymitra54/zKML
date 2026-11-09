"""ZK workload harness for the DP-zkML Elsevier revision (Paper A, experiments).

Modules
-------
workloads : model definitions, public data loaders, ONNX export
runner    : one measured configuration (compile -> setup -> prove -> verify)
scaling   : resumable sweep driver writing results/zk_scaling.csv
utility   : float vs quantized accuracy on held-out data -> results/zk_utility.csv
"""

__all__ = ["workloads", "runner", "scaling", "utility"]
