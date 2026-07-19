#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nets.torch_quantum import TorchStatevectorQNN, HybridTorchQuantumLinear
from ansatz.pce import ansatz


def pennylane_reference(inputs, theta, rotation, topology):
    import pennylane as qml

    n_qubits = inputs.size(-1)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        for wire in range(n_qubits):
            qml.RY(x[wire], wires=wire)
        ansatz(weights, R_type=rotation, n_qubits=n_qubits, topology=topology)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    return torch.stack([torch.stack(circuit(row, theta)) for row in inputs]).to(dtype=inputs.dtype)


def assert_close(name, actual, expected, atol, rtol):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff {diff}")


def parity_case(n_qubits, n_layers, rotation, topology, batch, dtype, device, atol, rtol):
    torch.manual_seed(1234 + n_qubits + n_layers)
    inputs = torch.randn(batch, n_qubits, dtype=dtype, device=device, requires_grad=True)
    theta = 0.2 * torch.randn(
        n_layers,
        n_qubits,
        len([rotation[i:i + 2] for i in range(0, len(rotation), 2)]),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    model = TorchStatevectorQNN(n_qubits=n_qubits, n_layers=n_layers, rotation=rotation, topology=topology).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        model.theta.copy_(theta)

    torch_out = model(inputs)
    pl_out = pennylane_reference(inputs.detach().cpu().requires_grad_(True), theta.detach().cpu().requires_grad_(True), rotation, topology).to(device)
    assert_close(f"forward q={n_qubits} topo={topology}", torch_out, pl_out, atol, rtol)

    torch_loss = torch_out.pow(2).sum()
    torch_input_grad, torch_theta_grad = torch.autograd.grad(torch_loss, (inputs, model.theta))

    pl_inputs = inputs.detach().cpu().requires_grad_(True)
    pl_theta = theta.detach().cpu().requires_grad_(True)
    pl_loss = pennylane_reference(pl_inputs, pl_theta, rotation, topology).pow(2).sum()
    pl_input_grad, pl_theta_grad = torch.autograd.grad(pl_loss, (pl_inputs, pl_theta))

    assert_close("input gradient", torch_input_grad.cpu(), pl_input_grad, atol, rtol)
    assert_close("theta gradient", torch_theta_grad.cpu(), pl_theta_grad, atol, rtol)


def hybrid_backward_case(device):
    torch.manual_seed(4321)
    layer = HybridTorchQuantumLinear(128, 128, n_qubits=4, n_layers=2, rotation="RXRYRZ", topology="brickwall").to(device)
    x = torch.randn(8, 21, 128, device=device, requires_grad=True)
    y = layer(x)
    if y.shape != x.shape:
        raise AssertionError(f"Hybrid layer shape mismatch: got {tuple(y.shape)}, expected {tuple(x.shape)}")
    loss = y.square().mean()
    loss.backward()
    if x.grad is None or layer.theta.grad is None:
        raise AssertionError("Hybrid layer backward did not populate gradients")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    atol = 1e-9 if dtype == torch.float64 else 2e-5
    rtol = 1e-9 if dtype == torch.float64 else 2e-5

    for topology in ["brickwall", "chain", "lambda"]:
        parity_case(3, 2, "RXRYRZ", topology, batch=5, dtype=dtype, device=device, atol=atol, rtol=rtol)
    parity_case(4, 1, "RY", "brickwall", batch=4, dtype=dtype, device=device, atol=atol, rtol=rtol)
    hybrid_backward_case(device)
    print(f"torch QNN smoke tests passed on {device} with {args.dtype}")


if __name__ == "__main__":
    main()
