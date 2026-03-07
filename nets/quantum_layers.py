from torch import nn
import torch


def build_qnn_config(opts):
    return {
        "ansatz_name": opts.qnn_ansatz,
        "n_qubits": opts.qnn_qubits,
        "n_layers": opts.qnn_layers,
        "rotation": opts.qnn_rotation,
        "topology": opts.qnn_topology,
    }


def _parse_rot_sequence(rotation):
    cleaned = rotation.upper().replace(" ", "")
    if len(cleaned) == 0 or len(cleaned) % 2 != 0:
        raise ValueError(f"Invalid rotation sequence '{rotation}'")
    return [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]


def _load_ansatz(ansatz_name):
    if ansatz_name == "pce":
        from ansatz.pce import ansatz

        return ansatz
    raise ValueError(f"Unsupported ansatz '{ansatz_name}'")


class HybridQuantumLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_qubits=8,
        n_layers=4,
        rotation="RXRYRZ",
        topology="brickwall",
        ansatz_name="pce",
        bias=False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation = rotation
        self.topology = topology
        self.ansatz_name = ansatz_name
        self._weight_shape = (n_layers, n_qubits, len(_parse_rot_sequence(rotation)))

        self.input_proj = nn.Linear(input_dim, n_qubits, bias=bias)
        self.output_proj = nn.Linear(n_qubits, output_dim, bias=bias)

        self.qml = None
        self.qnode = None
        self.q_layer = None
        self.ansatz = None

    def _build_layer(self):
        if self.q_layer is not None:
            return

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("PennyLane is required when project_fixed_context_backend='qnn'") from exc

        self.qml = qml
        self.ansatz = _load_ansatz(self.ansatz_name)
        diff_method = "backprop"
        try:
            device = qml.device("lightning.qubit", wires=self.n_qubits)
            diff_method = "adjoint"
        except Exception:
            device = qml.device("default.qubit", wires=self.n_qubits)

        qnode = qml.QNode(self._circuit, device, interface="torch", diff_method=diff_method)
        self.q_layer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"theta": self._weight_shape},
            init_method={"theta": torch.nn.init.zeros_},
        )

    def _circuit(self, inputs, theta):
        for wire in range(self.n_qubits):
            self.qml.RY(inputs[..., wire], wires=wire)

        self.ansatz(
            theta,
            R_type=self.rotation,
            n_qubits=self.n_qubits,
            topology=self.topology,
        )
        return [self.qml.expval(self.qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, x):
        self._build_layer()
        encoded = torch.tanh(self.input_proj(x)) * torch.pi
        return self.output_proj(self.q_layer(encoded).to(dtype=x.dtype, device=x.device))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["qml"] = None
        state["qnode"] = None
        state["q_layer"] = None
        state["ansatz"] = None
        return state


class SwitchableLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, backend="classical", qnn_config=None):
        super().__init__()
        qnn_config = qnn_config or {}
        if backend == "classical":
            self.layer = nn.Linear(input_dim, output_dim, bias=bias)
        elif backend == "qnn":
            self.layer = HybridQuantumLinear(
                input_dim=input_dim,
                output_dim=output_dim,
                bias=bias,
                **qnn_config,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'")

    def forward(self, x):
        return self.layer(x)
