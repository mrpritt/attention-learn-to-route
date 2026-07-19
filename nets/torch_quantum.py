import torch
from torch import nn


def parse_rot_sequence(rotation):
    cleaned = rotation.upper().replace(" ", "")
    if len(cleaned) == 0 or len(cleaned) % 2 != 0:
        raise ValueError(f"Invalid rotation sequence '{rotation}'")
    tokens = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]
    allowed = {"RX", "RY", "RZ"}
    if any(token not in allowed for token in tokens):
        raise ValueError(f"rotation tokens must be in {allowed}, got {tokens}")
    return tokens


class TorchStatevectorQNN(nn.Module):
    """Small state-vector QNN implemented directly in PyTorch.

    Wire ordering follows PennyLane's default basis order for wires 0..q-1:
    wire 0 is the most significant bit in the flattened state index.
    """

    def __init__(self, n_qubits=4, n_layers=2, rotation="RXRYRZ", topology="brickwall"):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation = rotation
        self.topology = topology.lower().replace("-", "").replace("_", "")
        self.rot_seq = parse_rot_sequence(rotation)
        self.theta = nn.Parameter(torch.zeros(n_layers, n_qubits, len(self.rot_seq)))

        if self.topology not in {"brickwall", "chain", "lambda"}:
            raise ValueError("topology must be 'brickwall', 'chain', or 'lambda'")

        for control, target in self._cnot_pairs():
            self.register_buffer(
                f"_cnot_perm_{control}_{target}",
                self._make_cnot_perm(control, target),
                persistent=False,
            )
        for wire in range(n_qubits):
            self.register_buffer(f"_z_signs_{wire}", self._make_z_signs(wire), persistent=False)

    @property
    def state_dim(self):
        return 1 << self.n_qubits

    def _cnot_pairs(self):
        pairs = []
        if self.topology == "brickwall":
            pairs.extend((left, left + 1) for left in range(0, self.n_qubits - 1, 2))
            pairs.extend((left, left + 1) for left in range(1, self.n_qubits - 1, 2))
        elif self.topology == "chain":
            pairs.extend((left, left + 1) for left in range(self.n_qubits - 1))
        elif self.topology == "lambda":
            for delta in range(self.n_qubits - 1):
                left = delta
                right = self.n_qubits - 2 - delta
                if left <= right:
                    pairs.append((left, left + 1))
                if right > left:
                    pairs.append((right, right + 1))
                if left >= right:
                    break
        return pairs

    def _wire_mask(self, wire):
        return 1 << (self.n_qubits - 1 - wire)

    def _make_cnot_perm(self, control, target):
        idx = torch.arange(self.state_dim, dtype=torch.long)
        control_on = (idx & self._wire_mask(control)) != 0
        return torch.where(control_on, idx ^ self._wire_mask(target), idx)

    def _make_z_signs(self, wire):
        idx = torch.arange(self.state_dim)
        return torch.where((idx & self._wire_mask(wire)) == 0, 1.0, -1.0)

    def _initial_state(self, batch_size, device, real_dtype):
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        psi = torch.zeros(batch_size, self.state_dim, device=device, dtype=complex_dtype)
        psi[:, 0] = 1.0
        return psi

    def _apply_1q(self, psi, wire, a, b, c, d):
        batch_size = psi.size(0)
        shape = [batch_size] + [2] * self.n_qubits
        target_axis = wire + 1
        perm = [0] + [axis for axis in range(1, self.n_qubits + 1) if axis != target_axis] + [target_axis]
        inv_perm = [0] * len(perm)
        for i, axis in enumerate(perm):
            inv_perm[axis] = i

        paired = psi.reshape(shape).permute(perm).reshape(batch_size, -1, 2)
        v0 = paired[..., 0]
        v1 = paired[..., 1]

        def prep(x):
            x = x.to(dtype=psi.dtype, device=psi.device)
            return x[:, None] if x.dim() > 0 else x

        a, b, c, d = prep(a), prep(b), prep(c), prep(d)
        out = torch.empty_like(paired)
        out[..., 0] = a * v0 + b * v1
        out[..., 1] = c * v0 + d * v1
        return out.reshape([batch_size] + [2] * self.n_qubits).permute(inv_perm).reshape(batch_size, self.state_dim)

    def _apply_rotation(self, psi, wire, gate, angle):
        half = angle / 2
        cos = torch.cos(half)
        sin = torch.sin(half)
        zeros = torch.zeros_like(cos)
        if gate == "RX":
            minus_i_sin = torch.complex(zeros, -sin)
            return self._apply_1q(psi, wire, cos, minus_i_sin, minus_i_sin, cos)
        if gate == "RY":
            return self._apply_1q(psi, wire, cos, -sin, sin, cos)
        if gate == "RZ":
            return self._apply_1q(
                psi,
                wire,
                torch.exp(torch.complex(zeros, -half)),
                zeros,
                zeros,
                torch.exp(torch.complex(zeros, half)),
            )
        raise ValueError(f"Unsupported gate '{gate}'")

    def _apply_cnot(self, psi, control, target):
        perm = getattr(self, f"_cnot_perm_{control}_{target}").to(device=psi.device)
        return psi.index_select(1, perm)

    def forward(self, inputs):
        if inputs.size(-1) != self.n_qubits:
            raise ValueError(f"Expected last dimension {self.n_qubits}, got {inputs.size(-1)}")
        original_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, self.n_qubits)
        psi = self._initial_state(flat_inputs.size(0), flat_inputs.device, flat_inputs.dtype)

        for wire in range(self.n_qubits):
            psi = self._apply_rotation(psi, wire, "RY", flat_inputs[:, wire])

        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                for rot_idx, gate in enumerate(self.rot_seq):
                    psi = self._apply_rotation(psi, wire, gate, self.theta[layer, wire, rot_idx])
            for control, target in self._cnot_pairs():
                psi = self._apply_cnot(psi, control, target)

        probs = psi.abs().pow(2).real
        expvals = []
        for wire in range(self.n_qubits):
            signs = getattr(self, f"_z_signs_{wire}").to(device=inputs.device, dtype=inputs.dtype)
            expvals.append((probs * signs).sum(dim=-1))
        return torch.stack(expvals, dim=-1).reshape(*original_shape, self.n_qubits)


class HybridTorchQuantumLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_qubits=4,
        n_layers=2,
        rotation="RXRYRZ",
        topology="brickwall",
        ansatz_name="pce",
        device_name="auto",
        diff_method="auto",
        bias=False,
    ):
        super().__init__()
        if ansatz_name != "pce":
            raise ValueError(f"Torch QNN backend only supports ansatz 'pce', got '{ansatz_name}'")
        self.n_qubits = n_qubits
        self.input_proj = nn.Linear(input_dim, n_qubits, bias=bias)
        self.q_layer = TorchStatevectorQNN(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            topology=topology,
        )
        self.output_proj = nn.Linear(n_qubits, output_dim, bias=bias)
        # Kept for args/checkpoint readability; intentionally unused by this backend.
        self.device_name = device_name
        self.diff_method = diff_method

    @property
    def theta(self):
        return self.q_layer.theta

    def materialize(self):
        pass

    def forward(self, x):
        encoded = torch.tanh(self.input_proj(x)) * torch.pi
        return self.output_proj(self.q_layer(encoded).to(dtype=x.dtype, device=x.device))
