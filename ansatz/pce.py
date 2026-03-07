import pennylane as qml


def parse_rot_sequence(rotation):
    cleaned = rotation.upper().replace(" ", "")
    if len(cleaned) == 0 or len(cleaned) % 2 != 0:
        raise ValueError("rotation must be a non-empty concatenation like 'RY' or 'RXRYRZ'")

    tokens = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]
    allowed = {"RX", "RY", "RZ"}
    if any(token not in allowed for token in tokens):
        raise ValueError(f"rotation tokens must be in {allowed}, got {tokens}")
    return tokens


def apply_rot_sequence(params_1q, wire, rot_seq):
    for theta, gate in zip(params_1q, rot_seq):
        if gate == "RX":
            qml.RX(theta, wires=wire)
        elif gate == "RY":
            qml.RY(theta, wires=wire)
        else:
            qml.RZ(theta, wires=wire)


def ansatz(params, R_type="RXRYRZ", n_qubits=8, topology="brickwall"):
    rot_seq = parse_rot_sequence(R_type)
    topology = topology.lower().replace("-", "").replace("_", "")

    for layer in params:
        for wire in range(n_qubits):
            apply_rot_sequence(layer[wire], wire=wire, rot_seq=rot_seq)

        if topology == "brickwall":
            for left in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[left, left + 1])
            for left in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[left, left + 1])
        elif topology == "chain":
            for left in range(n_qubits - 1):
                qml.CNOT(wires=[left, left + 1])
        elif topology == "lambda":
            for delta in range(n_qubits - 1):
                left = delta
                right = n_qubits - 2 - delta
                if left <= right:
                    qml.CNOT(wires=[left, left + 1])
                if right > left:
                    qml.CNOT(wires=[right, right + 1])
                if left >= right:
                    break
        else:
            raise ValueError("topology must be 'brickwall', 'chain', or 'lambda'")
