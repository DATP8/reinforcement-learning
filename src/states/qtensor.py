import hashlib

import torch
from qiskit.circuit.quantumcircuit import QuantumCircuit

accepted_mult_funcs = ["torch.cat", "torch.stack"]


class Qtensor:
    def __init__(self, data, gates=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        if gates is None:
            if self._t.dim() > 1:
                self.gates = self._t.size()[1]
            else:
                self.gates = 0
        else:
            self.gates = gates

    def __repr__(self):
        return "gates: {}\ndata:\n{}".format(self.gates, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        func_name = torch.overrides.resolve_name(func)
        match func_name:
            case "torch.Tensor.__setitem__":
                return
            case "torch.equal":
                return func(*[getattr(a, "_t", a) for a in args])
        if func_name in accepted_mult_funcs:
            args_new = []
            # pyrefly: ignore[bad-index]
            gates = tuple(a.gates for a in args[0] if hasattr(a, "gates"))
            # pyrefly: ignore[bad-index]
            args_new.append(([getattr(a, "_t", a) for a in args[0]]))
            for i in range(1, len(args)):
                args_new.append(args[i])
            args = tuple(args_new)
        else:
            gates = tuple(a.gates for a in args if hasattr(a, "gates"))
            args = [getattr(a, "_t", a) for a in args]
        assert len(gates) > 0
        ret = func(*args, **kwargs)
        return Qtensor(ret)

    def __getitem__(self, key):
        return Qtensor(self._t[key], gates=self.gates)

    def __setitem__(self, key, value):
        if isinstance(value, Qtensor):
            self._t[key] = value._t
        else:
            self._t[key] = value

    def __mul__(self, other):
        return Qtensor(self._t * other, gates=self.gates)

    def __hash__(self):
        return self.tensor_hash(self._t)

    @staticmethod
    def tensor_hash(t: torch.Tensor) -> int:
        return hash(hashlib.blake2b(t.numpy().tobytes(), digest_size=8).digest())

    @classmethod
    def from_circuit(cls, qc: QuantumCircuit, horizon: int = 0):
        c = torch.zeros((qc.num_qubits, horizon))
        i = 0
        for gate in qc.data:
            if gate.operation.num_qubits == 2:
                for qubit in gate.qubits:
                    c[qubit._index, i] = 1
                i += 1
        return cls(c, gates=len(qc.data))

    def to(self, device: torch.device):
        return self._t.to(device)

    def unsqueeze(self, **args):
        return Qtensor(self._t.unsqueeze(**args), gates=self.gates)

    def size(self, **args):
        return self._t.size(**args)

    def dim(self, **args):
        return self._t.dim(**args)

    def clone(self, **args):
        return Qtensor(self._t.clone(**args), gates=self.gates)

    def item(self, **args):
        return self._t.item(**args)

    def unwrap(self):
        return self._t

    # !THIS IS NOT IMPLEMENTED
    def to_circuit(self):
        raise NotImplementedError


if __name__ == "__main__":
    c = QuantumCircuit(4)
    c.cx(0, 1)
    tensor = Qtensor.from_circuit(c, 10)
    # torch.cat((tensor, torch.as_tensor([0,1])), dim=1)
    # print(tensor)
    # print(tensor[1])
    print(dict(dim=1))
    print({"dim": 1})
