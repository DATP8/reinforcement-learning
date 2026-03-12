import torch
from qiskit.circuit.quantumcircuit import QuantumCircuit
class Qtensor():
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
        if torch.overrides.resolve_name(func) == "torch.Tensor.__setitem__":
            return
        if kwargs is None:
            kwargs = {}
        gates = tuple(a.gates for a in args if hasattr(a, 'gates'))
        args_new = [getattr(a, '_t', a) for a in args]
        assert len(gates) > 0
        ret = func(*args_new, **kwargs)
        return Qtensor(ret)
    
    def __getitem__(self, key):
        return Qtensor(self._t[key], self.gates)
    
    def __setitem__(self, key, value):
        self._t[key] = value
        
    def __eq__(self, value):
        return self._t == value
    
    def __mul__(self, other):
        return Qtensor(self._t * other, self.gates)
    
    
    
    def from_circuit(circuit: QuantumCircuit, horizon: int):
        c = torch.zeros((circuit.num_qubits, horizon))
        i = 0
        for gate in circuit.data:
            if gate.operation.num_qubits == 2:
                for qubit in gate.qubits:
                    c[qubit._index, i] = 1
                i += 1
        return Qtensor(c, len(circuit.data))
    
    def to(self, device: torch.device):
        return self._t.to(device)
    
    def unsqueeze(self, **args):
        return Qtensor(self._t.unsqueeze(**args), self.gates)
    
    def size(self, **args):
        return self._t.size(**args)
    
    def dim(self, **args):
        return self._t.dim(**args)
    
    def clone(self, **args):
        return Qtensor(self._t.clone(**args), self.gates)
    
    def cat(self, other, **args):
        return Qtensor(torch.cat((self._t, other), **args), self.gates)
    
    def item(self, **args):
        return self._t.item(**args)
    
    def unwrap(self):
        return self._t


if __name__ == "__main__":
    c = QuantumCircuit(4)
    c.cx(0,1)
    tensor = Qtensor.from_circuit(c, 10)
    print(tensor)
    print(tensor[1])