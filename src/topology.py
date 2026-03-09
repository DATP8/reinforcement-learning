import networkx as nx
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

class Topology:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def draw_topology(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title("Quantum Hardware Topology")
        plt.show()

    def add_ibm_fez_topology(self):
        service = QiskitRuntimeService()
        backend = service.backend('ibm_fez')
        coupling_map = backend.coupling_map
        self.graph.add_edges_from(coupling_map.get_edges())
    
    def add_ibm_torino_topology(self):
        service = QiskitRuntimeService()
        backend = service.backend('ibm_torino')
        coupling_map = backend.coupling_map
        self.graph.add_edges_from(coupling_map.get_edges())

    def add_ibm_marrakesh_topology(self):
        service = QiskitRuntimeService()
        backend = service.backend('ibm_marrakesh')
        coupling_map = backend.coupling_map
        self.graph.add_edges_from(coupling_map.get_edges())

    def edges(self):
        return list(self.graph.edges())

    def nodes(self):
        return list(self.graph.nodes())

def get_topology_from_file(file_path: str) -> Topology:
    graph = nx.read_edgelist(file_path, nodetype=int)
    return Topology(graph)

if __name__ == "__main__":
    #topology = Topology(nx.Graph())
    #topology.add_ibm_marrakesh_topology()
    #with open('./src/topologies/marrakesh_topology.txt', 'w') as f: 
    #    for edge in topology.edges():
    #        f.write(f"{edge[0]} {edge[1]}\n")
    top_from_file = get_topology_from_file('./src/topologies/torino_topology.txt')
    print(top_from_file.edges())