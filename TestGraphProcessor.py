import unittest
import networkx as nx
from GraphProcessor import GraphProcessor

class TestGraphProcessor(unittest.TestCase):



    def test_cyclic_to_dag_multiple_inputs(self):


        # Test Case 1: Two separate input nodes in a cyclic graph
        G1 = nx.DiGraph([
            (1, 2), (2, 3), (3, 1),  # Cycle 1
            (4, 5), (5, 6), (6, 4),  # Cycle 2
            (3, 5)  # Cross connection
        ])
        processor1 = GraphProcessor(G1)  # Instantiate with the graph
        input_nodes = {1, 4}
        expected_edges = [(1, 2), (2, 3), (3, 5), (4, 5), (5, 6)]  # Cycles (3,1) and (6,4) removed
        dag1 = processor1.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag1), "Graph should be a DAG")
        self.assertEqual(sorted(dag1.edges()), sorted(expected_edges), "Edges do not match expected DAG")


        # Topological sort validation
        layers, node_to_layer = processor1.topological_sort_with_layers()

        # Validate all nodes are present in node_to_layer and layers
        nodes_in_layers = {node for layer in layers for node in layer}
        self.assertEqual(nodes_in_layers, set(dag1.nodes()), "Nodes in layers do not match graph nodes")

        # Validate topological order
        for u, v in dag1.edges():
            self.assertLess(node_to_layer[u], node_to_layer[v], f"Edge ({u},{v}) violates topological order.")

        self.assertEqual([[1, 4], [2], [3], [5], [6]], [sorted(ar) for ar in layers], "wrong layer groupings")

        # Test Case 2: Multiple input nodes with shared successors
        G2 = nx.DiGraph([
            (1, 3), (2, 3), (3, 4), (4, 5),
            (5, 2)  # Cycle back to 2
        ])
        processor2 = GraphProcessor(G2)
        input_nodes = {1, 2}
        expected_edges = [(1, 3), (2, 3), (3, 4), (4, 5)]  # (5,2) should be removed
        dag2 = processor2.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag2), "Graph should be a DAG")
        self.assertEqual(sorted(dag2.edges()), sorted(expected_edges), "Edges do not match expected DAG")

        # Test Case 3: Multiple input nodes with isolated components
        G3 = nx.DiGraph([
            (1, 2), (2, 3),
            (4, 5), (5, 6),
            (6, 4)  # Cycle in second component
        ])
        processor3 = GraphProcessor(G3)
        input_nodes = {1, 4}
        expected_edges = [(1, 2), (2, 3), (4, 5), (5, 6)]  # Cycle (6,4) removed
        dag3 = processor3.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag3), "Graph should be a DAG")
        self.assertEqual(sorted(dag3.edges()), sorted(expected_edges), "Edges do not match expected DAG")

        # Test Case 4: Multiple input nodes where one is unreachable
        G4 = nx.DiGraph([
            (1, 2), (2, 3),
            (4, 5)  # Unconnected to (1,2,3)
        ])
        processor4 = GraphProcessor(G4)
        input_nodes = {1, 6}  # Node 6 does not exist
        expected_edges = [(1, 2), (2, 3)]  # (4,5) is untouched
        dag4 = processor4.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag4), "Graph should be a DAG")
        self.assertEqual(sorted(dag4.edges()), sorted(expected_edges), "Edges do not match expected DAG")


        # Test Case 5:
        G5 = nx.DiGraph([
            (1, 2), (2, 3),(1,4),(4,3),
            (4, 5),(5,1)
        ])
        processor5 = GraphProcessor(G5)
        input_nodes = {1, 2}  # Node 6 does not exist
        expected_edges = [(1, 4), (2, 3),(4,3),(4,5)]
        dag5 = processor5.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag5), "Graph should be a DAG")
        self.assertEqual(sorted(dag5.edges()), sorted(expected_edges), "Edges do not match expected DAG")



        # Test Case 5:
        G6 = nx.DiGraph([
            (1, 2), (2, 3),(1,4),(4,3),(3,1),
            (4, 5),(5,1),(1,6)
        ])
        processor6 = GraphProcessor(G6)
        input_nodes = {1, 2}  # Node 6 does not exist
        expected_edges = [(1, 4), (2, 3),(4,3),(4,5),(1,6)]
        dag6 = processor6.cyclic_to_dag(input_nodes)
        self.assertTrue(nx.is_directed_acyclic_graph(dag6), "Graph should be a DAG")
        self.assertEqual(sorted(dag6.edges()), sorted(expected_edges), "Edges do not match expected DAG")

        # Topological sort validation
        layers, node_to_layer = processor6.topological_sort_with_layers()

        # Validate all nodes are present in node_to_layer and layers
        nodes_in_layers = {node for layer in layers for node in layer}
        self.assertEqual(nodes_in_layers, set(dag6.nodes()), "Nodes in layers do not match graph nodes")

        # Validate topological order
        for u, v in dag6.edges():
            self.assertLess(node_to_layer[u], node_to_layer[v], f"Edge ({u},{v}) violates topological order.")

        self.assertEqual([[1, 2], [4], [3,5,6]], [sorted(ar) for ar in layers], "wrong layer groupings")

if __name__ == '__main__':
    # Suppress the SystemExit by specifying exit=False
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
