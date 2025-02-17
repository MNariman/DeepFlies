from collections import defaultdict, deque
import networkx as nx
import numpy as np
import random

class GraphProcessor:
    def __init__(self, graph):
        """
        Initialize the class with a directed graph.

        :param graph: A NetworkX DiGraph instance.
        :type graph: nx.DiGraph
        """
        assert isinstance(graph, nx.DiGraph), "Input graph must be a NetworkX DiGraph"
        assert graph is not None, "Input graph cannot be None"
        self.graph = graph
        self.layers = []
        self.node_to_layer = {}
        self.layer_connections = defaultdict(set)
        self.connection_masks = {}

    def cyclic_to_dag(self,input_nodes):
        """
        Converts a cyclic graph into a DAG by starting from input_nodes and only keeping forward edges,
        preserving edges that jump across layers.

        Parameters:
        - graph: A NetworkX DiGraph (may contain cycles)
        - input_nodes: A set of nodes to be considered as layer zero

        Returns:
        - A NetworkX DiGraph representing a DAG
        """
        # make sure input nodes are contained in the graph
        input_nodes=set(input_nodes) & set(self.graph.nodes())


        dag = nx.DiGraph()
        dag.add_nodes_from(self.graph.nodes)

        visited = set(input_nodes)
        stack = list(input_nodes)


        while stack:
            # assert len(set(stack))==len(stack), "Stack contains duplicate nodes"
            # assert set(stack) <= visited, "Stack contains nodes not in visited"
            node = stack.pop()
            for successor in self.graph.successors(node):
                if successor not in visited:
                    dag.add_edge(node, successor)
                    stack.append(successor)
                    visited.add(successor)

                elif not nx.has_path(dag, successor, node) and successor not in input_nodes: #avoid cycle

                    dag.add_edge(node, successor)

        self.graph=dag
        return self.graph


    def topological_sort_with_layers(self):
        """
        Perform topological sorting while grouping nodes into layers.

        Terminal nodes are accumulated and added to the final layer in the end.
        """
        in_degree = {node: self.graph.in_degree(node) for node in self.graph.nodes}


        sources = deque([node for node in in_degree if in_degree[node] == 0])
        leaves=[]
        while sources:
            current_layer = []
            next_sources = deque()

            while sources:
                node = sources.popleft()
                if self.graph.out_degree(node) == 0:
                    leaves.append(node)

                else:
                  current_layer.append(node)
                  self.node_to_layer[node] = len(self.layers)
                  for neighbor in self.graph.neighbors(node):
                      in_degree[neighbor] -= 1
                      if in_degree[neighbor] == 0:
                        next_sources.append(neighbor)
            if current_layer:
              self.layers.append(current_layer)

            sources = next_sources
        # put all leaf nodes into the output layer
        self.layers.append(leaves)
        self.node_to_layer.update({node: len(self.layers) - 1 for node in leaves})
        return self.layers,self.node_to_layer

    def build_layer_connections(self):
        """
        Build connections between layers based on predecessor relationships.
        """
        for i, layer in enumerate(self.layers[1:], start=1):
            for node in layer:
                for neighbor in self.graph.predecessors(node):
                    self.layer_connections[i].add(self.node_to_layer[neighbor])


    def build_masks_between_connected_layers(self):
        """
        Build adjacency masks between connected layers.
        """
        self.connection_masks = {key: {} for key in range(len(self.layers))}
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=sum(self.layers, []))
        node_to_index = {node: i for i, node in enumerate(sum(self.layers, []))}

        for layer, predecessor_layer_indices in self.layer_connections.items():
            for predecessor_layer in predecessor_layer_indices:
                predecessor_indices = [node_to_index[node] for node in self.layers[predecessor_layer]]
                layer_indices = [node_to_index[node] for node in self.layers[layer]]
                mask = adj_matrix[np.ix_(predecessor_indices, layer_indices)]
                self.connection_masks[layer][predecessor_layer] = mask



    def process_graph(self,input_nodes,output_nodes=None):
        """
        Run all processing steps on the graph.
        """
        assert input_nodes is not None, "Input nodes cannot be None"
        print(f"total number of neurons:{len(self.graph.nodes())}")
        self.cyclic_to_dag(input_nodes)
        leaf_nodes = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        num_leaf_nodes = len(leaf_nodes)
        print(f"Number of leaf nodes: {num_leaf_nodes}")
        self.topological_sort_with_layers()
        print(f"Number of layers:{len(self.layers)}")
        print(f"total number of neurons:{len(self.graph.nodes())}")
        print(f"Number of neurons in the first layer: {len(self.layers[0])}")
        print(f"Number of neurons in the last layer: {len(self.layers[-1])}")

        self.build_layer_connections()
        self.build_masks_between_connected_layers()

        return self.graph
