from omegaconf import OmegaConf

class Node:
    def __init__(self, name, simulator_cls, inferrer_cls = None, 
                 parents=[], evidence=[], scaffolds=[], observed=False, resample=False,
                 simulator_config={}, inferrer_config = {}, actor_config={}):
        """Node definition for a graphical model.

        Args:
            name (str): Name of the node.
            create_distr (class): Distribution class to create the node.
            config (dict): Configuration for the distribution.
            parents (list): List of parent node names (forward model).
            evidence (list): List of evidence node names (inference model).
            observed (bool): Whether the node is observed (act as root nodes for inference model).
            actor_name (str): Optional name of the actor to deploy the node.
            resample (bool): Whether to resample the node
        """
        self.name = name

#        # Obtain class definitions by auto-importing modules (is here the right place?)
# Should go to wrapper
#        if isinstance(simulator_cls, str):
#            simulator_cls = LazyLoader(simulator_cls)
#        if isinstance(inferrer_cls, str):
#            inferrer_cls = LazyLoader(inferrer_cls)

        self.simulator_cls = simulator_cls
        self.inferrer_cls = inferrer_cls

        self.parents = parents
        self.evidence = evidence
        self.scaffolds = scaffolds
        self.observed = observed
        self.resample = resample
        self.train = self.inferrer_cls is not None

        self.simulator_config = simulator_config
        self.inferrer_config = inferrer_config
        self.actor_config = actor_config


class Graph:
    def __init__(self, node_list):
        # Storing the node list
        self.node_list = node_list
        self.node_dict = {node.name: node for node in node_list}
        self.simulator_cls_dict = {node.name: node.simulator_cls for node in node_list}

        # Storing the model graph structure
        self.name_list = [node.name for node in node_list]
        self.parents_dict = {node.name: node.parents for node in node_list}
        self.sorted_node_names = self._topological_sort(self.name_list, self.parents_dict)

        # Storing the inference graph structure.
        # Only observed nodes or nodes with evidence are included in the inference graph.
        self.evidence_dict = {node.name: node.evidence for node in node_list}
        self.scaffolds_dict = {node.name: node.scaffolds for node in node_list}
        self.observed_dict = {node.name: node.observed for node in node_list}
        self.inference_name_list = [node.name for node in node_list if node.observed or len(node.evidence) > 0]
        self.sorted_inference_node_names = self._topological_sort(
            self.inference_name_list, self.evidence_dict)

    def get_resample_parents_and_graph(self, evidence):
        evidence = evidence[:]  # Shallow copy
        evidence_offline = []
        resample_subgraph = []
        while len(evidence) > 0:
            k = evidence.pop()
            if self.node_dict[k].resample:
                resample_subgraph.append(k)
                for parent in self.parents_dict[k]:
                    evidence.append(parent)
            else:
                evidence_offline.append(k)
        resample_subgraph = resample_subgraph[::-1]  # Reverse the order
        evidence_offline = list(set(evidence_offline))  # Remove duplicates
        return evidence_offline, resample_subgraph

    def get_parents(self, node_name):
        return self.parents_dict[node_name]

    def get_evidence(self, node_name):
        return self.evidence_dict[node_name]

    def get_simulator_cls(self, node_name):
        return self.simulator_cls_dict[node_name]

    @staticmethod
    def _topological_sort(name_list, parents_dict):
        """Topological sort, based on parent structure. Should raise an error if there is a cycle."""
        # Create a dictionary to track the number of parents (incoming edges) for each node
        num_parents = {node: 0 for node in name_list}
        
        # Count the number of parents for each node (incoming edges)
        for node in name_list:
            for parent in parents_dict[node]:
                if parent in num_parents:
                    num_parents[node] += 1

        # Create a list of nodes with no parents (no incoming edges)
        no_parents = [node for node in name_list if num_parents[node] == 0]

        # List to hold the sorted nodes
        sorted_node_names = []
        
        while no_parents:
            node = no_parents.pop()
            sorted_node_names.append(node)

            # For each node, look at its children (nodes where it is a parent)
            for child in name_list:
                if node in parents_dict[child]:
                    num_parents[child] -= 1
                    if num_parents[child] == 0:
                        no_parents.append(child)

        # If the sorted list doesn't include all nodes, there must be a cycle
        if len(sorted_node_names) != len(name_list):
            # Print informative error message about what is going wrong exactly
            print("Sorted nodes:", sorted_node_names)
            raise ValueError("Graph has a cycle")

        return sorted_node_names

    def __add__(self, other):
        """Merge two graphs."""
        new_node_list = self.node_list + other.node_list
        return Graph(new_node_list)

    def __str__(self):
        # Return graph structure
        # - Based on topological sort
        # - Include node names and their parents in the form NAME <- PARENT1, PARENT2, ... [MODULE]
        graph_str = "Falcon graph structure:\n"
        graph_str += f"  Node name          List of parents                                 Class name\n"
        for node in self.sorted_node_names:
            parents = self.get_parents(node)
            simulator_cls = self.get_simulator_cls(node)
            if hasattr(simulator_cls, 'display_name'):
                class_name = simulator_cls.display_name
            else:
                class_name = str(simulator_cls)
            graph_str += f"* {node:<15} <- {', '.join(parents):<45} | {class_name:<20}\n"
        return graph_str


class Extractor:
    def __init__(self, index):
        self.index = index

    def sample(self, batch_dim, parent_conditions=[]):
        composite, = parent_conditions
        x = composite[self.index]
        return x


def CompositeNode(names, module, **kwargs):
    """Auxiliary function to create a composite node with multiple child nodes."""

    # Generate name of composite node from names of child nodes
    joined_names  = "comp_"+"_".join(names)

    # Instantiate composite node
    node_comp = Node(joined_names, module, **kwargs)

    # Instantiate child nodes, which extract the individual components
    nodes = []
    for i, name in enumerate(names):
        node = Node(name, Extractor, parents=[joined_names], simulator_config=dict(index=i))
        nodes.append(node)

    # Return composite node and child nodes, which both must be added to the graph
    return node_comp, *nodes


def create_graph_from_config(graph_config, _cfg=None):
    """Create a computational graph from YAML configuration.
    
    Args:
        graph_config: Dictionary containing graph node definitions
        _cfg: Full Hydra configuration object (optional)
        
    Returns:
        Graph: The computational graph
    """
    nodes = []
    
    for node_name, node_config in graph_config.items():
        # Extract node parameters
        parents = node_config.get('parents', [])
        evidence = node_config.get('evidence', [])
        scaffolds = node_config.get('scaffolds', [])
        observed = node_config.get('observed', False)
        resample = node_config.get('resample', False)
        actor_config = node_config.get('ray', {})
        
        # Extract target from simulator
        simulator = node_config.get("simulate")
        if isinstance(simulator, str):
            target = node_config.get('simulate')
            simulator_config = {}
        else:
            target = node_config.get('simulate').get('_class_')
            simulator_config = node_config.get('simulate', {})
            simulator_config = OmegaConf.to_container(simulator_config, resolve=True)
            simulator_config.pop("_class_", None)

        # Extract target from infer
        if "infer" in node_config:
            inferrer = node_config.get("infer")
            if isinstance(inferrer, str):
                inferrer_cls = node_config.get('infer')
                inferrer_config = {}
            else:
                inferrer_cls = node_config.get('infer').get('_class_')
                inferrer_config = node_config.get('infer', {})
                inferrer_config = OmegaConf.to_container(inferrer_config, resolve=True)
                inferrer_config.pop("_class_", None)
        else:
            inferrer_cls = None
            inferrer_config = {}

        # Create the node
        node = Node(
            name=node_name,
            simulator_cls=target,
            inferrer_cls=inferrer_cls,
            parents=parents,
            evidence=evidence,
            scaffolds=scaffolds,
            observed=observed,
            resample=resample,
            simulator_config=simulator_config,
            inferrer_config=inferrer_config,
            actor_config=actor_config
        )
        
        nodes.append(node)
    
    # Create and return the graph
    return Graph(nodes)
