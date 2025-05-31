import torch
import networkx as nx
from torch.utils.data import Dataset
from collections import OrderedDict
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils


class ProcessDataset(Dataset):
    def __init__(self, graph, label_file, dataset = 'Cora', decom_type = 'core', **kwargs):

        # taking the initial graph as input graph ################
        self.inp_graph = graph
        self.label_file = label_file
        self.device = kwargs.get('device', 'cuda')
        self.dataset = dataset
        self.PyG_data_graph = None
        self.decom_type = decom_type
        self.seed = 1
      
    def get_label_dict(self):
      label_dict = OrderedDict()

      with open(self.label_file, 'r') as labelfile:
          for line in labelfile:
              line = line.split(' ')
              line[1] = int(line[1].rstrip('\n'))
              label_dict[int(line[0])] = line[1]
          num_of_classes = len(set(label_dict.values()))
      return label_dict, num_of_classes

    ############################ Truss Decomposition #############################################
    def get_all_edges_trussness_and_decomposition(self, G):

        """
        Compute and assign the trussness score to each edge in graph G.

        This function iteratively finds the k-truss subgraph for increasing values
        of k, starting from k=2. For each value of k, it determines the edges that
        belong to the (k-1)-truss but not the k-truss. For these edges, it assigns
        the trussness score as k-1 and also calculates and assigns their support.
        The trussness score and support are stored in the 'weight' attribute of the
        edges in a copy of the original graph G.

        Parameters:
        - G (NetworkX graph): The input graph.

        Returns:
        - G_main (NetworkX graph): A copy of G where each edge has an assigned
                    'weight' attribute containing a dictionary with the trussness and
                    support.
        - trussness_dict (dict): A dictionary where the keys are trussness scores
                    and the values are lists of edges that have the corresponding trussness.
        """

        i = 2
        trussness_dict = dict()
        edge_percent_dict = dict() 
        nodes_trussness_dict = dict()
        
        G_main = G.copy()

        G1 = None
        total_edges = 0
        while (1):
            print(i, end=', ')
            G = nx.k_truss(G, i)
    
            if i > 2:
                edges = set(list(G1.edges())) - set(list(G.edges())) # get_trussness_edges(edge_list1, edge_list2) #
                edges = list(edges)
                nodes_set = set(list(G_main.edge_subgraph(edges).nodes()))
                # edges = get_trussness_edges(list(G1.edges()), list(G.edges()))
                for u, v in edges:
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = i-1
    
                if len(edges) != 0:
                    trussness_dict[i-1] = edges
                    nodes_trussness_dict[i-1] = nodes_set
                    edge_percent_dict[i-1] = len(edges)/len(G_main.edges()) 
                total_edges += len(edges)
    
                if len(G.edges()) == 0 and len(edges) == 0:
                    break
    
            G1 = G.copy()
            i += 1
        
        return G_main, trussness_dict #, edge_list_dict


    ################################# Core Decomposition  ###############################################
    def get_all_edges_coreness_and_decomposition(self, G):

        G_main = G.copy()  # print(G_main,'--------Main Graph Info-------')
        coreness_dict = dict()
        nodes_coresness_dict = dict()

        prev_Core_Graph = nx.Graph()
        edge_count = 0

        i = 0
        while(1):
            Gk = nx.k_core(G, i)

            # deducted_nodes =  set(list(prev_Core_Graph.nodes())).difference(set(list(Gk.nodes())))
            deducted_edges = set([tuple(sorted([u,v])) for u, v in prev_Core_Graph.edges()]) \
                                - (set([ tuple(sorted([u, v])) for u,v in Gk.edges()]))
            nodes_set = set(list(G_main.edge_subgraph(deducted_edges).nodes()))
            edge_count += len(deducted_edges)

            if i <= 1:
                pass
            # elif i == 1:
            #     coreness_dict[i-1] = list(deducted_nodes)
            else:
                for u, v in deducted_edges: 
                    if G_main.has_edge(u, v): 
                        G_main[u][v]['weight'] = i-1

                coreness_dict[i-1] = list(deducted_edges)
                nodes_coresness_dict[i-1] = nodes_set
            # nodeSubG = prev_Core_Graph.subgraph(deducted_nodes)
            edgeSubG = prev_Core_Graph.edge_subgraph(deducted_edges)


            if i > 0 and len(edgeSubG.nodes()) > 0:
                pass
 
            if len(Gk.nodes())== 0 and len(Gk.edges()) == 0:
                break
            i += 1
            prev_Core_Graph = Gk

        # print(coreness_dict,'------')
        unused_key_list = list()
        for key, value in coreness_dict.items():
            if len(coreness_dict[key]) == 0:
                unused_key_list.append(key)
        for key in unused_key_list:
            del coreness_dict[key]
                    
        return G_main, coreness_dict 
    ############################################################################

    def cohesive_subgraph_extraction(self, edge_index, edge_attr, x):

        sorted_edges, _ = torch.sort(edge_index, dim=0)
        standardized_edge_index = sorted_edges

        unique_edges, inverse_indices = torch.unique(standardized_edge_index, dim=1, return_inverse=True)
        expanded_edge_weights = edge_attr[inverse_indices.to(edge_attr.device)]

        unique_edge_attrs = torch.unique(expanded_edge_weights)

        subgraphs = {}
        for attr_value in unique_edge_attrs:
            mask = expanded_edge_weights == attr_value
            filtered_edge_index = edge_index[:, mask]
            filtered_edge_attr = expanded_edge_weights[mask]

            # Get original node indices in subgraph
            subgraph_node_indices = torch.unique(filtered_edge_index)
            subgraph_node_features = x[subgraph_node_indices]

  
            # Efficient tensor remapping using broadcasting
            idx_mapper = torch.zeros(torch.max(subgraph_node_indices) + 1, 
                                dtype=torch.long)
            idx_mapper[subgraph_node_indices] = torch.arange(len(subgraph_node_indices))
            remapped_edge_index = idx_mapper[filtered_edge_index.to(idx_mapper.device)]

            # Store mapping as tensor for efficient aggregation later
            subgraph_data = Data(
                x=subgraph_node_features,
                edge_index=remapped_edge_index,
                edge_attr=filtered_edge_attr,
                original_node_indices=subgraph_node_indices  # Tensor of original indices
            )
            
            subgraphs[attr_value.item()] = subgraph_data

        return subgraphs
    
    def get_train_val_test_masks(self, n_nodes, seed = 1):
        
        self.seed = seed
        indices = torch.arange(n_nodes)
        indices = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(self.seed))

        # train, test, validation
        trn = 0.48
        vld = 0.32
        # tst = (1 - (trn+vld))

        if self.dataset == 'PubMed':
          n_train, n_val, n_test = 18217, 500, 500 # int(n_nodes * trn)
        elif self.dataset == 'Cora':
          n_train, n_val, n_test = 1208, 500, 500 # int(n_nodes * trn)
        elif self.dataset == 'CiteSeer':
          n_train, n_val, n_test = 1812, 500, 500 # int(n_nodes * trn)
        else:
          n_train = int(n_nodes * trn)
          n_val = int(n_nodes * vld)
          n_test = int(n_nodes - (n_train + n_val))

        print(f"Train {n_train} \ Validation {n_val} \ Test {n_test}")

        train_indices = indices[:n_train]
        val_indices = indices[n_train:(n_train + n_val)]
        test_indices = indices[(n_train + n_val):(n_train + n_val + n_test)]

        # train-validation-test split
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        return train_mask, val_mask, test_mask
        
    def process(self):
        
        
        G = self.inp_graph

        # making the graph bi-directional
        edge_list = list()
        for u, v in G.edges():
            edge_list.append((u,v))
            edge_list.append((v,u))
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        if self.decom_type == 'truss':
            G_main, trussness_dict = \
                self.get_all_edges_trussness_and_decomposition(G)
        elif self.decom_type == 'core':
            G_main, trussness_dict = \
                self.get_all_edges_coreness_and_decomposition(G)

        
        # getting labels and number of classes
        label_dict, num_of_classes = self.get_label_dict()
        nodes_data = list(label_dict.keys())
        node_list = [node for node in G.nodes()]


        # node_labels = torch.tensor([label_dict[int(node)] for node in G.nodes()])
        node_labels = torch.tensor([int(value) for value in label_dict.values()])
        

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = len(nodes_data)
        train_mask, val_mask, test_mask = self.get_train_val_test_masks(n_nodes = n_nodes, seed = self.seed)
        

        # deg = torch.tensor([G.degree(node) for node in node_list])
        deg = torch.tensor([1 for node in node_list])
        node_features = torch.diag(deg).float()
        
        # self.PyG_data_graph = Data(x = node_features, edge_index = edge_index, y = node_labels,train_mask = train_mask, val_mask = val_mask, test_mask = test_mask ).to(device)
        self.PyG_data_graph = Data(x = node_features, edge_index = edge_index, y = node_labels,train_mask = train_mask, \
                                   val_mask = val_mask, test_mask = test_mask ).to(self.device)
        edges = list(G_main.edges(data = 'weight'))
        self.PyG_data_graph.edge_attr = torch.tensor([weight for _, _, weight in edges], dtype=torch.float)
        subgraph_tensor = self.cohesive_subgraph_extraction(self.PyG_data_graph.edge_index, self.PyG_data_graph.edge_attr, self.PyG_data_graph.x)
        self.PyG_data_graph.subgraphs = list(subgraph_tensor.values())
        print(self.PyG_data_graph)
        self.PyG_data_graph.num_classes = num_of_classes
            
        return self.PyG_data_graph
    
    def get_graph(self):
        return self.graph
    def __getitem__(self, idx):
         return self.PyG_data_graph
    
    def __len__(self):
        return 1