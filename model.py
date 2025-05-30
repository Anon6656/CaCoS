
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class CohesivePool(torch.nn.Module):
    def __init__(self, num_features, nhid, pooling_ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.subgraph_conv = GCNConv(nhid, nhid)
        self.pool = SAGPooling(nhid, ratio=pooling_ratio)
        self.nhid = nhid  # Store hidden dimension

    def forward(self, data):
        # Generate node embeddings
        x1 = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x1, data.edge_index))  # [num_nodes, nhid]
       
        # Process with SAGPooling
        x_pool, edge_index, _, batch, perm, _ = self.pool(x1, data.edge_index)
        
        
        # Create fixed-size subgraph embedding
        subgraph_emb1 = torch.cat([
            gmp(x_pool, batch),  # [1, nhid]
            gap(x_pool, batch)    # [1, nhid]
        ], dim=1)  # [1, nhid*2]
        
        x_sub = F.relu(self.subgraph_conv(x_pool, edge_index))
        subgraph_emb2 = torch.cat([
            gmp(x_sub, batch),  # [1, nhid]
            gap(x_sub, batch)    # [1, nhid]
        ], dim=1)  # [1, nhid*2]
        
        return x, (subgraph_emb1 + subgraph_emb2), edge_index, perm, batch

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = torch.nn.Linear(embed_dim, 3*embed_dim)
        self.out = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B = query.size(0)
        qkv = self.qkv(query).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, self.embed_dim)
        return self.out(out)

class NodeClassifier(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio = 0.5, num_heads = 2):
        super().__init__()
        self.cacos = CohesivePool(num_features, nhid, pooling_ratio=pooling_ratio)
        self.subgraph_attention = MultiHeadSelfAttention(nhid * 2, num_heads = num_heads)  # Match pooled dim
        self.conv_final = GCNConv(nhid * 3, num_classes)
        # self.lin = torch.nn.Linear(nhid * 3, num_classes)  # nhid (node) + nhid*2 (subgraph)
        self.nhid = nhid

    def forward(self, data, epoch = 0):
        all_subgraphs = data.subgraphs
        if not all_subgraphs:
            raise ValueError("No subgraphs found in graph")

        subgraph_info = []
        all_original_edge_indices = []
        
        for subg in all_subgraphs:
            # Get node emb (nhid) and subgraph emb (nhid*2)
            node_embs, subgraph_emb, edge_index, perm, _ = self.cacos(subg)
            
            ######### Mapping for testing ##############################
            # Get original node indices for the pooled subgraph
            original_node_indices = subg.original_node_indices[perm]  # [num_pooled_nodes]

            # Map edge indices to original graph's node IDs
            original_edge_index = original_node_indices[edge_index]  # [2, num_edges]
            all_original_edge_indices.append(original_edge_index.cpu())
            # print(original_edge_index)
            #############################################################
            
            subgraph_info.append({
                'node_embs': node_embs,
                'subgraph_emb': subgraph_emb.squeeze(0),  # Remove batch dim
                'mapping': subg.original_node_indices
            })
        
        # Stack subgraph embeddings (now all [nhid*2])
        subgraph_embeddings = torch.stack([info['subgraph_emb'] for info in subgraph_info])  # [num_subgraphs, nhid*2]
        
        # Apply cross-attention
        attended_subgraphs = self.subgraph_attention(
            subgraph_embeddings, 
            subgraph_embeddings, 
            subgraph_embeddings
        ).squeeze(1)  # [num_subgraphs, nhid*2]
        
        # Aggregate node + subgraph features
        global_emb = torch.zeros(data.num_nodes, self.nhid * 3, device=data.x.device)
        counts = torch.zeros(data.num_nodes, device=data.x.device)

        for idx, info in enumerate(subgraph_info):
            subg_emb = attended_subgraphs[idx]  # [nhid*2]
            node_embs = info['node_embs']  # [num_subgraph_nodes, nhid]
            mapping = info['mapping']  # Original node indices
            
            # Concatenate features
            combined = torch.cat([
                node_embs[:len(mapping)],  # [num_nodes, nhid]
                subg_emb.unsqueeze(0).expand(len(mapping), -1)  # [num_nodes, nhid*2]
            ], dim=1)  # [num_nodes, nhid*3]
            
            # Update global embeddings
            global_emb[mapping] += combined
            counts[mapping] += 1
            
        return F.log_softmax(self.conv_final(global_emb, data.edge_index), dim=-1), global_emb