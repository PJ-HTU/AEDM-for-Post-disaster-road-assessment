import torch
import numpy as np
import random
import networkx as nx

def get_random_problems(batch_size, problem_size, original_node_count, link_count, use_fixed_seed=False):
    train_s, train_d, adj_matrices = generate_network_batch(batch_size, problem_size,
                                                            original_node_count, link_count)
    depot_xy = train_s[:, 0:1, :] 
    node_xy = train_s[:, 1:, :]
    node_demand = train_d[:, 1:, :].squeeze()

    return depot_xy, node_xy, node_demand, adj_matrices

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

class GridNetworkGenerator:
    def __init__(self, node_count: int, link_count: int):
        
        self.node_count = node_count  
        self.link_count = link_count
        self.grid_size = int(np.ceil(np.sqrt(node_count - 1)))  

    def generate_initial_grid(self) -> tuple:

        nodes = {}
        edges = []
        offset_range = 1 / (2 * (self.grid_size - 1))
        
        depot_x = random.uniform(0, 1)
        depot_y = random.uniform(0, 1)
        nodes[0] = (depot_x, depot_y)
        
        node_id = 1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if node_id < self.node_count:
                    x = j / (self.grid_size - 1)
                    y = i / (self.grid_size - 1)
                    
                    x += random.uniform(-offset_range, offset_range)
                    y += random.uniform(-offset_range, offset_range)
                    
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    nodes[node_id] = (x, y)
                    node_id += 1
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current_id = i * self.grid_size + j + 1
                if current_id >= self.node_count:
                    break
                if j < self.grid_size - 1:
                    right_id = i * self.grid_size + (j + 1) + 1
                    if right_id < self.node_count:
                        edges.append((current_id, right_id))
                if i < self.grid_size - 1:
                    down_id = (i + 1) * self.grid_size + j + 1
                    if down_id < self.node_count:
                        edges.append((current_id, down_id))
        
        return nodes, edges

    def prune_edges(self, edges: list, nodes: int) -> list:

        G = nx.Graph(edges)
        if not nx.is_connected(G):
            raise ValueError("Error, graph is not connected!")
  
        actual_node_count = nodes - 1  
        min_edges = actual_node_count - 1 
        
        center_range = max(1, int(nodes * 0.25))  
        center_nodes = list(range(center_range, nodes - center_range))
        
        if center_nodes:
            start_node = random.choice(center_nodes)
        else:
            start_node = random.choice(list(G.nodes()))
        
        visited = {start_node}
        tree_edges = []
        queue = [start_node]
        
        while queue and len(tree_edges) < min_edges:
            node = queue.pop(0)
            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    tree_edges.append((node, neighbor))
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(tree_edges) == min_edges:
                        break
        
        need_add = self.link_count - min_edges
        
        all_edges = set(edges)
        tree_set = set([(u, v) for u, v in tree_edges]) | set([(v, u) for u, v in tree_edges])
        non_tree_edges = [e for e in all_edges if e not in tree_set]
        
        return tree_edges + random.sample(non_tree_edges, need_add)
        
    def split_edges(self, nodes: dict, edges: list) -> tuple:
        all_nodes = nodes.copy()
        new_edges = []
        next_id = max(nodes.keys()) + 1
        
        for a, b in edges:
            xa, ya = nodes[a]
            xb, yb = nodes[b]
            xc, yc = (xa+xb)/2, (ya+yb)/2
            
            length = np.sqrt((xb-xa)**2 + (yb-ya)**2)
            if length > 0:
                perp_x = -(yb-ya)/length
                perp_y = (xb-xa)/length
                dev = random.uniform(-length*0.3, length*0.3)
                xc += perp_x * dev
                yc += perp_y * dev
            
            xc_reflected, yc_reflected = xc, yc
            
            if xc < 0:
                xc_reflected = -xc  
            elif xc > 1:
                xc_reflected = 2 - xc 

            if yc < 0:
                yc_reflected = -yc  
            elif yc > 1:
                yc_reflected = 2 - yc  
            
            xc = max(0, min(1, xc_reflected))
            yc = max(0, min(1, yc_reflected))
            
            all_nodes[next_id] = (xc, yc)
            new_edges.append((a, next_id))
            new_edges.append((next_id, b))
            next_id += 1
        
        return all_nodes, new_edges

    def create_adjacency_matrix(self, nodes: dict, edges: list, original_ids: list) -> np.ndarray:

        n = len(nodes)
        adj = np.zeros((n, n), dtype=int)
        node2idx = {node_id: i for i, node_id in enumerate(sorted(nodes.keys()))}
        
        for u, v in edges:
            i, j = node2idx[u], node2idx[v]
            adj[i, j] = 1
            adj[j, i] = 1
        
        for idx in original_ids:
            for idy in original_ids:
                if idx != idy:  
                    adj[idx, idy] = 1
                    adj[idy, idx] = 1
        
        return adj

    def generate_demands(self, original_ids: list, new_ids: list) -> dict:

        demands = {oid: 0 for oid in original_ids}
        demands.update({nid: random.uniform(1, 10)/10 for nid in new_ids})
        return demands

    def generate_single_network(self) -> tuple:

        original_nodes, initial_edges = self.generate_initial_grid()
        pruned_edges = self.prune_edges(initial_edges, self.node_count)
        all_nodes, final_edges = self.split_edges(original_nodes, pruned_edges)
        
        original_ids = list(original_nodes.keys())
        new_ids = set(all_nodes.keys()) - set(original_ids)
        adj_matrix = self.create_adjacency_matrix(all_nodes, final_edges, original_ids)
        demands = self.generate_demands(original_ids, new_ids)
        
        sorted_ids = sorted(all_nodes.keys())
        coords = np.array([all_nodes[oid] for oid in sorted_ids], dtype=np.float32)
        node_demands = np.array([demands[oid] for oid in sorted_ids], dtype=np.float32).reshape(-1, 1)
        
        return coords, node_demands, adj_matrix

def generate_network_batch(batch_size, problem_size, original_node_count, link_count):
    
    s = torch.zeros(batch_size, problem_size+1, 2, dtype=torch.float32)
    d = torch.zeros(batch_size, problem_size+1, 1, dtype=torch.float32)
    adj_matrices = torch.zeros(batch_size, problem_size+1, problem_size+1, dtype=torch.float32)
    
    for i in range(0, batch_size):
        generator = GridNetworkGenerator(original_node_count, link_count)
        coords, demands, adj = generator.generate_single_network()
        s[i] = torch.from_numpy(coords)
        d[i] = torch.from_numpy(demands)
        adj_matrices[i] = torch.from_numpy(adj)
    
    return s, d, adj_matrices