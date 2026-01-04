# vessel_analysis.py
import numpy as np
import networkx as nx
from scipy import ndimage


# 26-neighborhood for 3D connectivity
_NEIGHBORS_26 = [(dz, dy, dx)
                 for dz in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dx in (-1, 0, 1)
                 if not (dz == 0 and dy == 0 and dx == 0)]


def build_skeleton_graph(skeleton: np.ndarray, spacing_zyx):
    """
    Costruisce grafo da skeleton 3D.
    Nodi: coordinate (z,y,x) dei voxel skeleton
    Edges: peso = distanza euclidea in mm
    
    Args:
        skeleton: Binary 3D array
        spacing_zyx: tuple (spacing_z, spacing_y, spacing_x) in mm
        
    Returns:
        networkx.Graph con edge weight in mm
    """
    spacing = np.asarray(spacing_zyx, dtype=float)
    coords = np.argwhere(skeleton)
    coord_set = set(map(tuple, coords.tolist()))

    G = nx.Graph()
    
    # Aggiungi nodi
    for c in coord_set:
        G.add_node(c)

    # Aggiungi edges con peso
    for z, y, x in coord_set:
        for dz, dy, dx in _NEIGHBORS_26:
            neighbor = (z + dz, y + dy, x + dx)
            if neighbor in coord_set:
                # Distanza euclidea pesata per spacing
                displacement = np.array([dz, dy, dx], dtype=float) * spacing
                weight = float(np.linalg.norm(displacement))
                G.add_edge((z, y, x), neighbor, weight=weight)
    
    return G


def compute_radius_map_mm(vessel_mask: np.ndarray, spacing_zyx):
    """
    Calcola mappa raggi per ogni voxel usando EDT.
    
    Args:
        vessel_mask: Binary 3D mask
        spacing_zyx: tuple (spacing_z, spacing_y, spacing_x) in mm
        
    Returns:
        3D array con raggio in mm per ogni voxel
    """
    radius_map = ndimage.distance_transform_edt(
        vessel_mask.astype(bool), 
        sampling=spacing_zyx
    )
    return radius_map.astype(np.float32)


def identify_endpoints_and_bifurcations(graph: nx.Graph):
    """
    Identifica endpoints (degree 1) e biforcazioni (degree >= 3).
    
    Args:
        graph: NetworkX graph
        
    Returns:
        tuple: (num_endpoints, num_bifurcations, degree_distribution)
    """
    if graph.number_of_nodes() == 0:
        return 0, 0, {}
    
    degrees = np.array([d for _, d in graph.degree()], dtype=int)
    
    endpoints = int((degrees == 1).sum())
    bifurcations = int((degrees >= 3).sum())
    
    degree_dist = {}
    unique, counts = np.unique(degrees, return_counts=True)
    for deg, count in zip(unique, counts):
        degree_dist[int(deg)] = int(count)
    
    return endpoints, bifurcations, degree_dist


def compute_branch_lengths(graph: nx.Graph):
    """
    Calcola lunghezza di ogni ramo nel grafo.
    
    Args:
        graph: NetworkX graph con edge weights
        
    Returns:
        list: Liste di lunghezze rami in mm
    """
    branch_lengths = []
    
    for u, v, data in graph.edges(data=True):
        length = data.get('weight', 0.0)
        branch_lengths.append(length)
    
    return branch_lengths


def find_central_nodes(graph: nx.Graph, n=5, criterion='degree'):
    """
    Trova n nodi centrali secondo criterio specificato.
    
    Args:
        graph: NetworkX graph
        n: numero nodi da restituire
        criterion: 'degree', 'betweenness', 'closeness'
        
    Returns:
        list: Lista di nodi centrali
    """
    if graph.number_of_nodes() == 0:
        return []
    
    if criterion == 'degree':
        # Nodi con degree più alto
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:n]]
    
    elif criterion == 'betweenness':
        # Betweenness centrality (nodi su più shortest paths)
        betweenness = nx.betweenness_centrality(graph, weight='weight')
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:n]]
    
    elif criterion == 'closeness':
        # Closeness centrality (nodi con somma distanze minima)
        if nx.is_connected(graph):
            closeness = nx.closeness_centrality(graph, distance='weight')
            sorted_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:n]]
        else:
            # Per grafo disconnesso, usa solo la componente più grande
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph, distance='weight')
            sorted_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:n]]
    
    else:
        # Default: degree
        return find_central_nodes(graph, n, 'degree')


def compute_shortest_paths_from_nodes(graph: nx.Graph, source_nodes):
    """
    Calcola shortest paths da lista di nodi sorgente.
    
    Args:
        graph: NetworkX graph con edge weights
        source_nodes: lista di nodi sorgente
        
    Returns:
        dict: {source_node: {target: distance_mm}}
    """
    all_paths = {}
    
    for source in source_nodes:
        if source not in graph:
            continue
        
        # Dijkstra da questo source
        lengths = nx.single_source_dijkstra_path_length(
            graph, source, weight='weight'
        )
        all_paths[source] = lengths
    
    return all_paths


def assign_nodes_to_closest_source(graph: nx.Graph, source_nodes):
    """
    Assegna ogni nodo del grafo alla sorgente più vicina.
    
    Args:
        graph: NetworkX graph
        source_nodes: lista nodi sorgente
        
    Returns:
        dict: {node: (closest_source, distance_mm)}
    """
    path_lengths = compute_shortest_paths_from_nodes(graph, source_nodes)
    
    assignment = {}
    
    for node in graph.nodes():
        min_dist = np.inf
        closest_source = None
        
        for source, lengths in path_lengths.items():
            dist = lengths.get(node, np.inf)
            if dist < min_dist:
                min_dist = dist
                closest_source = source
        
        assignment[node] = (closest_source, min_dist)
    
    return assignment