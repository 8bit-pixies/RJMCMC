import networkx as nx

def graph_to_dict(G):
    return dict(nodes=[[n, G.node[n]] for n in G.nodes()],
                   edges=[[u, v, G.edge[u][v]] for u,v in G.edges()])

def dict_to_graph(dict_obj):
    G = nx.DiGraph()
    G.add_nodes_from(dict_obj['nodes'])
    G.add_edges_from(dict_obj['edges'])
    return G

def print_graph_from_dict(dict_obj, kwargs={}):
    G = dict_to_graph(dict_obj)
    nx.draw_networkx(G, **kwargs)

def set_graph_node_attributes(G1, node, attri_dict):
    """
    G is a networkx graph...
    
    This will **overwrite** all attributes associated with that node.
    """
    G = G1.copy()
    
    #dict_items = attri_dict.items()    
    for attri_name, attri_value in attri_dict.items():
        nx.set_node_attributes(G, attri_name, {node: attri_value})
    return G.copy()

