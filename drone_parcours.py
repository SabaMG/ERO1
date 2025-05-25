import osmnx as ox
import networkx as nx
import itertools
from networkx.algorithms.euler import eulerian_circuit

def solve_cpp(G):
    G = G.copy()

    # 1. Identifier les sommets de degré impair
    odd_nodes = [v for v, d in G.degree() if d % 2 == 1]

    if len(odd_nodes) == 0:
        print("Graphe déjà eulérien.")
    else:
        # 2. Trouver tous les couples de sommets impairs
        pairs = list(itertools.combinations(odd_nodes, 2))

        # 3. Calculer la distance minimale entre chaque paire
        pair_weights = {}
        for u, v in pairs:
            try:
                dist = nx.shortest_path_length(G, u, v, weight='length')
                pair_weights[(u, v)] = dist
            except nx.NetworkXNoPath:
                continue  # ignorer si aucun chemin

        # 4. Graphe auxiliaire pour le matching
        M = nx.Graph()
        for (u, v), w in pair_weights.items():
            M.add_edge(u, v, weight=-w)  # négatif pour maximiser le gain

        # 5. Appariement parfait
        matching = nx.algorithms.matching.max_weight_matching(M, maxcardinality=True)

        # 6. Ajouter les plus courts chemins au graphe d’origine
        for u, v in matching:
            path = nx.shortest_path(G, u, v, weight='length')
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1], length=G[path[i]][path[i+1]][0]["length"])

    # 7. Vérification et calcul du circuit eulérien
    assert nx.is_eulerian(G), "Le graphe n'est pas eulérien après appariement."
    circuit = list(eulerian_circuit(G))
    return circuit

G = ox.load_graphml("montreal_undirected.graphml")
circuit = solve_cpp(G)
print("Nombre d’arêtes dans le circuit :", len(circuit))
route = [u for u, v in circuit] + [circuit[-1][1]]
ox.plot_graph_route(G, route, route_linewidth=2)
