import osmnx as ox
import os
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import rustworkx as rx
import matplotlib.pyplot as plt

PLACE_NAME = "Montréal, Québec, Canada"
DIRECTED_FILE = "montreal_directed.graphml"
UNDIRECTED_FILE = "montreal_undirected.graphml"

def load_and_simplify(place_name: str):
    """Charge le graphe non orienté depuis fichier si présent,
    sinon le reconstruit, le sauvegarde (fichiers directed+undirected) et le renvoie."""
    if os.path.exists(UNDIRECTED_FILE):
        print(f"⇨ Chargement du graphe non orienté depuis {UNDIRECTED_FILE}")
        G_und = ox.load_graphml(UNDIRECTED_FILE)
        return G_und

    if os.path.exists(DIRECTED_FILE):
        print(f"⇨ Chargement du graphe orienté depuis {DIRECTED_FILE}")
        G = ox.load_graphml(DIRECTED_FILE)
    else:
        print("⇨ Téléchargement du graphe via OSMnx")
        G = ox.graph_from_place(place_name, network_type="drive")
        # on peut simplifier manuellement si besoin
        # G = ox.simplify_graph(G)
        print(f"⇨ Sauvegarde du graphe orienté dans {DIRECTED_FILE}")
        ox.save_graphml(G, filepath=DIRECTED_FILE)

    # conversion en non-orienté, sauvegarde, puis retour
    G_und = G.to_undirected()
    print(f"⇨ Sauvegarde du graphe non orienté dans {UNDIRECTED_FILE}")
    ox.save_graphml(G_und, filepath=UNDIRECTED_FILE)
    return G_und

def plot_graph(G: nx.Graph, show: bool = True, savepath: str = None):
    """Trace le graphe avec OSMnx + Matplotlib."""
    fig, ax = ox.plot_graph(G, show=False, close=False)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def compute_pairwise_distances(G: nx.Graph,
                               odd_nodes: list,
                               batch_size: int = 300):
    """
    Calcule dist[(u,v)] pour chaque paire u<v de odd_nodes,
    en lançant dijkstra en multi‐source par lots.
    """
    # 1) Indexation
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    M = len(odd_nodes)
    odd_idx = [idx[u] for u in odd_nodes]

    # 2) Construction unique de la matrice CSR symétrique
    rows, cols, data = [], [], []
    for u, v, attr in G.edges(data=True):
        w = attr.get("length", 1.0)
        iu, iv = idx[u], idx[v]
        rows += [iu, iv]
        cols += [iv, iu]
        data += [w, w]
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # 3) Batch Dijkstra + extraction mémoire‐maîtrisée
    dist = {}
    n_batches = (M + batch_size - 1) // batch_size
    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end   = min(start + batch_size, M)
        batch = odd_idx[start:end]  # indices CSgraph pour ce lot

        # **multi‐source** en C, shape = (len(batch), n)
        dist_batch = dijkstra(csgraph=A,
                              directed=False,
                              indices=batch)

        # On parcourt chaque ligne du lot et n’extrait que les colonnes
        # correspondantes aux odd_idx dont j_global > start+bi
        for bi, iu in enumerate(batch):
            u = nodes[iu]
            # pour éviter doublons, on ne prend que v tels que
            # index_global > start+bi
            for j_global in range(start + bi + 1, M):
                iv = odd_idx[j_global]
                v = nodes[iv]
                dist[(u, v)] = dist_batch[bi, iv]

        # on libère tout de suite la grosse matrice temporaire
        del dist_batch
        print(f"dijkstra batch no {batch_i}/{n_batches} finished")

    return dist

def minimum_weight_matching(odd_nodes: list, dist: dict):
    """
    Sur le graphe complet K d'impairs, trouve l'appariement
    parfait de poids minimal via retworkx (max_weight sur -poids).
    """
    # Construit PyGraph avec nœuds [0..m-1]
    m = len(odd_nodes)
    g = rx.PyGraph()
    g.extend_from_node_list(range(m))

    # Ajoute les arêtes avec poids négatif
    for i in range(m):
        for j in range(i+1, m):
            u, v = odd_nodes[i], odd_nodes[j]
            w = dist[(u, v)]
            g.add_edge(i, j, -w)

    # Matching max weight = minimal weight sur +dist
    matching = rx.max_weight_matching(g)
    # transforme en paires de nœuds originels
    return [(odd_nodes[i], odd_nodes[j]) for i, j in matching]


def duplicate_edges(G: nx.Graph, pairs: list):
    """Duplique dans un MultiGraph chaque chemin entre paires impaires."""
    G_euler = nx.MultiGraph(G)
    for u, v in pairs:
        path = nx.shortest_path(G, u, v, weight="length")
        for a, b in zip(path[:-1], path[1:]):
            data = G_euler.get_edge_data(a, b)
            key = next(iter(data))
            attrs = data[key].copy()
            G_euler.add_edge(a, b, **attrs)
    return G_euler


# ---------------------------------------------------------------------------- #
# 3) EXTRACTION DU CIRCUIT EULÉRIEN
# ---------------------------------------------------------------------------- #

def extract_eulerian_circuit(G_euler: nx.MultiGraph):
    """
    Renvoie la liste des arêtes (u,v) dans l'ordre du circuit.
    """
    assert nx.is_eulerian(G_euler), "Le graphe n'est pas eulérien !"
    return list(nx.eulerian_circuit(G_euler))


def main():
    # 1) Chargement / simplification / affichage initial
    print("--- load and simplify ---")
    G_und = load_and_simplify(PLACE_NAME)
    print(f"Graph chargé : {len(G_und.nodes)} nœuds, {len(G_und.edges)} arêtes")
    plot_graph(G_und, show=False, savepath="montreal_simplified.png")

    # 2) Calcul des nœuds impairs & distances
    odd = [n for n, d in G_und.degree() if d % 2 == 1]
    print(f"{len(odd)} sommets impairs")
    dist = compute_pairwise_distances(G_und, odd, 500)
    print("pairwise distance computed")
    pairs = minimum_weight_matching(odd, dist)
    print(f"{len(pairs)} paires optimales calculées")

    # 3) Duplication & extraction du circuit
    G_euler = duplicate_edges(G_und, pairs)
    circuit = extract_eulerian_circuit(G_euler)
    print(f"Circuit eulérien généré : {len(circuit)} arêtes au total")

    # 4) Affichage final du circuit
    # on crée un sous-raphe orienté pour tracer le parcours
    H = nx.DiGraph()
    H.add_edges_from(circuit)
    plot_graph(H, show=True)

if __name__ == "__main__":
    main()
