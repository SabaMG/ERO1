import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

PLACE_NAME = "Montréal, Québec, Canada"
DIRECTED_FILE = "montreal_directed.graphml"
UNDIRECTED_FILE = "montreal_undirected.graphml"


def load_and_simplify(place_name: str) -> nx.Graph:
    """
    Charge le graphe non orienté depuis fichier s'il existe,
    sinon le télécharge, sauvegarde directed + undirected, et le retourne.
    """
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
        # OSMnx simplifie par défaut en version ≥1.2, sinon décommenter :
        # G = ox.simplify_graph(G)
        print(f"⇨ Sauvegarde du graphe orienté dans {DIRECTED_FILE}")
        ox.save_graphml(G, filepath=DIRECTED_FILE)

    G_und = G.to_undirected()
    print(f"⇨ Sauvegarde du graphe non orienté dans {UNDIRECTED_FILE}")
    ox.save_graphml(G_und, filepath=UNDIRECTED_FILE)
    return G_und


def plot_graph(G: nx.Graph, show: bool = True, savepath: str = None):
    """
    Trace le graphe avec OSMnx + Matplotlib.
    """
    fig, ax = ox.plot_graph(G, show=False, close=False)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_survol_cost(G_und: nx.Graph, G_euler: nx.MultiGraph, path_edges: list):
    """
    Calcule la distance totale (en km) et le coût (en €) pour le circuit eulérien.
    """
    total_length_m = 0.0
    for edge in path_edges:
        # edge = (u, v) pour Graph simple, ou (u, v, key) pour MultiGraph
        if len(edge) == 2:
            u, v = edge
            data = G_und.get_edge_data(u, v)
            key0 = next(iter(data))
            total_length_m += data[key0]["length"]
        else:
            u, v, k = edge
            data_euler = G_euler.get_edge_data(u, v, k)
            total_length_m += data_euler["length"]

    total_distance_km = total_length_m / 1000.0
    cost_fixed = 100.0           # 100 € fixe / jour
    cost_per_km = 0.01           # 0.01 € / km
    cost_variable = cost_per_km * total_distance_km
    total_cost = cost_fixed + cost_variable
    return total_distance_km, total_cost


def main():
    # 1) Chargement et simplification
    print("--- load and simplify ---")
    G_und = load_and_simplify(PLACE_NAME)
    print(f"Graph chargé : {len(G_und.nodes)} nœuds, {len(G_und.edges)} arêtes")

    # Affichage facultatif du graphe simplifié
    plot_graph(G_und, show=False, savepath="montreal_simplified.png")

    # 2) Calcul de la longueur totale de toutes les arêtes existantes
    total_length_m = sum(
        data.get("length", 0.0) for _, _, data in G_und.edges(data=True)
    )
    total_length_km = total_length_m / 1000.0
    print(f"Longueur totale de toutes les arêtes : {total_length_km:.2f} km")

    # 3) Rendre le graphe eulérien (ajoute les duplications nécessaires)
    print("-- eulerizing graph via nx.eulerize --")
    # On passe weight="length" pour que nx.eulerize duplique en minimisant la somme des 'length'
    G_euler = nx.eulerize(G_und, weight="length")
    print(f"Graphe eulérien créé : {len(G_euler.nodes)} nœuds, {G_euler.number_of_edges()} arêtes")

    # 4) Extraire un chemin eulérien (ou circuit) dans ce graphe
    print("-- extracting eulerian path --")
    # Sur un graphe eulérien, nx.eulerian_path génère soit un circuit fermé, soit un chemin
    path_edges = list(nx.eulerian_path(G_euler))
    print(f"Chemin eulérien extrait : {len(path_edges)} arêtes au total")

    # 5) Calculer la distance et le coût du survol
    distance_km, cost = compute_survol_cost(G_und, G_euler, path_edges)
    print(f"Distance totale à parcourir : {distance_km:.2f} km")
    print(f"Coût total du survol : {cost:.2f} €")

    # 6) Préparer la liste de nœuds pour le tracé
    nodes_route = [path_edges[0][0]]
    for edge in path_edges:
        # edge = (u, v) ou (u, v, key)
        nodes_route.append(edge[1])

    # 7) Affichage du chemin eulérien sur le graphe
    print("-- plotting eulerian path --")
    fig2, ax2 = ox.plot_graph_route(
        G_und,
        nodes_route,
        route_linewidth=2,
        node_size=0,
        bgcolor="w"
    )
    plt.show()


if __name__ == "__main__":
    main()