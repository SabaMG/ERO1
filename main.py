import osmnx as ox
import os
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
import numpy as np

#PLACE_NAME = "Montréal, Québec, Canada"
PLACE_NAME = "Outremont, Montreal, Canada"
#DIRECTED_FILE = "montreal_directed.graphml"
DIRECTED_FILE = "outremont_directed.graphml"
UNDIRECTED_FILE = "outremont_undirected.graphml"
#UNDIRECTED_FILE = "montreal_undirected.graphml"


def load_and_simplify(place_name: str) -> nx.Graph:
    """
    Charge le graphe non orienté depuis fichier s'il existe,
    sinon le télécharge, sauvegarde directed + undirected, et le retourne.
    """
    if os.path.exists(UNDIRECTED_FILE):
        print(f"⇨ Chargement du graphe non orienté depuis {UNDIRECTED_FILE}")
        return ox.load_graphml(UNDIRECTED_FILE)

    if os.path.exists(DIRECTED_FILE):
        print(f"⇨ Chargement du graphe orienté depuis {DIRECTED_FILE}")
        G = ox.load_graphml(DIRECTED_FILE)
    else:
        print("⇨ Téléchargement du graphe via OSMnx")
        G = ox.graph_from_place(place_name, network_type="drive")
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


def compute_pairwise_distances_euclid(G: nx.Graph,
                                      odd_nodes: list,
                                      batch_size: int = 500) -> dict:
    """
    Calcule dist[(u, v)] = distance euclidienne (vol direct) entre chaque paire u<v
    de odd_nodes, en utilisant des calculs vectorisés par blocs (batch) pour rester
    en mémoire raisonnable.

    Retourne un dict dont les clefs sont des tuples (u,v) avec u<v, et les valeurs
    la distance en mètres.
    """
    # 1) On extrait la position (y=lat, x=lon) de chaque odd_node dans deux vecteurs NumPy.
    M = len(odd_nodes)
    lat = np.zeros(M, dtype=np.float64)
    lon = np.zeros(M, dtype=np.float64)
    for i, n in enumerate(odd_nodes):
        lat[i] = G.nodes[n]["y"]
        lon[i] = G.nodes[n]["x"]

    # Fonction Haversine vectorisée (opère sur arrays)
    def haversine_vec(lat1, lon1, lat2, lon2):
        """
        Attendu : lat1, lon1, lat2, lon2 sont en radians (NumPy arrays) ;
        renvoie la matrice des distances (en mètres).
        """
        R = 6371000.0  # rayon moyen de la Terre en m
        dphi = lat2 - lat1[:, None]    # dim = (len(lat1), len(lat2))
        dlambda = lon2 - lon1[:, None] # idem
        a = np.sin(dphi / 2) ** 2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # 2) On convertit lat/lon en radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dist = {}  # on va stocker dist[(u,v)] en mètres

    # 3) On découpe les indices [0..M-1] en lots de taille batch_size
    n_batches = (M + batch_size - 1) // batch_size
    for bi in range(n_batches):
        i_start = bi * batch_size
        i_end = min(i_start + batch_size, M)
        lat_i = lat_rad[i_start:i_end]
        lon_i = lon_rad[i_start:i_end]

        for bj in range(bi, n_batches):
            j_start = bj * batch_size
            j_end = min(j_start + batch_size, M)
            lat_j = lat_rad[j_start:j_end]
            lon_j = lon_rad[j_start:j_end]

            # 4) Calcul vectorisé des distances entre le bloc i et le bloc j
            #    Si bi == bj, on calcule la matrice complète puis on ne garde que i<j
            D = haversine_vec(lat_i, lon_i, lat_j, lon_j)
            # D.shape = (i_end - i_start, j_end - j_start)

            for ii in range(i_end - i_start):
                u = odd_nodes[i_start + ii]
                # si bi == bj, on ne prend que j > i
                if bi == bj:
                    for jj in range(ii + 1, j_end - j_start):
                        v = odd_nodes[j_start + jj]
                        dist[(u, v)] = float(D[ii, jj])
                else:
                    for jj in range(j_end - j_start):
                        u_idx = i_start + ii
                        v_idx = j_start + jj
                        # toujours u_idx < v_idx parce que bi < bj
                        uu = odd_nodes[u_idx]
                        vv = odd_nodes[v_idx]
                        dist[(uu, vv)] = float(D[ii, jj])

        print(f"batch {bi+1}/{n_batches} done")

    return dist


def minimum_weight_matching(odd_nodes: list, dist: dict) -> list:
    """
    Sur le graphe complet K d'impairs, trouve l'appariement
    parfait de poids minimal via retworkx (max_weight sur -poids).
    """
    m = len(odd_nodes)
    g = rx.PyGraph()
    g.add_nodes_from(list(range(m)))

    for i in range(m):
        for j in range(i + 1, m):
            u, v = odd_nodes[i], odd_nodes[j]
            w = dist[(u, v)]
            g.add_edge(i, j, -w)

    matching = rx.max_weight_matching(g)
    return [(odd_nodes[i], odd_nodes[j]) for i, j in matching]


def duplicate_edges_direct(G: nx.Graph, pairs: list, dist: dict) -> nx.MultiGraph:
    """
    Duplique dans un MultiGraph une SEULE arête directe (vol) entre u et v
    pour chaque paire issue du matching. La longueur de cette arête est donnée
    par dist[(u,v)] ou dist[(v,u)] (distance euclidienne).
    """
    G_euler = nx.MultiGraph(G)
    for u, v in pairs:
        # On cherche la distance en imposant l'ordre (u < v) dans le dict
        if (u, v) in dist:
            length_uv = dist[(u, v)]
        else:
            length_uv = dist[(v, u)]
        # Ajouter une seule arête directe entre u et v, avec le poids "length"
        G_euler.add_edge(u, v, length=length_uv)
    return G_euler


def extract_eulerian_path(G_euler: nx.MultiGraph) -> list:
    """
    Renvoie la liste des arêtes (u, v, key) dans l'ordre du chemin eulérien.
    """
    assert nx.is_eulerian(G_euler), "Le graphe n'est pas eulérien !"
    return list(nx.eulerian_path(G_euler))


def compute_survol_cost(G_und: nx.Graph, G_euler: nx.MultiGraph, path_edges: list):
    """
    G_und : graphe routier original (pour récupérer length de toute arête existante).
    G_euler : MultiGraph eulérien contenant duplications directes.
    path_edges : liste retournée par nx.eulerian_path(G_euler).
    """
    total_length_m = 0.0
    for edge in path_edges:
        # edge = (u, v) ou (u, v, key)
        if len(edge) == 2:
            u, v = edge
            if G_und.has_edge(u, v):
                data = G_und.get_edge_data(u, v)
                key0 = next(iter(data))
                total_length_m += data[key0]["length"]
            else:
                data_euler = G_euler.get_edge_data(u, v)
                key0 = next(iter(data_euler))
                total_length_m += data_euler[key0]["length"]
        else:
            u, v, k = edge
            data_euler = G_euler.get_edge_data(u, v, k)
            total_length_m += data_euler["length"]

    total_distance_km = total_length_m / 1000.0
    cost_fixed = 100.0
    cost_per_km = 0.01
    cost_variable = cost_per_km * total_distance_km
    total_cost = cost_fixed + cost_variable
    return total_distance_km, total_cost


def main():
    # 1) Chargement / simplification
    print("--- load and simplify ---")
    G_und = load_and_simplify(PLACE_NAME)
    print(f"Graph chargé : {len(G_und.nodes)} nœuds, {len(G_und.edges)} arêtes")
    plot_graph(G_und, show=False, savepath="montreal_simplified.png")

    # 2) Longueur totale de toutes les arêtes
    total_length_m = sum(
        data.get("length", 0.0) for _, _, data in G_und.edges(data=True)
    )
    total_length_km = total_length_m / 1000.0
    print(f"Longueur totale de toutes les arêtes : {total_length_km:.2f} km")

    # 3) Sommet impairs
    odd = [n for n, d in G_und.degree() if d % 2 == 1]
    print(f"{len(odd)} sommets impairs")

    # 4) Distances euclidiennes (vol direct) en mode batch
    print("-- computing euclidean distances batch --")
    dist = compute_pairwise_distances_euclid(G_und, odd, batch_size=500)
    print("Distances directes calculées")

    # 5) Matching minimal
    pairs = minimum_weight_matching(odd, dist)
    print(f"{len(pairs)} paires optimales calculées")

    # 6) Dupliquer une arête directe par paire
    G_euler = duplicate_edges_direct(G_und, pairs, dist)
    print(f"Graphe eulérien créé : {len(G_euler.nodes)} nœuds, {G_euler.number_of_edges()} arêtes")

    # 7) Extraction du chemin eulérien
    path_edges = extract_eulerian_path(G_euler)
    print(f"Chemin eulérien extrait : {len(path_edges)} arêtes au total")

    # 8) Coût du survol
    distance_km, cost = compute_survol_cost(G_und, G_euler, path_edges)
    print(f"Distance totale à parcourir : {distance_km:.2f} km")
    print(f"Coût total du survol : {cost:.2f} €")

    # 9) Préparer la liste de nœuds pour tracer
    nodes_route = [path_edges[0][0]]
    for edge in path_edges:
        nodes_route.append(edge[1])

    # 10) Affichage du chemin eulérien sur le graphe
    print("-- plotting eulerian path --")
    # a) On trace d’abord le graphe simplifié en fond
    fig, ax = ox.plot_graph(G_und, show=False, close=False)

    # b) On récupère les coordonnées x,y de chaque nœud dans l’ordre du chemin
    xs = [G_und.nodes[n]["x"] for n in nodes_route]
    ys = [G_und.nodes[n]["y"] for n in nodes_route]

    # c) On superpose le tracé : on relie chaque paire (n_i → n_{i+1}) par une ligne droite
    ax.plot(xs, ys, linewidth=2, color="r")

    # d) Affichage final
    plt.show()


if __name__ == "__main__":
    main()
