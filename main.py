import os
import osmnx as ox
import networkx as nx
import rustworkx as rx
import numpy as np
import matplotlib.pyplot as plt
from osmnx.truncate import truncate_graph_polygon

PLACE_NAME = "Montréal, Québec, Canada"
NETWORK_TYPE = "drive"
BATCH_SIZE = 500

def compute_pairwise_distances_euclid(G: nx.Graph,
                                      odd_nodes: list,
                                      batch_size: int = BATCH_SIZE) -> dict:
    """
    Pour chaque paire u<v de odd_nodes, calcule la
    distance euclidienne plane (dans un CRS métrique)
    entre u et v, en lots pour limiter la mémoire.
    Retourne {(u,v): distance_en_mètres}.
    """
    G_proj = ox.project_graph(G)
    M = len(odd_nodes)
    xs = np.zeros(M, dtype=np.float64)
    ys = np.zeros(M, dtype=np.float64)
    for i, n in enumerate(odd_nodes):
        xs[i] = G_proj.nodes[n]["x"]
        ys[i] = G_proj.nodes[n]["y"]

    dist = {}
    n_batches = (M + batch_size - 1) // batch_size
    for bi in range(n_batches):
        i0, i1 = bi * batch_size, min((bi + 1) * batch_size, M)
        x_i, y_i = xs[i0:i1], ys[i0:i1]
        for bj in range(bi, n_batches):
            j0, j1 = bj * batch_size, min((bj + 1) * batch_size, M)
            x_j, y_j = xs[j0:j1], ys[j0:j1]

            dx = x_i[:, None] - x_j[None, :]
            dy = y_i[:, None] - y_j[None, :]
            D  = np.hypot(dx, dy)

            for ii in range(i1 - i0):
                u = odd_nodes[i0 + ii]
                if bi == bj:
                    for jj in range(ii + 1, j1 - j0):
                        v = odd_nodes[j0 + jj]
                        dist[(u, v)] = float(D[ii, jj])
                else:
                    for jj in range(j1 - j0):
                        uu = odd_nodes[i0 + ii]
                        vv = odd_nodes[j0 + jj]
                        dist[(uu, vv)] = float(D[ii, jj])
        print(f"batch {bi+1}/{n_batches} done")
    return dist

def minimum_weight_matching(odd_nodes: list, dist: dict) -> list:
    """
    Trouve, via retworkx Blossom, l'appariement parfait
    minimal sur le graphe complet des odd_nodes pondéré
    par dist[(u,v)].
    """
    m = len(odd_nodes)
    g = rx.PyGraph()
    g.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i+1, m):
            w = dist[(odd_nodes[i], odd_nodes[j])]
            g.add_edge(i, j, -w)
    matching = rx.max_weight_matching(g)
    return [(odd_nodes[i], odd_nodes[j]) for i, j in matching]

def duplicate_edges_direct(G: nx.Graph, pairs: list, dist: dict) -> nx.MultiGraph:
    """
    Pour chaque paire (u,v), ajoute UNE arête directe de longueur dist[(u,v)].
    """
    G_euler = nx.MultiGraph(G)
    for u, v in pairs:
        length_uv = dist.get((u, v), dist.get((v, u)))
        G_euler.add_edge(u, v, length=length_uv)
    return G_euler

def extract_eulerian_path(G_euler: nx.MultiGraph) -> list:
    """
    Extrait un chemin eulérien (ouvert ou fermé) du graphe eulérien.
    """
    assert nx.is_eulerian(G_euler), "Le graphe n'est pas eulérien !"
    return list(nx.eulerian_path(G_euler))

def main():
    # 1) Charger le graphe routier complet de Montréal
    print("Loading full street network for Montréal…")
    G_full = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE)

    # 2) Récupérer les limites de chaque arrondissement ("borough")
    print("Fetching borough boundaries…")
    city = ox.geocode_to_gdf(PLACE_NAME)
    tags = {"boundary": "administrative", "admin_level": "8", "name": True}
    geo = ox.features_from_place(PLACE_NAME, tags)
    print("features extracted")
    boroughs = geo[geo.geom_type.isin(["Polygon","MultiPolygon"])].copy()
    boroughs = boroughs.to_crs(city.crs)
    boroughs = boroughs[boroughs.geometry.intersects(city.geometry.iloc[0])]

    # 3) Pour chaque quartier, calculer le circuit eulérien
    routes, names = [], []
    for idx, row in boroughs.iterrows():
        name = row["name"]
        print(f"> Processing borough: {name}")
        poly = row.geometry
        try:
            # on utilise truncate_by_edge=True comme indiqué dans la doc
            G_sub = truncate_graph_polygon(G_full, poly, truncate_by_edge=True)
        except ValueError:
            # si aucun nœud ne tombe dans le polygone, on passe au suivant
            print(f"  >> no street nodes in '{name}', skipping")
            continue
        if len(G_sub.nodes) == 0:
            # double-sécurité si la fonction renvoie un graphe vide
            print(f"  >> empty subgraph for '{name}', skipping")
            continue
        odd = [n for n,d in G_sub.degree() if d % 2 == 1]
        print("\t• computing pairwise distances")
        dist = compute_pairwise_distances_euclid(G_sub, odd, batch_size=BATCH_SIZE)
        print("\t• minimum weight matching")
        pairs = minimum_weight_matching(odd, dist)
        print("\t• duplicating edges")
        G_euler = duplicate_edges_direct(G_sub, pairs, dist)
        print("\t• extracting eulerian path")
        path = extract_eulerian_path(G_euler)

        # convertir en séquence de nœuds [u0, u1, …]
        route = [path[0][0]] + [v for u,v,*_ in path]
        routes.append(route)
        names.append(name)

    # 4) Tracer le tout en couleurs distinctes
    print("Plotting all borough routes…")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(routes))]
    fig, ax = ox.plot_graph_routes(
        G_full,
        routes,
        route_colors=colors,
        route_linewidth=2,
        orig_dest_size=0,
        figsize=(12,12),
        show=False
    )
    # légende
    for color, name in zip(colors, names):
        ax.plot([], [], color=color, label=name)
    ax.legend(fontsize="small", loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
