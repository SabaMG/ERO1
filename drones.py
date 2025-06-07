import os
import time
import osmnx as ox
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

PLACE_NAME = "Montréal, Québec, Canada"
DIRECTED_FILE = "montreal_directed.graphml"
UNDIRECTED_FILE = "montreal_undirected.graphml"
NUM_DRONES = 4
BATCH_SIZE = 500
COST_FIXED = 100.0    # € / jour
COST_PER_KM = 0.01    # €/km
DRONE_SPEED = 50 # km/h

# --------------------------------------------------------------------------
# 1) CHARGEMENT DU GRAPHE
# --------------------------------------------------------------------------
def load_and_simplify(place_name: str) -> nx.Graph:
    if os.path.exists(UNDIRECTED_FILE):
        print(f"⇨ Chargement du graphe non orienté depuis {UNDIRECTED_FILE}")
        return ox.load_graphml(UNDIRECTED_FILE)
    if os.path.exists(DIRECTED_FILE):
        print(f"⇨ Chargement du graphe orienté depuis {DIRECTED_FILE}")
        G = ox.load_graphml(DIRECTED_FILE)
    else:
        print("⇨ Téléchargement du graphe via OSMnx")
        G = ox.graph_from_place(place_name, network_type="drive")
        ox.save_graphml(G, filepath=DIRECTED_FILE)
    G_und = G.to_undirected()
    print(f"⇨ Sauvegarde du graphe non orienté dans {UNDIRECTED_FILE}")
    ox.save_graphml(G_und, filepath=UNDIRECTED_FILE)
    return G_und

# --------------------------------------------------------------------------
# 2) PARTITION SPATIALE DU GRAPHE
# --------------------------------------------------------------------------
def partition_graph(G: nx.Graph, num_clusters: int):
    G_proj = ox.project_graph(G)
    nodes = list(G_proj.nodes)
    coords = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']] for n in nodes])
    labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(coords)
    subgraphs = []
    for k in range(num_clusters):
        nodes_k = [nodes[i] for i, lab in enumerate(labels) if lab == k]
        subgraphs.append(G.subgraph(nodes_k).copy())
    return subgraphs

# --------------------------------------------------------------------------
# 3) DISTANCES EUCLIDIENNES BATCH
# --------------------------------------------------------------------------
def compute_pairwise_distances_euclid(G: nx.Graph, odd_nodes: list, batch_size: int = BATCH_SIZE) -> dict:
    G_proj = ox.project_graph(G)
    M = len(odd_nodes)
    xs = np.zeros(M)
    ys = np.zeros(M)
    for i, n in enumerate(odd_nodes):
        xs[i] = G_proj.nodes[n]['x']
        ys[i] = G_proj.nodes[n]['y']
    dist = {}
    n_batches = (M + batch_size - 1) // batch_size
    for bi in range(n_batches):
        i0 = bi * batch_size
        i1 = min(i0 + batch_size, M)
        xi, yi = xs[i0:i1], ys[i0:i1]
        for bj in range(bi, n_batches):
            j0 = bj * batch_size
            j1 = min(j0 + batch_size, M)
            xj, yj = xs[j0:j1], ys[j0:j1]
            dx = xi[:, None] - xj[None, :]
            dy = yi[:, None] - yj[None, :]
            D = np.hypot(dx, dy)
            for ii in range(i1 - i0):
                u = odd_nodes[i0 + ii]
                if bi == bj:
                    for jj in range(ii + 1, j1 - j0):
                        v = odd_nodes[j0 + jj]
                        dist[(u, v)] = float(D[ii, jj])/10.0
                else:
                    for jj in range(j1 - j0):
                        uu = odd_nodes[i0 + ii]
                        vv = odd_nodes[j0 + jj]
                        dist[(uu, vv)] = float(D[ii, jj])/10.0
        print(f"batch {bi+1}/{n_batches} done")
    return dist

# --------------------------------------------------------------------------
# 4) MATCHING BLOSSOM
# --------------------------------------------------------------------------
def minimum_weight_matching(odd_nodes: list, dist: dict) -> list:
    m = len(odd_nodes)
    g = rx.PyGraph()
    g.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i+1, m):
            w = dist[(odd_nodes[i], odd_nodes[j])]
            g.add_edge(i, j, w)
    matching = rx.max_weight_matching(g)
    return [(odd_nodes[i], odd_nodes[j]) for i, j in matching]

# --------------------------------------------------------------------------
# 5) DUPLICATION DES ARÊTES DIRECTES
# --------------------------------------------------------------------------
def duplicate_edges_direct(G: nx.Graph, pairs: list, dist: dict) -> nx.MultiGraph:
    G_euler = nx.MultiGraph(G)
    for u, v in pairs:
        length_uv = dist.get((u, v), dist.get((v, u)))
        G_euler.add_edge(u, v, length=length_uv)
    return G_euler

# --------------------------------------------------------------------------
# 6) EXTRACTION DU CHEMIN EULÉRIEN
# --------------------------------------------------------------------------
def extract_eulerian_path(G_euler: nx.MultiGraph) -> list:
    #assert nx.is_eulerian(G_euler), "Le graphe n'est pas eulérien !"
    return list(nx.eulerian_path(G_euler))

# --------------------------------------------------------------------------
# 7) CALCUL DU COÛT
# --------------------------------------------------------------------------
def compute_survol_cost(G_und, G_euler, path_edges):
    total_m = 0.0
    for edge in path_edges:
        if len(edge) == 2:
            u, v = edge
            if G_und.has_edge(u, v):
                data = G_und.get_edge_data(u, v)
                total_m += data[next(iter(data))]['length']
            else:
                data = G_euler.get_edge_data(u, v)
                total_m += data[next(iter(data))]['length']
        else:
            u, v, k = edge
            total_m += G_euler.get_edge_data(u, v, k)['length']
    km = total_m / 1000.0
    cost = COST_FIXED + COST_PER_KM * km
    return km, cost

def connect_components(sub: nx.Graph) -> None:
    # projeté pour coords métriques
    Gp = ox.project_graph(sub)
    comps = list(nx.connected_components(sub))
    # pour chaque paire de composantes consécutives trouver plus proches
    for k in range(len(comps)-1):
        A, B = comps[k], comps[k+1]
        best = None
        best_dist = float('inf')
        for u in A:
            x_u, y_u = Gp.nodes[u]['x'], Gp.nodes[u]['y']
            for v in B:
                x_v, y_v = Gp.nodes[v]['x'], Gp.nodes[v]['y']
                d = ((x_u-x_v)**2 + (y_u-y_v)**2)**0.5
                if d < best_dist:
                    best_dist, best = (d, (u, v))
        u, v = best
        sub.add_edge(u, v, length=best_dist)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    G_und = load_and_simplify(PLACE_NAME)
    print(f"Graph chargé : {len(G_und.nodes)} nœuds, {len(G_und.edges)} arêtes")
    subgraphs = partition_graph(G_und, NUM_DRONES)
    # préparer le fond projeté et la colormap
    G_proj = ox.project_graph(G_und)
    fig, ax = ox.plot_graph(G_proj, show=False, close=False)
    cmap = plt.cm.get_cmap('tab10', NUM_DRONES)

    dists = []
    total_compute_time = 0
    total_km = 0.0
    total_cost = 0.0
    for i, sub in enumerate(subgraphs):
        print(f"\n--- Drone {i+1}/{NUM_DRONES} sur {len(sub.nodes)} nœuds ---")
        start_time = time.time()

        n_comp = nx.number_connected_components(sub)
        connect_components(sub)
        print("Connected subgraph components")
        odd = [n for n, d in sub.degree() if d % 2 == 1]
        print(f"Sommets impairs: {len(odd)}")
        if sub.number_of_nodes() < 2:
            print(" trop petit, skip.")
            continue
        print("Computing distances...")
        dist = compute_pairwise_distances_euclid(sub, odd)
        print("Minimum weight matching...")
        pairs = minimum_weight_matching(odd, dist)
        print(f"Paires appariées: {len(pairs)}")
        G_euler = duplicate_edges_direct(sub, pairs, dist)
        path = extract_eulerian_path(G_euler)

        elapsed = time.time() - start_time
        total_compute_time += elapsed
        print(f"Temps de calcul itinéraire drone {i+1}: {elapsed:.2f}s")

        km, cost = compute_survol_cost(sub, G_euler, path)
        dists.append(km)
        total_km += km
        total_cost += cost
        print(f"Distance circuit: {km:.2f} km, coût: {cost:.2f} €")

        nodes_route = [path[0][0]] + [e[1] for e in path]
        xs = [G_proj.nodes[n]['x'] for n in nodes_route]
        ys = [G_proj.nodes[n]['y'] for n in nodes_route]
        ax.plot(xs, ys, linewidth=1.0, color=cmap(i), label=f"Drone {i+1}")

    print(f"\n=== Résumé global ===")
    print(f"Distance totale (tous drones) : {total_km:.2f} km")
    print(f"Coût total (tous drones) : {total_cost:.2f} €")

    travel_time = max(dists) / DRONE_SPEED
    print(f"Temps de reconnaissance pour un drone allant à {DRONE_SPEED}km/h : {travel_time:.2f}h ")
    print(f"Temps de calcul total {total_compute_time:.2f}s")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
