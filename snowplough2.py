import sys
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import time
import os

district_names = [
    "Outremont, Montreal, Canada",
    "Verdun, Montreal, Canada",
    "Anjou, Montreal, Canada",
    "Rivière-des-Prairies–Pointe-aux-Trembles, Montreal, Canada",
    "Le Plateau-Mont-Royal, Montreal, Canada",
]

def cost_per_machine(type_, max_hours):
    """
    Renvoie (coût_total, capacité_km) pour une machine de type I ou II
    qui tourne exactement max_hours heures.
    """
    if type_ == "I":
        fixed = 500
        cost_km = 1.1
        hourly_1 = 1.1
        hourly_2 = 1.3
        speed = 10.0
    else:
        fixed = 800
        cost_km = 1.3
        hourly_1 = 1.3
        hourly_2 = 1.5
        speed = 20.0

    dist_couverte = speed * max_hours
    var_cost = cost_km * dist_couverte

    h_sup = max(0, max_hours - 8.0)
    h_base = min(max_hours, 8.0)
    hour_cost = hourly_1 * h_base + hourly_2 * h_sup

    return fixed + var_cost + hour_cost, dist_couverte

def find_best_mixed_fleet(total_distance_km, max_hours, max_vehicles=5):
    """
    Cherche (nI, nII) ≤ max_vehicles pour couvrir total_distance_km
    avec des machines de capacité respective cap_I et cap_II,
    et minimise le coût total.
    Renvoie (best_cost, best_nI, best_nII) ou (None, None, None) si impossible.
    """
    C_I, cap_I = cost_per_machine("I", max_hours)
    C_II, cap_II = cost_per_machine("II", max_hours)

    best_cost = float("inf")
    best_nI, best_nII = None, None

    for nI in range(0, max_vehicles + 1):
        for nII in range(0, max_vehicles + 1):
            if nI == 0 and nII == 0:
                continue
            capacité_tot = nI * cap_I + nII * cap_II
            if capacité_tot < total_distance_km:
                continue
            coût = nI * C_I + nII * C_II
            if coût < best_cost:
                best_cost = coût
                best_nI, best_nII = nI, nII

    if best_nI is None:
        return None, None, None
    return best_cost, best_nI, best_nII

def build_eulerian_routes(G, nI, nII, max_hours):
    """
    À partir d'un MultiDiGraph orienté G, on :
      1) copie G en non orienté U
      2) eulérise U (nx.eulerize)
      3) récupère le circuit eulérien sur U
      4) pour chaque arête (u,v) du circuit non orienté, on choisit la bonne orientation
         en se basant sur G : si G contient (u->v), on l'utilise ; sinon, on prend (v->u).
      5) découpe ce circuit orienté reconstitué en nI machines Type I (10 km/h) puis
         nII machines Type II (20 km/h) de manière séquentielle + greedy fallback
    Renvoie deux listes : routes_I (listes d'arêtes orientées), routes_II (listes d'arêtes).
    """
    U = G.to_undirected()
    M = nx.eulerize(U)

    start = next(iter(M.nodes()))
    tout_circuit_undirected = list(nx.eulerian_circuit(M, source=start))

    circuit_oriented = []
    for u, v in tout_circuit_undirected:
        if G.has_edge(u, v):
            circuit_oriented.append((u, v))
        elif G.has_edge(v, u):
            circuit_oriented.append((v, u))
        else:
            raise RuntimeError(f"Aucune orientation trouvée pour l'arête non orientée {u, v}")

    cap_I = 10.0 * max_hours   # Type I = 10 km/h × max_hours
    cap_II = 20.0 * max_hours  # Type II = 20 km/h × max_hours

    # 3.6) Découpage séquentiel du circuit orienté reconstitué
    routes_I = []
    routes_II = []
    idx = 0
    n_arcs = len(circuit_oriented)

    def attribue_par_machine(capacité):
        nonlocal idx
        itinéraire = []
        accu = 0.0
        while idx < n_arcs:
            u, v = circuit_oriented[idx]
            # Trouver la longueur dans G (en km)
            data_list = G.get_edge_data(u, v)
            if data_list is None:
                longueur_km = 0.0
            else:
                # S'il y a plusieurs clés, on prend la première
                attr = data_list[next(iter(data_list))]
                longueur_km = attr.get("length", 0) / 1000.0

            if accu + longueur_km > capacité:
                # Si c’est la première arête, on la prend malgré tout (fallback)
                if not itinéraire:
                    itinéraire.append((u, v))
                    accu += longueur_km
                    idx += 1
                break
            itinéraire.append((u, v))
            accu += longueur_km
            idx += 1
        return itinéraire

    # 3.6.1) Répartir sur les nI machines Type I
    for _ in range(nI):
        routes_I.append(attribue_par_machine(cap_I))

    # 3.6.2) Puis répartir sur les nII machines Type II
    for _ in range(nII):
        routes_II.append(attribue_par_machine(cap_II))

    return routes_I, routes_II


# -----------------------------------------------
# 4) Tracé manuel avec Matplotlib + affichage du temps
# -----------------------------------------------
def plot_network_and_route_with_time(G, route_edges, type_machine, title, save_path=None):
    """
    Trace le sous-graphe d'OSMnx G en gris clair, puis surcouche l'itinéraire
    donné sous forme de route_edges = [(u,v), ...] en couleur selon type_machine.
    Affiche en console la distance totale (en km) et le temps nécessaire (en h).
    """
    U = G.to_undirected()

    # 4.1) Calculer la distance totale (en km) en ignorant length = 0
    total_dist_km = 0.0
    for u, v in route_edges:
        data_list = G.get_edge_data(u, v)
        if data_list is None:
            continue
        attr = data_list[next(iter(data_list))]
        longueur_m = attr.get("length", 0)
        total_dist_km += longueur_m / 1000.0

    # 4.2) Choisir la vitesse et la couleur
    if type_machine == "I":
        speed = 10.0
        color = "blue"
    else:
        speed = 20.0
        color = "red"
    time_hours = total_dist_km / speed

    # 4.3) Afficher en console
    print(f"    • Distance de l'itinéraire Type {type_machine} : {total_dist_km:.2f} km")
    print(f"      → Temps estimé à {speed:.0f} km/h : {time_hours:.2f} heures\n")

    # 4.4) Tracer l’arrière-plan (arêtes de U en gris léger)
    fig, ax = plt.subplots(figsize=(8, 8))
    for x, y, data in U.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, linewidth=0.4, color="lightgray", zorder=1)
        else:
            x1, y1 = U.nodes[x]["x"], U.nodes[x]["y"]
            x2, y2 = U.nodes[y]["x"], U.nodes[y]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=0.4, color="lightgray", zorder=1)

    # 4.5) Tracer l’itinéraire orienté en surcouche
    for u, v in route_edges:
        data_list = G.get_edge_data(u, v)
        if data_list is None:
            # En principe, ça ne devrait pas arriver
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)
        else:
            attr = data_list[next(iter(data_list))]
            if "geometry" in attr:
                xs, ys = attr["geometry"].xy
                ax.plot(xs, ys, linewidth=1.8, color=color, zorder=2)
            else:
                x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
                x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
                ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# -----------------------------------------------
# 5) Fonction principale
# -----------------------------------------------
def main():
    # Lecture du temps max (défaut 20 h)
    if len(sys.argv) >= 2:
        try:
            max_hours = float(sys.argv[1])
            if max_hours <= 0:
                raise ValueError
        except ValueError:
            print("Usage: python3 snowplough.py <heures_max> (strictement positif)")
            sys.exit(1)
    else:
        max_hours = 20.0

    os.makedirs("plots_solutions2", exist_ok=True)
    
    for name in district_names:
        try:
            print(f"Traitement du district : {name}")
            start = time.time()
            end = None
            G = ox.graph_from_place(name, network_type="drive")

            total_m = sum(d.get("length", 0) for u, v, d in G.edges(data=True))
            total_km = total_m / 1000.0

            best_cost, best_nI, best_nII = find_best_mixed_fleet(
                total_km, max_hours, max_vehicles=10
            )
            if best_nI is None:
                print(
                    f"  Impossible de couvrir {total_km:.2f} km avec ≤10 machines en {max_hours:.1f} h.\n"
                )
                continue

            print(f"  • Distance du réseau : {total_km:.2f} km")
            print(
                f"  • Flotte optimale → {best_nI} machine(s) Type I + {best_nII} machine(s) Type II"
            )
            print(f"    Coût global estimé = {best_cost:.2f} $\n")

            routes_I, routes_II = build_eulerian_routes(G, best_nI, best_nII, max_hours)

            if best_nI > 0:
                for idx, r in enumerate(routes_I, start=1):
                    print(f"  ── Machine I#{idx} ──")
                    if (end == None):
                        end = time.time()
                    plot_network_and_route_with_time(
                        G, r, type_machine="I",
                        title=f"Itinéraire I#{idx} → {name}",
                        save_path=f"plots_solutions2/{name.replace(',', '').replace(' ', '_')}_I{idx}.png"
                    )

            if best_nII > 0:
                for idx, r in enumerate(routes_II, start=1):
                    print(f"  ── Machine II#{idx} ──")
                    if (end == None):
                        end = time.time()
                    plot_network_and_route_with_time(
                        G, r, type_machine="II",
                        title=f"Itinéraire II#{idx} → {name}",
                        save_path=f"plots_solutions2/{name.replace(',', '').replace(' ', '_')}_II{idx}.png"
                    )

            print(f"Temps total pour {name} : {end - start:.2f} secondes\n")

        except Exception as e:
            print(f"Erreur pour {name} : {e}\n")


if __name__ == "__main__":
    main()