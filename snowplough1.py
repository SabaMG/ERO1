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

def cost_per_machine_time(type_, T):
    """
    Pour une machine de type I ou II qui travaille T heures,
    renvoie (coût_total, capacité_km), où capacité_km = vitesse×T.
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

    dist_couverte = speed * T
    var_cost = cost_km * dist_couverte

    h_sup = max(0, T - 8.0)
    h_base = min(T, 8.0)
    hour_cost = hourly_1 * h_base + hourly_2 * h_sup

    return fixed + var_cost + hour_cost, dist_couverte

def find_fastest_fleet(total_distance_km, budget, max_vehicles=10):
    """
    Pour un réseau de distance D = total_distance_km, et un budget donné,
    on parcourt tous les couples (nI, nII) avec nI+nII ≤ max_vehicles (et nI+nII > 0).
    Pour chaque couple, on calcule :
      - T = D / (10·nI + 20·nII)
      - coût_total = nI·C_I(T) + nII·C_II(T)
    On ne retient que ceux dont coût_total ≤ budget,
    puis on choisit le couple qui donne le plus petit T.
    Renvoie (best_nI, best_nII, best_T, best_cost) ou (None, None, None, None)
    si aucune solution ne respecte le budget.
    """
    best_nI = best_nII = None
    best_T = float("inf")
    best_cost = None

    for nI in range(0, max_vehicles + 1):
        for nII in range(0, max_vehicles + 1):
            if nI == 0 and nII == 0:
                continue
            combined_speed = 10.0 * nI + 20.0 * nII
            if combined_speed <= 0:
                continue

            T = total_distance_km / combined_speed

            # Coût total pour ce T
            cost_I, _ = cost_per_machine_time("I", T)
            cost_II, _ = cost_per_machine_time("II", T)
            total_cost = nI * cost_I + nII * cost_II

            if total_cost <= budget:
                if T < best_T:
                    best_T = T
                    best_nI = nI
                    best_nII = nII
                    best_cost = total_cost

    if best_nI is None:
        return None, None, None, None
    return best_nI, best_nII, best_T, best_cost

def build_eulerian_routes(G, nI, nII, T):
    """
    1) Construire U = G.to_undirected() puis eulériser U (nx.eulerize)
    2) Extraire circuit non orienté, puis reconstituer circuit_oriented
       (en choisissant la bonne orientation dans G).
    3) Chaque machine de type I peut couvrir cap_I = 10·T km,
       chaque machine de type II peut couvrir cap_II = 20·T km.
    4) On répartit le circuit total proportionnellement à ces capacités
       (aucune machine n'est laissée vide sauf si nI+nII > nécessaire).
    5) On renvoie routes_I, routes_II et le nombre effectif de machines non vides.
    """
    U = G.to_undirected()
    M = nx.eulerize(U)

    start = next(iter(M.nodes()))
    tout_circuit_undirected = list(nx.eulerian_circuit(M, source=start))

    circuit_oriented = []
    for u, v in tout_circuit_undirected:
        if G.has_edge(u, v):
            circuit_oriented.append((u, v))
        else:
            circuit_oriented.append((v, u))

    arc_lengths_km = []
    for u, v in circuit_oriented:
        data_list = G.get_edge_data(u, v)
        if data_list is None:
            length_m = 0
        else:
            attr = data_list[next(iter(data_list))]
            length_m = attr.get("length", 0)
        arc_lengths_km.append(length_m / 1000.0)
    total_circuit_km = sum(arc_lengths_km)

    cap_I = 10.0 * T
    cap_II = 20.0 * T

    capacities = [cap_I]*nI + [cap_II]*nII
    num_vehicles = nI + nII
    if num_vehicles == 0:
        return [], [], 0, 0

    total_cap = sum(capacities)

    thresholds = []
    cum = 0.0
    for c in capacities:
        thresholds.append(cum)
        cum += (c / total_cap) * total_circuit_km
    thresholds.append(total_circuit_km)

    routes = [[] for _ in range(num_vehicles)]
    cum_km = 0.0
    current_vehicle = 0

    for idx, (u, v) in enumerate(circuit_oriented):
        l_km = arc_lengths_km[idx]

        while (
            current_vehicle < num_vehicles - 1
            and cum_km + l_km > thresholds[current_vehicle + 1]
            and routes[current_vehicle]
        ):
            current_vehicle += 1

        routes[current_vehicle].append((u, v))
        cum_km += l_km

    # --- 3.9) Séparer les routes Type I et Type II, puis retirer les machines vides ---
    routes_I = [r for r in routes[:nI] if r]
    routes_II = [r for r in routes[nI:] if r]

    used_nI = len(routes_I)
    used_nII = len(routes_II)

    return routes_I, routes_II, used_nI, used_nII


# -----------------------------------------------
# 4) Tracé manuel avec Matplotlib + affichage du temps
# -----------------------------------------------
def plot_network_and_route_with_time(G, route_edges, type_machine, title, save_path=None):
    """
    Trace le réseau (U en gris clair) + l’itinéraire orienté en bleu (I) ou rouge (II),
    affiche distance (km) et temps (h).
    """
    U = G.to_undirected()

    total_dist_km = 0.0
    for u, v in route_edges:
        data_list = G.get_edge_data(u, v)
        if data_list is None:
            continue
        attr = data_list[next(iter(data_list))]
        length_m = attr.get("length", 0)
        total_dist_km += length_m / 1000.0

    speed = 10.0 if type_machine == "I" else 20.0
    color = "blue" if type_machine == "I" else "red"
    time_h = total_dist_km / speed

    print(f"    • Distance Type {type_machine}: {total_dist_km:.2f} km")
    print(f"      → Temps estimé ≃ {time_h:.2f} h\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    for x, y, data in U.edges(data=True):
        if data.get("length", 0) == 0:
            continue
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, linewidth=0.4, color="lightgray", zorder=1)
        else:
            x1, y1 = U.nodes[x]["x"], U.nodes[x]["y"]
            x2, y2 = U.nodes[y]["x"], U.nodes[y]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=0.4, color="lightgray", zorder=1)

    for u, v in route_edges:
        data_list = G.get_edge_data(u, v)
        if data_list is None:
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)
            continue
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 snowplough.py <budget> [<max_vehicles>]")
        sys.exit(1)

    try:
        budget = float(sys.argv[1])
        if budget < 0:
            raise ValueError
    except ValueError:
        print("Budget doit être un nombre positif.")
        sys.exit(1)

    max_vehicles = 10
    if len(sys.argv) >= 3:
        try:
            max_vehicles = int(sys.argv[2])
            if max_vehicles < 1:
                raise ValueError
        except ValueError:
            print("max_vehicles doit être un entier ≥ 1.")
            sys.exit(1)

    print(f"=== Budget total : {budget:.2f} $ — Max machines trialé : {max_vehicles} ===\n")

    os.makedirs("plots_solutions1", exist_ok=True)

    for name in district_names:
        try:
            print(f"Traitement du district : {name}")
            start = time.time()
            end = None

            G = ox.graph_from_place(name, network_type="drive")

            total_m = sum(d.get("length", 0) for u, v, d in G.edges(data=True))
            total_km = total_m / 1000.0

            best_nI, best_nII, best_T, best_cost = find_fastest_fleet(
                total_km, budget, max_vehicles=max_vehicles
            )
            if best_nI is None:
                print(f"    Impossible de respecter le budget de {budget:.2f}$.")
                print(f"    Même 1 machine Type I coûte {cost_per_machine_time('I', 1)[0]:.2f}$ pour 1h, etc.\n")
                continue

            print(f"  • Distance du réseau : {total_km:.2f} km")
            print(f"  • Flotte choisie   → {best_nI} machine(s) Type I + {best_nII} machine(s) Type II")
            print(f"  • Temps estimé     → {best_T:.2f} heures (épuisement simultané)")
            print(f"  • Coût total final → {best_cost:.2f} $\n")

            routes_I, routes_II, used_nI, used_nII = build_eulerian_routes(
                G, best_nI, best_nII, best_T
            )

            if used_nI < best_nI or used_nII < best_nII:
                print(
                    f"    Machines vides supprimées : "
                    f"initialement {best_nI}×I + {best_nII}×II → "
                    f"{used_nI}×I + {used_nII}×II\n"
                )

            if used_nI > 0:
                for idx, r in enumerate(routes_I, start=1):
                    print(f"  ── Machine I#{idx} ──")
                    if (end == None):
                        end = time.time()
                    plot_network_and_route_with_time(
                        G, r, type_machine="I",
                        title=f"Itinéraire I#{idx} → {name}",
                        save_path=f"plots_solutions1/{name.replace(',', '').replace(' ', '_')}_I{idx}.png"
                    )
            if used_nII > 0:
                for idx, r in enumerate(routes_II, start=1):
                    print(f"  ── Machine II#{idx} ──")
                    if (end == None):
                        end = time.time()
                    plot_network_and_route_with_time(
                        G, r, type_machine="II",
                        title=f"Itinéraire II#{idx} → {name}",
                        save_path=f"plots_solutions1/{name.replace(',', '').replace(' ', '_')}_II{idx}.png"
                    )
                    
            print(f"Temps total pour {name} : {end - start:.2f} secondes\n")

        except Exception as e:
            print(f"Erreur pour {name} : {e}\n")


if __name__ == "__main__":
    main()