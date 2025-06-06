import sys
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# -----------------------------------------------
# Liste des quartiers
# -----------------------------------------------
district_names = [
    "Outremont, Montreal, Canada",
    "Verdun, Montreal, Canada",
    "Anjou, Montreal, Canada",
    "Rivière-des-Prairies–Pointe-aux-Trembles, Montreal, Canada",
    "Le Plateau-Mont-Royal, Montreal, Canada",
]


# -----------------------------------------------
# 1) Calcul du coût par machine (type I ou II)
# -----------------------------------------------
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

    dist_couverte = speed * max_hours  # km que la machine peut parcourir
    var_cost = cost_km * dist_couverte

    h_sup = max(0, max_hours - 8.0)
    h_base = min(max_hours, 8.0)
    hour_cost = hourly_1 * h_base + hourly_2 * h_sup

    return fixed + var_cost + hour_cost, dist_couverte


# -----------------------------------------------
# 2) Recherche de la flotte mixte optimale (n_I, n_II)
# -----------------------------------------------
def find_best_mixed_fleet(total_distance_km, max_hours, max_vehicles=10):
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


# -----------------------------------------------
# 3) Construction du circuit eulérien + découpage séquentiel
# -----------------------------------------------
def build_eulerian_routes(G, nI, nII, max_hours):
    """
    À partir d'un MultiDiGraph G, on :
     1) le convertit en MultiGraph non orienté U = G.to_undirected()
     2) le rend eulérien (nx.eulerize)
     3) récupère le circuit eulérien (liste d'arêtes successives)
     4) découpe ce circuit **séquentiellement** :
        – on affecte d'abord aux nI machines de type I environ (10*max_hours) km chacune
        – puis aux nII machines de type II environ (20*max_hours) km chacune
       afin que chaque segment du circuit soit utilisé exactement une fois.
    Renvoie deux listes : routes_I (nI listes d'arêtes), routes_II (nII listes d'arêtes).
    """
    # 3.1) Construire U = graphe non orienté, puis euleriser
    U = G.to_undirected()
    M = nx.eulerize(U)  # sans argument weight

    # 3.2) Récupérer la liste d'arêtes du circuit eulérien
    start = next(iter(M.nodes()))
    circuit = list(nx.eulerian_circuit(M, source=start))
    # circuit est une liste [(u0, v0), (u1, v1), ...], couvrant chaque arête (duplications incluses).

    # 3.3) Définir les capacités en km de chaque type
    cap_I = 10.0 * max_hours  # Type I → 10 km/h * max_hours
    cap_II = 20.0 * max_hours  # Type II → 20 km/h * max_hours

    # 3.4) On va consommer le circuit dans l'ordre, d'abord pour Type I, puis pour Type II
    routes_I = []
    routes_II = []

    idx = 0  # position courante dans "circuit"
    n_edges = len(circuit)

    # Fonction auxiliaire pour attribuer à une machine (Type I ou II) jusqu'à sa capacité
    def attribue_par_machine(capacité):
        nonlocal idx
        itinéraire = []
        accu = 0.0
        # On parcourt les arêtes du circuit à partir d'idx
        while idx < n_edges:
            u, v = circuit[idx]
            longueur_km = U[u][v][0]["length"] / 1000.0
            if accu + longueur_km > capacité:
                break
            itinéraire.append((u, v))
            accu += longueur_km
            idx += 1
        return itinéraire

    # 3.4.1) Donner aux nI machines de type I (capacité cap_I chacune)
    for _ in range(nI):
        route_I = attribue_par_machine(cap_I)
        routes_I.append(route_I)

    # 3.4.2) Ensuite, donner aux nII machines de type II (capacité cap_II chacune)
    for _ in range(nII):
        route_II = attribue_par_machine(cap_II)
        routes_II.append(route_II)

    # Note : Si idx < n_edges après avoir attribué I et II, cela signifie
    # que la somme des capacités est inférieure à la longueur totale du circuit.
    # Or par construction de find_best_mixed_fleet, la capacité totale ≥ distance réseau.
    # Donc, on doit toujours consommer tout le circuit avant d'arriver aux machines finales.
    return routes_I, routes_II


# -----------------------------------------------
# 4) Tracé manuel avec Matplotlib + affichage du temps
# -----------------------------------------------
def plot_network_and_route_with_time(G, route_edges, type_machine, title):
    """
    Trace le sous-graphe d'OSMnx G en gris clair, puis surcouche l'itinéraire
    donné sous forme de route_edges = [(u,v), ...] en couleur selon type_machine.
    Affiche en console la distance totale et le temps nécessaire.
    """
    U = G.to_undirected()

    # 4.1) Calculer la distance totale de cette route (en km)
    total_dist_km = 0.0
    for u, v in route_edges:
        longueur_m = U[u][v][0]["length"]
        total_dist_km += longueur_m / 1000.0

    # 4.2) Déterminer la vitesse selon le type (I = 10 km/h, II = 20 km/h)
    if type_machine == "I":
        speed = 10.0
        color = "blue"
    else:
        speed = 20.0
        color = "red"
    time_hours = total_dist_km / speed

    # 4.3) Afficher en console
    print(
        f"    • Distance de l'itinéraire Type {type_machine} : {total_dist_km:.2f} km"
    )
    print(f"      → Temps estimé à {speed:.0f} km/h : {time_hours:.2f} heures\n")

    # 4.4) Tracer l’arrière-plan (toutes les arêtes du réseau en gris clair)
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v, data in U.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, linewidth=0.4, color="lightgray", zorder=1)
        else:
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=0.4, color="lightgray", zorder=1)

    # 4.5) Tracer l’itinéraire de la machine en surcouche
    for u, v in route_edges:
        data = U.get_edge_data(u, v)
        if data is None:
            # si l'arête n'existe pas (rare), tracer un segment direct
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)
        else:
            attrib = data[0]  # choisir la première arête s’il y en a plusieurs
            if "geometry" in attrib:
                xs, ys = attrib["geometry"].xy
                ax.plot(xs, ys, linewidth=1.8, color=color, zorder=2)
            else:
                x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
                x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
                ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
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
            print("Usage: python3 test.py <heures_max> (strictement positif)")
            sys.exit(1)
    else:
        max_hours = 20.0

    for name in district_names:
        try:
            print(f"➡️ Traitement du district : {name}")
            # 1) Télécharger le graphe routier
            G = ox.graph_from_place(name, network_type="drive")

            # 2) Calculer la distance totale du réseau (en km)
            total_m = sum(d.get("length", 0) for u, v, d in G.edges(data=True))
            total_km = total_m / 1000.0

            # 3) Trouver la flotte optimale et son coût
            best_cost, best_nI, best_nII = find_best_mixed_fleet(
                total_km, max_hours, max_vehicles=10
            )
            if best_nI is None:
                print(
                    f"  ❌ Impossible de couvrir {total_km:.2f} km avec ≤10 machines en {max_hours:.1f} h.\n"
                )
                continue

            print(f"  • Distance du réseau : {total_km:.2f} km")
            print(
                f"  • Flotte optimale → {best_nI} machine(s) Type I + {best_nII} machine(s) Type II"
            )
            print(f"    Coût global estimé = {best_cost:.2f} $\n")

            # 4) Générer les itinéraires (découpage séquentiel)
            routes_I, routes_II = build_eulerian_routes(G, best_nI, best_nII, max_hours)

            # 5) Pour chaque machine Type I, afficher distance/temps, puis tracer
            if best_nI > 0:
                for idx, r in enumerate(routes_I, start=1):
                    print(f"  ── Machine I#{idx} ──")
                    plot_network_and_route_with_time(
                        G,
                        r,
                        type_machine="I",
                        title=f"Itinéraire Machine I#{idx} → {name}",
                    )

            # 6) Pour chaque machine Type II, afficher distance/temps, puis tracer
            if best_nII > 0:
                for idx, r in enumerate(routes_II, start=1):
                    print(f"  ── Machine II#{idx} ──")
                    plot_network_and_route_with_time(
                        G,
                        r,
                        type_machine="II",
                        title=f"Itinéraire Machine II#{idx} → {name}",
                    )

            print("\n")

        except Exception as e:
            print(f"❌ Erreur pour {name} : {e}\n")


if __name__ == "__main__":
    main()
