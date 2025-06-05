import sys
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

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
    avec des machines de capacité respectives cap_I et cap_II,
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
# 3) Construction du circuit eulérien + découpage
# -----------------------------------------------
def build_eulerian_routes(G, nI, nII, max_hours):
    """
    À partir d'un MultiDiGraph G, on :
     1) le convertit en MultiGraph non orienté U = G.to_undirected()
     2) le rend eulérien (nx.eulerize)
     3) récupère le circuit eulérien (liste d'arêtes successives)
     4) découpe ce circuit en nI itinéraires Type I et nII itinéraires Type II,
        chacun conforme à sa capacité (10*max_hours pour Type I, 20*max_hours pour Type II).
    Renvoie deux listes : routes_I (nI listes d'arêtes), routes_II (nII listes d'arêtes).
    """
    # Convertir en non orienté
    U = G.to_undirected()
    # Eulerisation (sans argument weight) pour rendre U eulérien
    M = nx.eulerize(U)

    # Calcul du circuit eulérien (séquence d'arêtes (u, v))
    start = next(iter(M.nodes()))
    circuit = list(nx.eulerian_circuit(M, source=start))

    cap_I = 10.0 * max_hours  # km max pour une machine Type I
    cap_II = 20.0 * max_hours  # km max pour une machine Type II

    def découpe(circuit, cap, nmach):
        """
        Découpe la liste d'arêtes 'circuit' en 'nmach' sous-listes,
        chaque sous-liste ne dépassant pas 'cap' km cumulés (en sommant U[u][v][0]['length'] / 1000).
        """
        if nmach == 0:
            return []
        listes = [[] for _ in range(nmach)]
        idx = 0
        accu = 0.0
        for u, v in circuit:
            longueur_km = U[u][v][0]["length"] / 1000.0
            if accu + longueur_km > cap and idx + 1 < nmach:
                idx += 1
                accu = 0.0
            listes[idx].append((u, v))
            accu += longueur_km
        return listes

    routes_I = découpe(circuit, cap_I, nI)
    routes_II = découpe(circuit, cap_II, nII)
    return routes_I, routes_II


# -----------------------------------------------
# 4) Tracé manuel avec Matplotlib (aucun GeoPandas)
# -----------------------------------------------
def plot_network_and_route(G, route_edges, color, title):
    """
    Trace le sous-graphe d'OSMnx G en gris clair, puis surcouche l'itinéraire
    donné sous forme de route_edges = [(u,v), ...] en couleur 'color'.
    'title' sert pour le titre de la figure.
    """
    U = (
        G.to_undirected()
    )  # MultiGraph non orienté, conserve 'length' et éventuellement 'geometry'

    # 1) Dessiner le fond : toutes les arêtes du réseau en gris léger
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v, data in U.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            ax.plot(xs, ys, linewidth=0.4, color="lightgray", zorder=1)
        else:
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=0.4, color="lightgray", zorder=1)

    # 2) Dessiner la route de la machine en surcouche
    for u, v in route_edges:
        data = U.get_edge_data(u, v)
        if data is None:
            # si l'arête n'existe pas tel quel, on la trace à la main
            x1, y1 = U.nodes[u]["x"], U.nodes[u]["y"]
            x2, y2 = U.nodes[v]["x"], U.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], linewidth=1.8, color=color, zorder=2)
        else:
            # On prend la première variante si arêtes multiples
            attrib = data[0]
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
    # Lecture du temps max (défaut 20h)
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

            # 2) Calculer la distance totale
            total_m = sum(d.get("length", 0) for u, v, d in G.edges(data=True))
            total_km = total_m / 1000.0

            # 3) Trouver la flotte optimale
            best_cost, best_nI, best_nII = find_best_mixed_fleet(
                total_km, max_hours, max_vehicles=10
            )
            if best_nI is None:
                print(
                    f"  ❌ Impossible de couvrir {total_km:.2f} km avec ≤ 10 machines en {max_hours:.1f}h.\n"
                )
                continue

            print(f"  • Distance du réseau : {total_km:.2f} km")
            print(
                f"  • Flotte optimale → {best_nI} I + {best_nII} II → {best_cost:.2f} $\n"
            )

            # 4) Générer les itinéraires
            routes_I, routes_II = build_eulerian_routes(G, best_nI, best_nII, max_hours)

            # 5) Tracer la ou les machines Type I en bleu
            if best_nI > 0:
                for idx, r in enumerate(routes_I, start=1):
                    plot_network_and_route(
                        G, r, color="blue", title=f"Itinéraire Machine I#{idx} → {name}"
                    )

            # 6) Tracer la ou les machines Type II en rouge
            if best_nII > 0:
                for idx, r in enumerate(routes_II, start=1):
                    plot_network_and_route(
                        G, r, color="red", title=f"Itinéraire Machine II#{idx} → {name}"
                    )

            print("\n")

        except Exception as e:
            print(f"❌ Erreur pour {name} : {e}\n")


if __name__ == "__main__":
    main()
