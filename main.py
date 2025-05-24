import osmnx as ox
import networkx as nx

place_name = "Montréal, Québec, Canada"

G = ox.graph_from_place(place_name, network_type='drive')
ox.save_graphml(G, filepath="montreal_directed.graphml")
G_undirected = G.to_undirected()
ox.save_graphml(G_undirected, filepath="montreal_undirected.graphml")
