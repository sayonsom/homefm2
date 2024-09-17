import networkx as nx
import matplotlib.pyplot as plt

# Create a new directed graph
G = nx.DiGraph()

# Add nodes (activities/settings)
activities = [
    "Wake up", "Turn on lights", "Start coffee maker", 
    "Open refrigerator", "Prepare breakfast", "Eat breakfast", 
    "Shower", "Get dressed", "Leave for work"
]
G.add_nodes_from(activities)

# Add edges with probabilities
edges_with_probabilities = [
    ("Wake up", "Turn on lights", 0.9),
    ("Wake up", "Start coffee maker", 0.7),
    ("Turn on lights", "Open refrigerator", 0.6),
    ("Start coffee maker", "Open refrigerator", 0.8),
    ("Open refrigerator", "Prepare breakfast", 0.9),
    ("Prepare breakfast", "Eat breakfast", 1.0),
    ("Eat breakfast", "Shower", 0.7),
    ("Shower", "Get dressed", 1.0),
    ("Get dressed", "Leave for work", 0.95)
]
G.add_weighted_edges_from(edges_with_probabilities)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Daily Activity Event Graph")
plt.axis('off')
plt.tight_layout()
plt.show()

# Analysis
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Most central node:", max(nx.degree_centrality(G).items(), key=lambda x: x[1])[0])