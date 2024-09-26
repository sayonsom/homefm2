import networkx as nx
import matplotlib.pyplot as plt
import random

class ACKnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.initialize_graph()

    def initialize_graph(self):
        # Define the nodes (entities) with detailed attributes
        entities = {
            "Air Conditioner": {"type": "main", "brand": "CoolTech", "model": "CT-2000"},
            "Room Temperature": {"type": "environment", "range": "16-32°C", "current": "24°C"},
            "Humidity": {"type": "environment", "range": "30-70%", "current": "50%"},
            "Power Consumption": {"type": "performance", "unit": "kWh", "range": "0.5-3.5 kWh/h"},
            "Cooling Capacity": {"type": "performance", "unit": "BTU", "value": "12000"},
            "User Preferences": {"type": "user", "temp_preference": "22°C", "mode": "auto"},
            "Compressor": {"type": "component", "type": "inverter", "power": "1000W"},
            "Smart Thermostat": {"type": "component", "accuracy": "±0.5°C"},
            "Air Filter": {"type": "component", "type": "HEPA", "last_cleaned": "2023-09-01"},
            "Energy Efficiency": {"type": "performance", "rating": "SEER 21"},
            "Maintenance Schedule": {"type": "service", "next_service": "2024-03-15"},
            "Mobile App": {"type": "interface", "version": "2.3.1", "last_updated": "2023-11-30"},
            "Voice Assistant": {"type": "interface", "supported": ["Alexa", "Google Assistant"]},
            "Electricity Cost": {"type": "economic", "rate": "0.15 $/kWh"},
            "CO2 Emission": {"type": "impact", "value": "0.5 kg/kWh"},
        }

        # Add the nodes to the graph
        for entity, attributes in entities.items():
            self.G.add_node(entity, **attributes)

        # Define relationships (edges) between entities with weights
        edges = [
            ("Air Conditioner", "Room Temperature", {"label": "controls", "weight": 0.9}),
            ("Air Conditioner", "Humidity", {"label": "reduces", "weight": 0.7}),
            ("Air Conditioner", "Power Consumption", {"label": "requires", "weight": 0.8}),
            ("Air Conditioner", "Cooling Capacity", {"label": "has", "weight": 0.9}),
            ("Compressor", "Cooling Capacity", {"label": "determines", "weight": 0.9}),
            ("Smart Thermostat", "Room Temperature", {"label": "measures", "weight": 0.8}),
            ("User Preferences", "Air Conditioner", {"label": "sets", "weight": 0.7}),
            ("Room Temperature", "User Preferences", {"label": "influences", "weight": 0.6}),
            ("Air Filter", "Air Quality", {"label": "improves", "weight": 0.7}),
            ("Energy Efficiency", "Power Consumption", {"label": "affects", "weight": 0.8}),
            ("Maintenance Schedule", "Energy Efficiency", {"label": "maintains", "weight": 0.6}),
            ("Mobile App", "User Preferences", {"label": "adjusts", "weight": 0.7}),
            ("Voice Assistant", "User Preferences", {"label": "controls", "weight": 0.6}),
            ("Power Consumption", "Electricity Cost", {"label": "determines", "weight": 0.8}),
            ("Power Consumption", "CO2 Emission", {"label": "contributes to", "weight": 0.7}),
        ]

        # Add the edges to the graph
        self.G.add_edges_from((source, target, attr) for source, target, attr in edges)

        # Add specific scenarios as nodes
        scenarios = [
            ("Heatwave", {"temp": "35°C", "humidity": "70%", "action": "max cooling"}),
            ("Night Mode", {"time": "22:00-06:00", "temp": "20°C", "fan_speed": "low"}),
            ("Energy Saving", {"temp_increase": "2°C", "fan_speed": "auto", "compressor": "eco mode"}),
            ("Remote Preheat", {"trigger": "geofencing", "action": "start 30 min before arrival"}),
            ("Air Quality Alert", {"trigger": "AQI > 150", "action": "activate air purification"}),
        ]
        for scenario, attributes in scenarios:
            self.G.add_node(scenario, **attributes)
            self.G.add_edge("User Preferences", scenario, label="activates", weight=0.7)
            self.G.add_edge(scenario, "Air Conditioner", label="adjusts", weight=0.8)

    def update_node_attribute(self, node, attribute, value):
        if node in self.G.nodes:
            self.G.nodes[node][attribute] = value
            print(f"Updated {node}: {attribute} = {value}")
        else:
            print(f"Node {node} not found in the graph.")

    def add_user_feedback(self, feedback):
        feedback_node = f"Feedback_{len([n for n in self.G.nodes if n.startswith('Feedback_')])}"
        self.G.add_node(feedback_node, type="user_feedback", content=feedback)
        self.G.add_edge("User Preferences", feedback_node, label="provides", weight=0.6)
        print(f"Added user feedback: {feedback}")

    def simulate_dynamic_update(self):
        # Simulate changes in room conditions
        new_temp = round(random.uniform(18, 30), 1)
        new_humidity = round(random.uniform(30, 70), 1)
        self.update_node_attribute("Room Temperature", "current", f"{new_temp}°C")
        self.update_node_attribute("Humidity", "current", f"{new_humidity}%")

        # Simulate user feedback
        feedbacks = ["Too cold", "Perfect temperature", "A bit humid", "Energy bill too high"]
        self.add_user_feedback(random.choice(feedbacks))

    def visualize_graph(self):
        pos = nx.spring_layout(self.G, k=0.9, iterations=50)
        plt.figure(figsize=(16, 14))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=3000, node_color="lightblue")
        nx.draw_networkx_labels(self.G, pos, font_size=8, font_weight="bold")
        
        # Draw edges
        edge_weights = [self.G[u][v].get('weight', 0.5) for u, v in self.G.edges()]
        nx.draw_networkx_edges(self.G, pos, width=edge_weights, alpha=0.7, edge_color="gray", arrows=True)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red', font_size=6)
        
        plt.title("Advanced Air Conditioner Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Usage example
ac_graph = ACKnowledgeGraph()
ac_graph.visualize_graph()

# Simulate dynamic updates
for _ in range(3):
    ac_graph.simulate_dynamic_update()

ac_graph.visualize_graph()