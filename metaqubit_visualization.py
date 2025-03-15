import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from meta_qubit import MetaQubit

# Entropy Calculation
def calculate_entropy(values):
    probabilities = (values - min(values)) / (max(values) - min(values) + 1e-9)
    probabilities = probabilities / np.sum(probabilities)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

# Coherence Calculation
def calculate_coherence(connections):
    return np.mean([weight for _, _, weight in connections])

# Tunneling Probability Calculation
def calculate_tunneling_probability(weight):
    return np.exp(-weight)  # Exponential decrease of probability with weight

# Graph Creation
def create_advanced_graph(meta_qubit, num_frames):
    num_qubits = meta_qubit.num_qubits
    graphs = []
    entropies = []
    coherences = []
    mean_weights = []
    mean_tunnelings = []

    for _ in range(num_frames):
        output = meta_qubit.run_circuit()
        G = nx.Graph()

        for i in range(num_qubits):
            state = '|1>' if output[i] > 0 else '|0>'
            G.add_node(i, state=state, value=output[i])

        weights = []
        tunnelings = []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                weight = np.abs(output[i] - output[j])
                tunneling = calculate_tunneling_probability(weight)
                G.add_edge(i, j, weight=weight, tunneling=tunneling)
                weights.append(weight)
                tunnelings.append(tunneling)

        # Compute metrics for the frame
        entropy = calculate_entropy(output)
        coherence = calculate_coherence(G.edges(data='weight'))
        mean_weight = np.mean(weights)
        mean_tunneling = np.mean(tunnelings)

        entropies.append(entropy)
        coherences.append(coherence)
        mean_weights.append(mean_weight)
        mean_tunnelings.append(mean_tunneling)
        graphs.append(G)

    return graphs, entropies, coherences, mean_weights, mean_tunnelings

# Animation with Tunneling Indications and Detailed Statistics
def animate_advanced_graph(graphs, entropies, coherences, mean_weights, mean_tunnelings):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Two subplots: Graph and Statistics
    pos = nx.circular_layout(graphs[0])  # Circular layout

    def update(frame):
        ax[0].clear()
        ax[1].clear()

        G = graphs[frame]

        # Node colors based on state
        node_colors = ['red' if G.nodes[node]['state'] == '|1>' else 'blue' for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8, ax=ax[0])

        # Dynamic edge thickness
        edges = G.edges(data=True)
        edge_weights = [edge[2]['weight'] for edge in edges]
        nx.draw_networkx_edges(G, pos, width=[2 * w for w in edge_weights], alpha=0.6, ax=ax[0])

        # Tunneling indications with dots
        for u, v, d in G.edges(data=True):
            tunneling_prob = d['tunneling']
            if tunneling_prob > 0.7:  # Display only strong connections
                mid_point = (pos[u] + pos[v]) / 2
                ax[0].scatter(*mid_point, s=tunneling_prob * 200, color='green', alpha=0.7)

        # Node labels
        labels = {node: f"Q{node}\n{G.nodes[node]['state']}\n({G.nodes[node]['value']:.2f})" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax[0])

        # Title and settings
        ax[0].set_title(f"Frame {frame+1}/{len(graphs)} | Entropy: {entropies[frame]:.2f}, "
                        f"Coherence: {coherences[frame]:.2f}, Mean Weight: {mean_weights[frame]:.2f}, "
                        f"Mean Tunneling: {mean_tunnelings[frame]:.2f}")
        ax[0].axis('off')

        # Metrics on the second subplot
        ax[1].plot(range(1, frame + 2), entropies[:frame + 1], label='Entropy', color='blue')
        ax[1].plot(range(1, frame + 2), coherences[:frame + 1], label='Coherence', color='purple', linewidth=2)
        ax[1].plot(range(1, frame + 2), mean_weights[:frame + 1], label='Mean Weight', color='yellow')
        ax[1].plot(range(1, frame + 2), mean_tunnelings[:frame + 1], label='Mean Tunneling', color='green')
        ax[1].set_ylim(0, max(max(entropies), max(coherences), max(mean_weights), max(mean_tunnelings)) + 0.5)
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Frame')
        ax[1].set_ylabel('Value')
        ax[1].set_title("Evolution of Metrics Over Time")
        ax[1].grid(True)

    ani = FuncAnimation(fig, update, frames=len(graphs), interval=500, repeat=True)
    plt.show()

# Execution
meta_qubit = MetaQubit(num_qubits=14)
num_frames = 50
graphs, entropies, coherences, mean_weights, mean_tunnelings = create_advanced_graph(meta_qubit, num_frames)
animate_advanced_graph(graphs, entropies, coherences, mean_weights, mean_tunnelings)

import matplotlib.animation as animation

def save_animation(graphs, entropies, coherences, mean_weights, mean_tunnelings, filename="meta_qubit_simulation.mp4"):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.circular_layout(graphs[0])

    def update(frame):
        ax[0].clear()
        ax[1].clear()

        G = graphs[frame]

        node_colors = ['red' if G.nodes[node]['state'] == '|1>' else 'blue' for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8, ax=ax[0])
        edges = G.edges(data=True)
        edge_weights = [edge[2]['weight'] for edge in edges]
        nx.draw_networkx_edges(G, pos, width=[2 * w for w in edge_weights], alpha=0.6, ax=ax[0])

        labels = {node: f"Q{node}\n{G.nodes[node]['state']}\n({G.nodes[node]['value']:.2f})" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax[0])

        # ✅ Add title with metric values in animation
        fig.suptitle(f"Frame {frame+1}/{len(graphs)} | Entropy: {entropies[frame]:.2f}, "
                     f"Coherence: {coherences[frame]:.2f}, Mean Weight: {mean_weights[frame]:.2f}, "
                     f"Mean Tunneling: {mean_tunnelings[frame]:.2f}", fontsize=14)

        # Metrics plot
        ax[1].plot(range(1, frame + 2), entropies[:frame + 1], label='Entropy', color='blue')
        ax[1].plot(range(1, frame + 2), coherences[:frame + 1], label='Coherence', color='purple', linewidth=2)
        ax[1].plot(range(1, frame + 2), mean_weights[:frame + 1], label='Mean Weight', color='yellow')
        ax[1].plot(range(1, frame + 2), mean_tunnelings[:frame + 1], label='Mean Tunneling', color='green')
        ax[1].set_ylim(0, max(max(entropies), max(coherences), max(mean_weights), max(mean_tunnelings)) + 0.5)
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Frame')
        ax[1].set_ylabel('Value')
        ax[1].set_title("Evolution of Metrics Over Time")
        ax[1].grid(True)

    ani = FuncAnimation(fig, update, frames=len(graphs), interval=1000)  # ✅ 1 frame per second (1000ms)

    # ✅ Low FPS to make the video run at 1 second per frame
    writer = animation.FFMpegWriter(fps=1, metadata={"title": "MetaQubit Simulation"})
    ani.save(filename, writer=writer)
    print(f"🎥 Simulation saved as {filename}!")

# Run the animation save function
save_animation(graphs, entropies, coherences, mean_weights, mean_tunnelings, filename="meta_qubit_simulation.mp4")
