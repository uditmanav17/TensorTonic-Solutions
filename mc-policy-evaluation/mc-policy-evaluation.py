import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    V = np.zeros(n_states)
    counts = np.zeros(n_states)

    for episode in episodes:
        G = 0
        returns = [0] * len(episode)

        # Compute returns backwards
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            returns[t] = G

        visited = set()

        # First-visit updates
        for t, (state, _) in enumerate(episode):
            if state not in visited:
                V[state] += returns[t]
                counts[state] += 1
                visited.add(state)

    for s in range(n_states):
        if counts[s] > 0:
            V[s] /= counts[s]

    return V