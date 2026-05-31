import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration using NumPy vectorization.
    """
    # Convert inputs to NumPy arrays
    V = np.array(values)          # Shape: (num_states,)
    R = np.array(rewards)         # Shape: (num_states, num_actions)
    T = np.array(transitions)     # Shape: (num_states, num_actions, num_states)
    
    # Compute expected future values for all (s, a) pairs:
    # T matrix-multiplied by V along the last axis.
    # Shape of expected_future_value will be (num_states, num_actions)
    expected_future_value = np.dot(T, V)
    
    # Calculate the Q-values for all state-action pairs
    Q = R + gamma * expected_future_value
    
    # Take the maximum over the action axis (axis 1)
    new_values = np.max(Q, axis=1)
    
    # Return as a list of floats to match the original signature
    return new_values.tolist()