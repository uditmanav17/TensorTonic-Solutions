def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = [0.0] * num_states
    
    # Iterate over each state s
    for s in range(num_states):
        max_q_value = float('-inf')
        num_actions = len(transitions[s])
        
        # Iterate over all possible actions a from state s
        for a in range(num_actions):
            q_value = rewards[s][a]
            
            # Compute the expected future value: sum over s' of T(s, a, s') * V(s')
            expected_future_value = 0.0
            for s_next in range(num_states):
                expected_future_value += transitions[s][a][s_next] * values[s_next]
                
            # Full Q-value calculation: R(s, a) + gamma * expected_future_value
            q_value += gamma * expected_future_value
            
            # Keep track of the maximum Q-value across all actions
            if q_value > max_q_value:
                max_q_value = q_value
                
        # Update the value of state s with the maximum Q-value found
        new_values[s] = float(max_q_value)
        
    return new_values