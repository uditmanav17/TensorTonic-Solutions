def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
# 1. Warmup Phase
    if step < warmup_steps:
        return step * initial_lr / warmup_steps
    
    # 2. Linear Decay Phase
    elif warmup_steps <= step <= total_steps:
        # Calculate how far we are into the decay phase
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return final_lr
        
        # Standard linear interpolation: η_f + (η_0 - η_f) * (remaining_steps / total_decay_steps)
        return final_lr + (initial_lr - final_lr) * (total_steps - step) / decay_steps
    
    # 3. Post-training Phase
    else:
        return final_lr