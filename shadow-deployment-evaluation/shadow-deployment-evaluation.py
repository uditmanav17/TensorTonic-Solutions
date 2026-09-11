import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    # Write code here
    shadow_latencies = []
    shadow_prod_agree = total = prod_correct = shadow_correct = 0

    for p_log, s_log in zip(production_log, shadow_log):
        total += 1
        shadow_latencies.append(s_log["latency_ms"])
        if p_log["actual"] == p_log["prediction"]:
            prod_correct += 1
        if s_log["actual"] == s_log["prediction"]:
            shadow_correct += 1
        if p_log["prediction"] == s_log["prediction"]:
            shadow_prod_agree += 1

    prod_acc = prod_correct / total
    shad_acc = shadow_correct / total
    acc_gain = shad_acc - prod_acc

    agree_rate = shadow_prod_agree / total

    shadow_latencies.sort()
    rank = math.ceil(0.95 * total)
    idx = rank - 1
    p95 = shadow_latencies[idx]

    ans = {
        "promote": all([
            p95 <= criteria["max_latency_p95"],
            acc_gain >= criteria["min_accuracy_gain"],
            agree_rate >= criteria["min_agreement_rate"],
        ]), 
        "metrics": { 
            "shadow_accuracy": shad_acc, 
            "production_accuracy": prod_acc, 
            "accuracy_gain": acc_gain, 
            "shadow_latency_p95": p95, 
            "agreement_rate": agree_rate
        } 
    }
    return ans        

        
        