import numpy as np


def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    if len(production_log) != len(shadow_log):
        raise ValueError("Production and shadow logs must have the same length")

    if not production_log:
        raise ValueError("Logs cannot be empty")

    p_actual = np.array([x["actual"] for x in production_log])
    p_pred = np.array([x["prediction"] for x in production_log])

    s_actual = np.array([x["actual"] for x in shadow_log])
    s_pred = np.array([x["prediction"] for x in shadow_log])

    latencies = np.array([x["latency_ms"] for x in shadow_log])

    production_accuracy = np.mean(p_actual == p_pred)
    shadow_accuracy = np.mean(s_actual == s_pred)
    agreement_rate = np.mean(p_pred == s_pred)

    accuracy_gain = shadow_accuracy - production_accuracy

    idx = int(np.ceil(0.95 * len(production_log))) - 1
    p95 = np.partition(latencies, idx)[idx]

    promote = (
        p95 <= criteria["max_latency_p95"]
        and accuracy_gain >= criteria["min_accuracy_gain"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": bool(promote),
        "metrics": {
            "shadow_accuracy": float(shadow_accuracy),
            "production_accuracy": float(production_accuracy),
            "accuracy_gain": float(accuracy_gain),
            "shadow_latency_p95": float(p95),
            "agreement_rate": float(agreement_rate),
        },
    }
