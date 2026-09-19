def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    train_days = []
    remaining_budget = config["budget"]
    last_retrain_day = None
    days_since_retrain = 0

    for d_stat in daily_stats:
        day = d_stat["day"]

        # Increment before checking today's conditions.
        days_since_retrain += 1

        requests_retrain = (
            d_stat["drift_score"] > config["drift_threshold"]
            or d_stat["performance"] < config["performance_threshold"]
            or days_since_retrain >= config["max_staleness"]
        )

        # Initial cooldown is already satisfied.
        cooldown_ok = (
            last_retrain_day is None
            or day - last_retrain_day >= config["cooldown"]
        )

        budget_ok = remaining_budget >= config["retrain_cost"]

        if requests_retrain and cooldown_ok and budget_ok:
            train_days.append(day)
            remaining_budget -= config["retrain_cost"]
            last_retrain_day = day
            days_since_retrain = 0

    return train_days
