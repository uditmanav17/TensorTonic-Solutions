from collections import defaultdict

def target_encoding(categories, targets):
    stats = defaultdict(lambda: [0, 0])

    for cat, target in zip(categories, targets):
        stats[cat][0] += target
        stats[cat][1] += 1

    return [stats[cat][0] / stats[cat][1] for cat in categories]