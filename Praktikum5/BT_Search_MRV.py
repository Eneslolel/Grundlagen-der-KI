def select_mrv_degree(assignment, domains):
    unassigned = [v for v in variables if v not in assignment]
    mrv = min(len(domains[v]) for v in unassigned)
    candidates = [v for v in unassigned if len(domains[v]) == mrv]
    if len(candidates) == 1:
        return candidates[0]
    # Gradheuristik als Tie breaker
    best, best_deg = None, -1
    for v in candidates:
        deg = sum(1 for n in neighbors[v] if n not in assignment)
        if deg > best_deg:
            best_deg, best = deg, v
    return best
