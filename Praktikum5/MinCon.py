def min_conflicts(domains, max_steps, seed):
    rng = random.Random(seed)

    # zufällige vollständige Belegung
    assignment = {v: rng.choice(list(domains[v])) for v in variables}

    for step in range(1, max_steps+1):
        if total_conflicts(assignment) == 0:
            return assignment, step

        conflicted = [v for v in variables
                      if count_conflicts(v, assignment[v], assignment) > 0]

        var = rng.choice(conflicted)
        # Wert mit minimalen Konflikten
        best_val, best_conf = None, float("inf")
        for val in domains[var]:
            c = count_conflicts(var, val, assignment)
            if c < best_conf:
                best_conf, best_val = c, val
        assignment[var] = best_val

    return None, max_steps   # keine Lösung gefunden
