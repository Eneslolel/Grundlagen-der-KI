def AC3(domains):
    queue = deque(all_arcs)   # alle (Xi, Xj) mit binärem Constraint
    while queue:
        Xi, Xj = queue.popleft()
        if revise(domains, Xi, Xj):
            if not domains[Xi]:
                return False
            for Xk in neighbors[Xi] - {Xj}:
                queue.append((Xk, Xi))
    return True

def revise(domains, Xi, Xj):
    revised = False
    for x in list(domains[Xi]):
        if not any(all(f(x, y) for f in constraints_between(Xi, Xj))
                   for y in domains[Xj]):
            domains[Xi].remove(x)
            revised = True
    return revised
