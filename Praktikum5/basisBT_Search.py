def BT_Search(assignment, domains):
    if complete(assignment):
        return assignment

    var = select_unassigned_variable_basic(assignment)  # einfach erste unzugewiesene

    for value in domains[var]:
        if consistent(var, value, assignment):   # prüft alle Constraints zu bereits belegten Variablen
            assignment[var] = value
            result = BT_Search(assignment, domains)
            if result is not None:
                return result
            del assignment[var]

    return None
