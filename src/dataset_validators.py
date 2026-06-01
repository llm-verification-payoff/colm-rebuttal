from datasets import Dataset


def trivial_validator(ds: Dataset) -> bool:
    return True


def validate_sat_dataset(ds: Dataset) -> bool:
    for i in range(len(ds)):
        ex = ds[i]
        cnf = ex['cnf']
        assignment = ex['assignment']

        vars_in_cnf = set()
        for clause in cnf:
            clause_satisfied = False
            for literal in clause:
                if literal.startswith("~"):
                    var = literal[1:]
                    vars_in_cnf.add(var)
                    if not (var in assignment): return False
                    if not assignment[var]:
                        clause_satisfied = True
                else:
                    vars_in_cnf.add(literal)
                    if not (literal in assignment): return False
                    if assignment[literal]:
                        clause_satisfied = True
            if not clause_satisfied:
                return False

        if not vars_in_cnf.issubset(set(assignment.keys())): return False
        for var, value in assignment.items():
            if var not in vars_in_cnf:
                if value != None: return False
        ex['assignment'] = {var: value for var, value in assignment.items() if value != None}

    return True


def validate_sudoku_dataset(ds: Dataset) -> bool:
    from oracle_verifiers import sudoku_is_correct

    for i in range(len(ds)):
        ex = ds[i]
        answer_text = ex['answer']
        grid = [[int(x) for x in line.split()] for line in answer_text.split('\n')]
        if not sudoku_is_correct(ex, grid):
            return False

    return True


def validate_matmul_dataset(ds: Dataset) -> bool:
    from oracle_verifiers import matmul_is_correct

    for i in range(len(ds)):
        ex = ds[i]
        answer_text = ex['answer']
        product = [[int(x) for x in line.split()] for line in answer_text.split('\n')]
        if not matmul_is_correct(ex, product):
            return False

    return True
