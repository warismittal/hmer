def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def exprate(preds, gts):
    n = len(gts)
    eds = [edit_distance(p, g) for p, g in zip(preds, gts)]
    return {"exprate": 100 * sum(e == 0 for e in eds) / n,
            "le1": 100 * sum(e <= 1 for e in eds) / n,
            "le2": 100 * sum(e <= 2 for e in eds) / n}
