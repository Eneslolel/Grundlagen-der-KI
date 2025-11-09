from collections import Counter, defaultdict
import math

data = [
    (">=35","hoch","Abitur","O"),
    ("<35","niedrig","Master","O"),
    (">=35","hoch","Bachelor","M"),
    (">=35","niedrig","Abitur","M"),
    (">=35","hoch","Master","O"),
    ("<35","hoch","Bachelor","O"),
    ("<35","niedrig","Abitur","M"),
]
attrs = ["Alter","Einkommen","Bildung"]
X = [d[:3] for d in data]
y = [d[3] for d in data]

def entropy(labels):
    n = len(labels)
    return -sum((c/n) * math.log2(c/n) for c in Counter(labels).values())

def info_gain(examples, labels, ai):
    n = len(labels)
    groups = defaultdict(list)
    for ex, lab in zip(examples, labels):
        groups[ex[ai]].append(lab)
    remainder = sum((len(g)/n) * entropy(g) for g in groups.values())
    return entropy(labels) - remainder, groups

def id3(examples, labels, attr_ids):
    if len(set(labels)) == 1:
        return ("Leaf", labels[0])
    if not attr_ids:
        return ("Leaf", Counter(labels).most_common(1)[0][0])
    best = max(attr_ids, key=lambda ai: info_gain(examples, labels, ai)[0])
    _, groups = info_gain(examples, labels, best)
    node = ("Node", attrs[best], {})
    for val, grp in groups.items():
        subX = [ex for ex, lab in zip(examples, labels) if ex[best] == val]
        node[2][val] = id3(subX, grp, [a for a in attr_ids if a != best])
    return node

def cal3(examples, labels, attr_ids, s1=4, s2=0.7):
    n = len(labels)
    majority, mcount = Counter(labels).most_common(1)[0]
    purity = mcount / n
    if n < s1 or purity >= s2 or len(set(labels)) == 1 or not attr_ids:
        return ("Leaf", majority, n, purity)
    best = max(attr_ids, key=lambda ai: info_gain(examples, labels, ai)[0])
    _, groups = info_gain(examples, labels, best)
    node = ("Node", attrs[best], n, purity, {})
    for val, grp in groups.items():
        subX = [ex for ex, lab in zip(examples, labels) if ex[best] == val]
        node[4][val] = cal3(subX, grp, [a for a in attr_ids if a != best], s1, s2)
    return node

def pretty(tree, indent=""):
    kind = tree[0]
    if kind == "Leaf":
        if len(tree) == 2:
            print(f"{indent}→ {tree[1]}")
        else:
            print(f"{indent}→ {tree[1]}  (n={tree[2]}, purity={tree[3]:.2f})")
        return
    if kind == "Node":
        name = tree[1]
        children = tree[2] if len(tree) == 3 else tree[4]
        print(f"{indent}[{name}]")
        for val, child in children.items():
            print(f"{indent}  {name} = {val}")
            pretty(child, indent + "    ")

print("ID3 Baum")
pretty(id3(X, y, [0,1,2]))
print("\nCAL3 Baum S1=4 S2=0.7")
pretty(cal3(X, y, [0,1,2], s1=4, s2=0.7))
