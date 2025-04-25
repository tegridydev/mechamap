import graphviz

def render(edges, path="circuit.dot"):
    dot = graphviz.Digraph()
    for u,v in edges:
        dot.edge(str(u), str(v))
    dot.save(path)
    return path
