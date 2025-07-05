import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    # brew install graphviz
    # pip install graphviz
    from graphviz import Digraph
    return (Digraph,)


@app.cell
def _():
    from micrograd.engine import Value
    return (Value,)


@app.cell
def _(Digraph):
    def trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw_dot(root, format='svg', rankdir='LR'):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ['LR', 'TB']
        nodes, edges = trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

        for n in nodes:
            dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
    return (draw_dot,)


@app.cell
def _(Value, draw_dot):
    _x = Value(1.0)
    _y = (_x * 2 + 1).relu()
    _y.backward()
    draw_dot(_y)
    return


@app.cell
def _(Value, draw_dot):
    import random
    from micrograd import nn
    random.seed(1337)
    n = nn.Neuron(2)
    _x = [Value(1.0), Value(-2.0)]
    _y = n(_x)
    _y.backward()
    dot = draw_dot(_y)
    dot
    return (dot,)


@app.cell
def _(dot):
    dot.render('gout')
    return


if __name__ == "__main__":
    app.run()
