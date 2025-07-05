import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###  MicroGrad demo""")
    return


@app.cell
def _():
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return np, plt, random


@app.cell
def _():
    from micrograd.engine import Value
    from micrograd.nn import Neuron, Layer, MLP
    return MLP, Value


@app.cell
def _(np, random):
    np.random.seed(1337)
    random.seed(1337)
    return


@app.cell
def _(plt):
    # make up a dataset

    from sklearn.datasets import make_moons, make_blobs
    X, y = make_moons(n_samples=100, noise=0.1)

    y = y*2 - 1 # make y be -1 or 1
    # visualize in 2D
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
    return X, y


@app.cell
def _(MLP):
    # initialize a model 
    model = MLP(2, [16, 16, 1]) # 2-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))
    return (model,)


@app.cell
def _(Value, X, model, np, y):
    def loss(batch_size=None):
        if batch_size is None:
            Xb, yb = (X, y)
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = (X[ri], y[ri])
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        scores = list(map(model, inputs))
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        alpha = 0.0001
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        _total_loss = data_loss + reg_loss
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return (_total_loss, sum(accuracy) / len(accuracy))
    _total_loss, _acc = loss()
    print(_total_loss, _acc)
    return (loss,)


@app.cell
def _(loss, model):
    for k in range(100):
        _total_loss, _acc = loss()
        model.zero_grad()
        _total_loss.backward()
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        if k % 1 == 0:
            print(f'step {k} loss {_total_loss.data}, accuracy {_acc * 100}%')
    return


@app.cell
def _(Value, X, model, np, plt, y):
    # visualize decision boundary

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
