import matplotlib.pyplot as plt

def plot_training_loss(logs, fig_path):
    X = []
    Y = []
    Y1 = []
    Y2 = []
    for epoch in logs:
        X.append(epoch)
        if 'Training loss' in logs[epoch]:
            Y.append(logs[epoch]['Training loss'])
        if 'Training loss X' in logs[epoch]:
            Y1.append(logs[epoch]['Training loss X'])
        if 'Training loss E' in logs[epoch]:
            Y2.append(logs[epoch]['Training loss E'])

    plt.figure()
    plt.plot(X, Y, label='loss')
    plt.plot(X, Y1, label='loss X')
    plt.plot(X, Y2, label='loss E')
    plt.title("epoch vs Training loss")
    plt.xlabel("epoch")
    plt.ylabel("Training loss")
    plt.legend()
    plt.grid()
    plt.savefig(fig_path)

def plot_X_acc(logs, fig_path):
    X = []
    Y = []
    for epoch in logs:
        if 'X_acc' in logs[epoch]:
            X.append(epoch)
            Y.append(logs[epoch]['X_acc'])
    sorted_pairs = sorted(zip(X, Y))
    X, Y = zip(*sorted_pairs)
    
    plt.figure()
    plt.plot(X, Y)
    plt.title("epoch vs X accuracy")
    plt.xlabel("epoch")
    plt.ylabel("X accuracy")
    plt.savefig(fig_path)

def plot_E_acc(logs, fig_path):
    X = []
    Y = []
    for epoch in logs:
        if 'E_acc' in logs[epoch]:
            X.append(epoch)
            Y.append(logs[epoch]['E_acc'])
    sorted_pairs = sorted(zip(X, Y))
    X, Y = zip(*sorted_pairs)

    plt.figure()
    plt.plot(X, Y)
    plt.title("epoch vs E accuracy")
    plt.xlabel("epoch")
    plt.ylabel("E accuracy")
    plt.savefig(fig_path)