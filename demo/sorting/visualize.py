import numpy as np
import matplotlib.pyplot as pyplot



def plot_graph_edges(graph,edge_output, sort_indexes,title=None,savefile=None):
    """Plot edges for a given graph assuming certain order of nodes."""
    sort_indexes = np.squeeze(sort_indexes).astype(np.int)
    fig = pyplot.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)

    nd = graph.get_num_nodes()
    probs = np.zeros((nd, nd))
    for s, r, ef in zip(graph.senders.numpy(), graph.receivers.numpy(), edge_output):
        probs[s, r] = ef
    ax.matshow(probs[sort_indexes][:, sort_indexes], cmap="viridis")
    ax.grid(False)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if savefile is not None:
        fig.savefig(savefile,dpi=150)
        pyplot.close(fig)

def plot_ranked_inputs(value_nodes,sort_indexes,ranks,title=None,savefile=None):
    fig = pyplot.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    num_elements = value_nodes.shape[0]
    inputs = np.squeeze(value_nodes)
    ranks = np.squeeze(ranks * (num_elements - 1.0)).astype(int)
    x = np.arange(inputs.shape[0])

    ax1.set_title("Inputs")
    ax1.barh(x, inputs, color="b")
    ax1.set_xlim(-0.01, 1.01)

    ax2.set_title("Sorted")
    ax2.barh(x, inputs[sort_indexes], color="k")
    ax2.set_xlim(-0.01, 1.01)

    ax3.set_title("Ranks")
    ax3.barh(x, ranks, color="r")
    ax3.set_xlim(0, len(ranks) + 0.5)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if savefile is not None:
        fig.savefig(savefile,dpi=150)
        pyplot.close(fig)

def plot_losses(train_loss,test_loss,train_correct,test_correct,train_solved,test_solved,title=None,savefile=None):
    fig = pyplot.figure(11, figsize=(18, 3))
    # Loss.
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(train_loss, "k", label="Training")
    ax.plot(test_loss, "r--", label="Test")
    ax.set_title("Loss across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (binary cross-entropy)")
    ax.legend()

    # Correct.
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(train_correct, "k", label="Training")
    ax.plot(test_correct, "r--", label="Test/generalization")
    ax.set_title("Fraction correct across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction nodes/edges correct")

    # Solved.
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(train_solved, "k", label="Training")
    ax.plot(test_solved, "r--", label="Test/generalization")
    ax.set_title("Fraction solved across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction examples solved")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if savefile is not None:
        fig.savefig(savefile,dpi=150)
        pyplot.close(fig)
