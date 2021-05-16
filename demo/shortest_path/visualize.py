# Copyright 2021 Rufaim (https://github.com/Rufaim)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np


class GraphPlotter(object):
    def __init__(self, graph):
        self._graph = graph
        pos = {k: v["pos"] for k, v in graph.nodes.items()}
        self._base_draw_kwargs = {"G":self._graph, "pos":pos}

    @property
    def solution_length(self):
        return len(self.solution_edges)

    @property
    def start_nodes(self):
        return [ n for n in self._graph.nodes() if self._graph.nodes[n].get("start", False) ]

    @property
    def end_nodes(self):
        return [ n for n in self._graph.nodes() if self._graph.nodes[n].get("end", False)]

    @property
    def solution_nodes(self):
        return [ n for n in self._graph.nodes() if self._graph.nodes[n].get("solution", False)]

    @property
    def intermediate_solution_nodes(self):
        return [ n for n in self._graph.nodes()
                if self._graph.nodes[n].get("solution", False) and
                   not self._graph.nodes[n].get("start", False) and
                   not self._graph.nodes[n].get("end", False)]

    @property
    def solution_edges(self):
        return [ e for e in self._graph.edges()
                if self._graph.get_edge_data(e[0], e[1]).get("solution", False)]

    @property
    def non_solution_nodes(self):
        return [ n for n in self._graph.nodes()
                if not self._graph.nodes[n].get("solution", False)]

    @property
    def non_solution_edges(self):
        return [ e for e in self._graph.edges()
                if not self._graph.get_edge_data(e[0], e[1]).get("solution", False) ]

    def _make_draw_kwargs(self, ax, **kwargs):
        kwargs["ax"] = ax
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, ax, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(ax, **kwargs)
        collection = draw_function(**draw_kwargs)
        if isinstance(collection,list):
            for c in collection:
                c.set_zorder(zorder)
        else:
            collection.set_zorder(zorder)
        return collection

    def _draw_nodes(self, ax, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        return self._draw(ax, nx.draw_networkx_nodes, **kwargs)

    def _draw_edges(self, ax, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(ax, nx.draw_networkx_edges, **kwargs)

    def draw_graph(self, ax,
                   node_size=200,
                   node_color=(0.4, 0.8, 0.4),
                   node_border_color=(0.0, 0.0, 0.0, 1.0),
                   node_linewidth=1.0,
                   edge_width=1.0):
        # Plot nodes.
        self._draw_nodes(ax,
            nodelist=self._graph.nodes(),
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20)
        # Plot edges.
        self._draw_edges(ax, edgelist=self._graph.edges(), width=edge_width, zorder=10)

    def draw_graph_with_solution(self, ax,
                                    node_size=200,
                                    node_color="limegreen",
                                    node_border_color="k",
                                    node_linewidth=1.0,
                                    edge_width=1.0,
                                    start_color="w",
                                    end_color="k",
                                    solution_node_linewidth=3.0,
                                    solution_edge_width=4.0):
        node_collections = {}
        # Plot start nodes.
        node_collections["start nodes"] = self._draw_nodes(
            ax,
            nodelist=self.start_nodes,
            node_size=node_size,
            node_color=start_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=100)
        # Plot end nodes.
        node_collections["end nodes"] = self._draw_nodes(
            ax,
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90)
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = node_color
        node_collections["intermediate solution nodes"] = self._draw_nodes(
            ax,
            nodelist=self.intermediate_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=80)
        # Plot solution edges.
        node_collections["solution edges"] = self._draw_edges(
            ax, edgelist=self.solution_edges, width=solution_edge_width, zorder=70)
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self._draw_nodes(
            ax,
            nodelist=self.non_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20)
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self._draw_edges(
            ax, edgelist=self.non_solution_edges, width=edge_width, zorder=10)
        # Set title as solution length.
        ax.set_title("Solution length: {}".format(self.solution_length))
        return node_collections


def plot_predicted_graphs(output_graphs,target_graphs,raw_graphs,num_steps_to_plot=4,
                          node_size=120,min_c=0.3,savefile=None):

    step_indices = np.floor(np.linspace(0, len(output_graphs) - 1, num_steps_to_plot)).astype(int).tolist()
    h = len(raw_graphs)
    w = num_steps_to_plot + 1
    fig = pyplot.figure(101, figsize=(18, h * 3))
    fig.clf()

    for i, graph in enumerate(raw_graphs):
        out_graphs = [output_graphs[j].get_graph_by_index(i) for j in step_indices]
        t_graphs = target_graphs.get_graph_by_index(i)
        t_graphs_nodes = t_graphs.nodes.numpy()[:,-1]

        color = {}
        for j, n in enumerate(graph.nodes):
            color[n] = np.array([1.0 - t_graphs_nodes[j], 0.0, t_graphs_nodes[j], 1.0]) * (1.0 - min_c) + min_c

        iax = i * (1 + num_steps_to_plot) + 1
        ax = fig.add_subplot(h, w, iax)
        plotter = GraphPlotter(graph)

        plotter.draw_graph_with_solution(ax,node_size=node_size,node_color=color)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_axis_bgcolor([0.9] * 3 + [1.0])
        ax.set_facecolor([0.9] * 3 + [1.0])
        ax.set_title("Ground truth\nSolution length: {:d}".format(t_graphs.globals.numpy()[0,0]))

        # Prediction.
        for k, outp in enumerate(out_graphs):
            iax = i * (1 + num_steps_to_plot) + 2 + k
            ax = fig.add_subplot(h, w, iax)
            color = {}
            prob = softmax_prob_last_dim(outp.nodes.numpy())
            for n in graph.nodes:
                color[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0]) * (1.0 - min_c) + min_c
            plotter.draw_graph_with_solution(ax,node_size=node_size, node_color=color)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(step_indices[k] + 1, step_indices[-1] + 1))
            ax.set_axis_off()

    fig.tight_layout()
    if savefile is not None:
        fig.savefig(savefile,dpi=150)
        pyplot.close(fig)

def plot_losses(train_loss,test_loss,train_correct,test_correct,train_solved,test_solved,title=None,savefile=None):
    fig = pyplot.figure(1, figsize=(18, 3))
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

def softmax_prob_last_dim(x):
  e = np.exp(x)
  return e[:, -1] / np.sum(e, axis=-1)