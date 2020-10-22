import math
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import seaborn as sns
import pandas

import csv
import random

from minigrid_experiment import get_operator_filename, NUM_TO_NAME_AND_GOAL

if __name__ == "__main__":
    G = nx.DiGraph()

    operator_filename = get_operator_filename('MiniGrid-SpotterLevel2-v0', '6e6ce2caf0bd47d0b1c81e3345c3ccd9')

    pos = {}
    label = {}

    node_data = []

    timestep_dict = {}
    timestep_map = defaultdict(list)
    with open(operator_filename + "_abbrev") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            episode, name, preconds, learner = tuple(row)
            if learner == '4':
                G.add_node(name)
                timestep_map[int(episode)].append(name)
                timestep_dict[name] = int(episode)
                node_data.append((name, {'subset': int(episode)}))
                label[name] = preconds

    max_episode = 70000
    num_ops = [0]
    first_op_episode = -1
    min_preconds = []
    cur_min = math.inf
    precond_scatter_x = []
    precond_scatter_y = []
    precond_scatter_names = []
    for i in range(max_episode):
        if i in timestep_map:
            if first_op_episode == -1:
                first_op_episode = i
            all_num_preconds = []
            for name in timestep_map[i]:
                all_num_preconds.append(int(label[name]))
                precond_scatter_x.append(i)
                precond_scatter_y.append(int(label[name]))
                precond_scatter_names.append(name)
            cur_min = min(cur_min, min(all_num_preconds))
            min_preconds.append(cur_min)
            num_ops.append(num_ops[-1] + len(timestep_map[i]))
        else:
            if first_op_episode != -1:
                min_preconds.append(cur_min)
            num_ops.append(num_ops[-1])
    num_ops = num_ops[1:]

    print(len(precond_scatter_y))
    print(max(precond_scatter_x))
    print(min(precond_scatter_x))

    sns.set()
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    frames = list(range(first_op_episode, 10000, 100)) + list(range(10000, 20000, 500)) + list(range(20000, 32100, 100))
    min_x = min(precond_scatter_y)
    max_x = max(precond_scatter_y)

    ax = sns.lineplot(x=list(range(max_episode)), y=num_ops)
    ax.set(xlabel="Number of episodes", ylabel="Number of operators discovered")
    plt.tight_layout()
    plt.savefig("precond_refinement_num_ops.png")
    plt.show()
    #
    # # ax = sns.lineplot(x=list(range(first_op_episode, 50000)), y=min_preconds)
    # # ax.set(xlabel="Number of episodes", ylabel="Minimal number of preconditions")
    # # plt.show()
    #
    df_scatter = pandas.DataFrame({"preconds": [str(y) for y in precond_scatter_y], "timestep": precond_scatter_x})

    ax = sns.stripplot(data=df_scatter, x="timestep", y="preconds", linewidth=0, palette="dark:b", size=3,
                       order=reversed([str(i) for i in range(126,147)]))
    ax.set(xlabel="Number of episodes", ylabel="Number of preconditions")
    ax.set_yticks(list(range(0, 20, 2)))
    ax.grid(b=True, which='major', axis='y')
    plt.tight_layout()
    plt.savefig("precond_refinement_scatter.png")
    plt.show()

    discovered = [timestep_dict[name] for name in precond_scatter_names]
    # extras = list(range(0, 30050,50))
    # extras_val = list(range(150, 751, 1))
    extras = [0, 10000, 20000, 30000]
    extras_val = [150, 151, 152, 153]
    # df1 = pandas.DataFrame({"preconds": precond_scatter_y + extras_val, "discovered": discovered + extras})
    #
    # fig, ax = plt.subplots()
    # sns.color_palette("mako", as_cmap=True)
    # g = sns.histplot(df1, x="preconds", palette="Blues", hue="discovered", discrete=True, multiple="stack",
    #                  legend=True, hue_norm=(0, 30000), ax=ax)
    # handles = ax.legend_.legendHandles
    # labels = ax.legend_.texts
    # ax.legend(handles=[handles[i] for i in range(len(labels)) if int(labels[i].get_text()) % 10000 == 0],
    #           labels=["Ep. {}".format(int(labels[i].get_text()))
    #                   for i in range(len(labels)) if int(labels[i].get_text()) % 10000 == 0])
    # ax.grid(b=False, which='major', axis='x')
    # plt.xlim(min_x - 0.5, max_x + 0.5)
    # plt.xlabel("Number of preconditions")
    # plt.ylabel("Number of operators")
    # plt.xticks(list(range(128, 147, 2)))
    # plt.setp(ax.patches, linewidth=0)
    # plt.tight_layout()
    # plt.savefig("precond_refinement_hist.png")
    # plt.show()

    # Bar chart with both hue for recency, and orange for dominated versions.
    # edges = []
    # dominated = {}
    # with open(operator_filename + "_edge") as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         outgoing, incoming = tuple(row)
    #         if outgoing in timestep_dict and outgoing not in dominated:
    #             dominated[outgoing] = timestep_dict[incoming]
    #         if G.has_node(outgoing):
    #             edges.append((outgoing, incoming))
    #
    # df1 = pandas.DataFrame({"preconds": precond_scatter_y + extras_val, "discovered": discovered +extras,
    #                         "name": precond_scatter_names + [""]*4, "dominated": [x in dominated for x in precond_scatter_names] + [False]*4})
    #
    # df1.loc[df1['dominated'],'discovered'] = 90000
    #
    # fig, ax = plt.subplots()
    # g = sns.histplot(df1, x="preconds", palette="icefire", hue="discovered", discrete=True, multiple="stack",
    #                  legend=True, hue_norm=(0, 110000), ax=ax)
    # handles = ax.legend_.legendHandles
    # labels = ax.legend_.texts
    # new_labels =["Ep. {}".format(int(labels[i].get_text()))
    #                   for i in range(len(labels)) if int(labels[i].get_text()) % 10000 == 0]
    # new_labels[-1]="Dominated"
    # ax.legend(handles=[handles[i] for i in range(len(labels)) if int(labels[i].get_text()) % 10000 == 0],
    #           labels=new_labels)
    # ax.grid(b=False, which='major', axis='x')
    # # for patch in ax.patches[418:]:
    # #     patch.set_color('red')
    # plt.xlim(min_x - 0.5, max_x + 0.5)
    # plt.xlabel("Number of preconditions")
    # plt.ylabel("Number of operators")
    # plt.xticks(list(range(128, 147, 2)))
    # plt.setp(ax.patches, linewidth=0)
    # plt.tight_layout()
    # plt.savefig("precond_refinement_hist.png")
    # plt.show()


    #
    # dist_fig, dist_ax = plt.subplots()
    #
    # plot_x = [x for x in precond_scatter_x if x <= i]
    # plot_y = precond_scatter_y[:len(plot_x)]
    # plot_names = precond_scatter_names[:len(plot_x)]
    # is_dominated = [plot_names[x] in dominated and dominated[plot_names[x]] < i for x in range(len(plot_names))]
    # df = pandas.DataFrame({"preconds": plot_y, "dominated": is_dominated})
    # plt.clf()
    # plt.xlim(min_x-0.5, max_x+0.5)
    # plt.xlabel("Number of preconditions")
    # plt.ylabel("Number of operators")
    #
    # plt.title("Episode {}".format(i))
    # graph = sns.histplot(data=df, x="preconds", hue="dominated", stat="count", discrete=True, multiple="stack")

    #
    # dist_fig, dist_ax = plt.subplots()
    #
    # def animate(i):
    #     plot_x = [x for x in precond_scatter_x if x <= i]
    #     plot_y = precond_scatter_y[:len(plot_x)]
    #     plt.clf()
    #     plt.xlim(min_x-0.5, max_x+0.5)
    #     plt.xlabel("Number of preconditions")
    #     plt.ylabel("Number of operators")
    #
    #     plt.title("Episode {}".format(i))
    #     graph = sns.histplot(x=plot_y, stat="count", discrete=True)
    #
    #
    # ani = anim.FuncAnimation(dist_fig, animate, frames=frames)
    # plt.show()
    #
    # edges = []
    #
    # dominated = {}
    # with open(operator_filename + "_edge") as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         outgoing, incoming = tuple(row)
    #         if outgoing in timestep_dict and outgoing not in dominated:
    #             dominated[outgoing] = timestep_dict[incoming]
    #         if G.has_node(outgoing):
    #             edges.append((outgoing, incoming))
    #
    # dist_fig, dist_ax = plt.subplots()
    #
    # def animate(i):
    #     plot_x = [x for x in precond_scatter_x if x <= i]
    #     plot_y = precond_scatter_y[:len(plot_x)]
    #     plot_names = precond_scatter_names[:len(plot_x)]
    #     is_dominated = [plot_names[x] in dominated and dominated[plot_names[x]] < i for x in range(len(plot_names))]
    #     df = pandas.DataFrame({"preconds": plot_y, "dominated": is_dominated})
    #     plt.clf()
    #     plt.xlim(min_x-0.5, max_x+0.5)
    #     plt.xlabel("Number of preconditions")
    #     plt.ylabel("Number of operators")
    #
    #     plt.title("Episode {}".format(i))
    #     graph = sns.histplot(data=df, x="preconds", hue="dominated", stat="count", discrete=True, multiple="stack")
    #
    # Writer = anim.writers["ffmpeg"]
    # writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
    #
    # ani = anim.FuncAnimation(dist_fig, animate, frames=frames, interval=200)
    # ani.save('precond_generalization.mp4', writer=writer)
    # plt.show()




