import pickle

import matplotlib.pyplot as plt
import csv

from minigrid_experiment import baseline_results_file, baseline_pickle_file, UnitRewarder

if __name__ == "__main__":

    with open(baseline_pickle_file('MiniGrid-SpotterLevel2-v0', '57fd422bc05642deaca651260c0ef0f6', True, False), "rb") as file:
        eps, learner = pickle.load(file)
        print("HELLO")

    operator_filename = baseline_results_file('MiniGrid-SpotterLevel2-v0', '57fd422bc05642deaca651260c0ef0f6', True, False)

    plot_avg = 100
    rewards = []
    episodes = []
    reward_sum = 0.
    with open(operator_filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            episode, steps, reward, epsilon = row
            reward_sum += float(reward)
            if int(episode) != 0 and int(episode) % plot_avg == 0:
                rewards.append(reward_sum/plot_avg)
                episodes.append(int(episode))
                reward_sum = 0.

    plt.plot(episodes, rewards)
    plt.show()