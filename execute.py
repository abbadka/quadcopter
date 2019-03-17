import sys
import pandas as pd
import numpy as np
from agents.ddpg.agent import DDPG
from tasks.maintain_position import MaintainPosition
from tasks.stay_up import StayUp
from math import inf

num_episodes = 100
task = MaintainPosition()
agent = DDPG(task)
agent_best_score = -inf
scores_stay_up = []
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()  # start a new episode
    agent_score = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        agent_score += reward
        if done:
            agent_best_score = max(agent_score, agent_best_score)
            scores_stay_up.append(agent_score)
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent_score, agent_best_score), task.final_position, end="")  # [debug]
            break
    print()
    # sys.stdout.flush()
