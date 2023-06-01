import os

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from datetime import datetime
import keyboard
import time
import pickle


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()


action = 0
wait = True


def press(x):
    global wait
    global action
    if x.event_type == "down" and x.name == 'q':
        action = 0
        wait = False
    elif x.event_type == "down" and x.name == 'e':
        action = 1
        wait = False
    elif x.event_type == "down" and x.name == 'w':
        action = 2
        wait = False
    elif x.event_type == "down" and x.name == 'a':
        action = 3
        wait = False
    elif x.event_type == "down" and x.name == 'd':
        action = 5
        wait = False
    else:
        wait = True


def test(env_id):
    print("Briefing\n")
    print("Do not take this study if you have done a similar one recently\n")
    print("You are free to leave at any time\n")
    print("This study will take around 10 minutes\n")
    print("You need to control an agent in a grid world\n")
    print("Please use the key to open the door and then get to the goal\n")
    print("Thanks for your participation!\n")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_dir = os.path.join('data_human', timestamp)
    os.mkdir(data_dir)

    aid = input("Your assigned ID: ")
    age = input("Your age: ")
    rl_level = input("Are you familiar with Reinforcement Learning? (yes / no): ")
    taks_level = input("Are you familiar with the DoorKey task? (yes / no): ")

    data = {
        "aid": aid,
        "age": age,
        "rl_level": rl_level,
        "taks_level": taks_level,
    }
    save_variable(data, os.path.join(data_dir, 'info.data'))

    global action
    global wait
    env = gym.make(env_id, render_mode="human", agent_pov=True)
    env = FlatObsWrapper(env)
    keyboard.hook(press)
    return_list = []
    obs_list = []
    action_list = []

    print("--------------- Play (Test: 5 round) ---------------")

    for seed in range(5):
        print("[ " + str(seed+1) + " round ]")
        return_list.append(0)
        obs_list.append([])
        action_list.append([])
        env.reset(seed=seed+100)
        terminated = False
        env.render()
        while not terminated:
            while wait:
                pass
            wait = True
            obs, reward, terminated, truncated, info = env.step(action)
            return_list[-1] += reward
            obs_list[-1].append(obs)
            action_list[-1].append(action)
            env.render()
            if terminated:
                print(" @Success")

    data = {
        "return_list": return_list,
        "obs_list": obs_list,
        "action_list": action_list,
    }
    print("[ Return: " + str(return_list) + " ]")
    save_variable(data, os.path.join(data_dir, 'test.data'))

    print("--------------- Play (Official: 10 round) ---------------")

    for seed in range(10):
        print("[ " + str(seed+1) + " round ]")

        return_list.append(0)
        obs_list.append([])
        action_list.append([])
        env.reset(seed=seed)
        terminated = False
        env.render()
        while not terminated:
            while wait:
                pass
            wait = True
            obs, reward, terminated, truncated, info = env.step(action)
            return_list[-1] += reward
            obs_list[-1].append(obs)
            action_list[-1].append(action)
            env.render()
            if terminated:
                print(" @Success")

    data = {
        "return_list": return_list,
        "obs_list": obs_list,
        "action_list": action_list,
    }
    print("[ Return: " + str(return_list) + " ]")
    save_variable(data, os.path.join(data_dir, 'official.data'))

    print("\n------------------- END -------------------\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MiniGrid-DoorKey-6x6-v0')
    args = parser.parse_args()

    test(args.env_id)
