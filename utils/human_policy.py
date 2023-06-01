from stable_baselines3 import PPO
import numpy as np
from datetime import datetime
import os
import pickle


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()

def get_human_policy(exp_name, random_ratio=0):
    use_human_advisor = True

    if use_human_advisor:
        def human_policy(obs, return_is_random=False):
            wait = True
            while wait:
                action = input("Your action: ")
                if action == 'q':
                    action = 0
                    wait = False
                elif action == 'e':
                    action = 1
                    wait = False
                elif action == 'w':
                    action = 2
                    wait = False
                elif action == 'a':
                    action = 3
                    wait = False
                elif action == 'd':
                    action = 5
                    wait = False
                else:
                    print("Illegal action!")
                    wait = True

            return action

    else:
        policy_path = None
        model = PPO.load(policy_path)
        
        def human_policy(obs, return_is_random=False):
            is_random = np.random.rand(obs.shape[0]) < random_ratio
            action, _ = model.predict(obs.cpu(), deterministic=False)
            random_action = np.random.randint(low=0, high=model.action_space.n, size=obs.shape[0])
            action = random_action * is_random + action * (~is_random)
        
            if return_is_random:
                return action, is_random
        
            return action

    return human_policy


