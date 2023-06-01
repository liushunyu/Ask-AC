import os
from datetime import datetime

from utils.default_env import get_env
from utils.default_model import get_model
from utils.human_policy import get_human_policy

import pickle


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()


def train(env_id, mode, level, algo, seed, log_dir, random_ratio, use_baseline_ask, use_ask_loss, ask_threshold):
    print("Briefing\n")
    print("Do not take this study if you have done a similar one recently\n")
    print("You are free to leave at any time\n")
    print("This study will take around 1 hour\n")
    print("You need to provide the action feed back to assist an agent in a grid world\n")
    print("Thanks for your participation!\n")

    aid = input("Your assigned ID: ")
    age = input("Your age: ")
    rl_level = input("Are you familiar with Reinforcement Learning? (yes / no): ")
    taks_level = input("Are you familiar with the DoorKey task? (yes / no): ")

    env = get_env(env_id, mode, level[0], seed, use_baseline_ask, algo)
    model, total_timesteps = get_model(env, env_id, mode, algo, use_baseline_ask, use_ask_loss, ask_threshold, seed, log_dir)

    if args.use_baseline_ask != '':
        model.use_ask_loss = False

    if mode == 'human':
        model.human_policy = get_human_policy(env_id + '_teacher_' + level[0], random_ratio)

    model.learn(total_timesteps=total_timesteps)

    for i in range(len(level)):
        if i == 0:
            continue

        env = get_env(env_id, mode, level[i], seed, use_baseline_ask, algo)

        model.set_env(env)
        model.env.reset()

        if mode == 'human':
            model.human_policy = get_human_policy(env_id + '_teacher_' + level[i])

        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    model.save(os.path.join(log_dir, 'final_model'))

    data = {
        "aid": aid,
        "age": age,
        "rl_level": rl_level,
        "taks_level": taks_level,
    }
    save_variable(data, os.path.join(log_dir, 'info.data'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MiniGrid-DoorKey')
    parser.add_argument('--mode', type=str, default='human')
    parser.add_argument('--level', type=str, default='[medium]')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', '-s', type=str, default='[401]')
    parser.add_argument('--exp', '-e', type=str, default='debug')
    parser.add_argument('--random_ratio', '-r', type=float, default=0)
    parser.add_argument('--use_baseline_ask', '-ba', type=str, default='')
    parser.add_argument('--use_ask_loss', action="store_false", default=True)
    parser.add_argument('--ask_threshold', '-th', type=float, default=0.6)
    parser.add_argument('--info', '-i', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if args.use_baseline_ask != '':
        assert args.use_baseline_ask in ['cm', 'diff', 'var'], 'use_baseline_ask not exist'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    level = list(map(str, args.level.strip('[]').split(',')))
    seed = list(map(int, args.seed.strip('[]').split(',')))

    log_datetime = datetime.now()

    for i in range(len(seed)):
        log_exp_name = 'debug'
        if args.exp == 'run':
            log_exp_name = args.env_id + '_' + args.mode + '_' + args.level + '_' + args.algo
            if args.mode == 'human':
                if args.random_ratio != '0':
                    log_exp_name = log_exp_name + '_r' + str(args.random_ratio)
                if args.use_baseline_ask == '':
                    if not args.use_ask_loss:
                        log_exp_name = log_exp_name + '_woselector'
                elif args.use_baseline_ask != '':
                    log_exp_name = log_exp_name + '_' + args.use_baseline_ask
                    if args.use_baseline_ask == 'diff' or args.use_baseline_ask == 'var':
                        log_exp_name = log_exp_name + '_th' + str(args.ask_threshold)
            if args.info != '':
                log_exp_name = log_exp_name + '_' + args.info
            log_exp_name = log_exp_name + '_' + str(seed[i])
        log_date_dir = os.path.join('exp_human', log_datetime.strftime('%Y-%m-%d_') + log_exp_name)
        log_dir = os.path.join(log_date_dir, log_datetime.strftime('%H-%M-%S'))

        train(args.env_id, args.mode, level, args.algo, seed[i], log_dir, args.random_ratio,
              args.use_baseline_ask, args.use_ask_loss, args.ask_threshold)

