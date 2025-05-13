import copy
import warnings
import random
from typing import Dict, List
import json
import networkx as nx
warnings.filterwarnings('ignore')
import gym
import os
import yaml
import pandas
import openpyxl
import csv
import tqdm
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from functools import partial
import hydra
import numpy as np
import torch
import time
import dashscope
import utils
from agent.logger import Logger
from agent.algorithm import PPOAlgorithm, BaseAlgorithm
from agent.constant import TASKS
from agent.model import PPOModel, BaseModel
from agent.sample import sample_rollouts
from agent.storage import RolloutStorage
from agent.wrapper import VecPyTorch
import wandb
import sys
from collections import deque
from text_crafter import constants
torch.backends.cudnn.benchmark = True
from text_crafter.text_env import BaseTextEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from parse_utils import *
from language_model import GPTLanguageModel
from language_model import BulletPrompt
from transformers import AutoTokenizer
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image
import torch.nn.functional as F
from PIL import Image

ACTION_NAMES = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep", "place_stone",
                "place_table", "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
                "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword"]


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.sbert_path, use_fast=True)
        self.step_num = 1
        self.env_spec = cfg.env_spec


        config_file = open(f"{self.cfg.ppo_config}.yaml", "r")
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.ppo_config = config

        # Create logger
        group_name = f"ppo-debug"
        run_name = f"causal_goal_{group_name}-s{cfg.seed:02}"

        if self.ppo_config['log_stats']:
            # JSON
            log_dir = os.path.join("./logs", run_name)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "stats.jsonl")
            self.log_file = open(log_path, "w")

            # W&B
            self.logger = Logger(config=config, group=group_name, name=run_name, use_wandb=True)

        # Create checkpoint directory
        if self.ppo_config['save_ckpt']:
            self.ckpt_dir = os.path.join("./models", run_name)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Language model setup.
        start_time = time.time()
        prompt_format = BulletPrompt()
        self.lm = GPTLanguageModel(prompt_format=prompt_format, **cfg.env_spec.lm_spec)
        print("load language model cost time: ", time.time() - start_time)
        self.evcg_matrix, self.evcg_objects = None, None
        self.already_verified_causal_relations = []
        self.already_complex = []
        self.past_actions_str = {i: '<null>' for i in range(config["nproc"])}
        self.past_goals_str = {i: '<null>' for i in range(config["nproc"])}
        self.query_interval = 100
        # create envs
        start_time = time.time()
        seeds = np.random.randint(0, 2 ** 31 - 1, size=config["nproc"])
        env_fns = [partial(BaseTextEnv, seed=seed, use_sbert=cfg.env_spec.use_sbert,
                           max_seq_len=cfg.env_spec.max_seq_len, tokenizer=self.tokenizer) for seed in seeds]
        venv = SubprocVecEnv(env_fns)
        venv = VecMonitor(venv)
        self.venv = VecPyTorch(venv, device=self.device)
        print("create envs cost time: ", time.time() - start_time)
        start_time = time.time()
        full_obs = self.venv.reset()
        self.query(full_obs, epoch=0)
        print("query cost time: ", time.time() - start_time)
        # Create model
        start_time = time.time()
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            **config["model_kwargs"],
            device=self.device,
            sbert_path=self.cfg.sbert_path
        )
        self.model = model.to(self.device)
        print("create model cost time: ", time.time() - start_time)
        print(model)

        # Create algorithm
        algorithm_cls = getattr(sys.modules[__name__], config["algorithm_cls"])
        self.algorithm: BaseAlgorithm = algorithm_cls(
            model=model,
            **config["algorithm_kwargs"],
        )

        # CLIP
        start_time = time.time()
        self.clip = pipeline(task=Tasks.multi_modal_embedding, model='damo/multi-modal_clip-vit-large-patch14_zh', model_revision='v1.0.1')
        print("load clip model cost time: ", time.time() - start_time)

        # Create storage
        self.storage = RolloutStorage(
            nstep=config["nstep"],
            nproc=config["nproc"],
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            hidsize=config["model_kwargs"]["hidsize"] * 2,
            device=self.device,
            tg_max_seq_len=cfg.env_spec.max_seq_len,
        )

        obs, text_obs_emd, goals_emd, text_obs_des, goals_str = self.extract_info_from_obs(full_obs)
        # copy 0th
        self.storage.obs[0].copy_(obs)
        self.storage.text_obs_emd[0].copy_(text_obs_emd)
        self.storage.goals_emd[0].copy_(goals_emd)
        self.storage.text_obs_des[0] = text_obs_des
        self.storage.goal_str[0] = goals_str

    def save_causal_graph(self, evcg_matrix, evgc_objects, need_output=False, stage: str = None):

        print("save causal graph successfully")
        print(f'Causal graph {stage} LLM correct is :')

        if need_output:
            size = len(evgc_objects)
            max_len = max(len(obj) for obj in evgc_objects)

            print("\n" + " " * (max_len + 1), end="")
            for obj in sorted(evgc_objects, key=evgc_objects.get):
                print(f"{obj:>{max_len}}", end=" ")
            print()

            # 打印矩阵内容
            for obj, index in sorted(evgc_objects.items(), key=lambda item: item[1]):
                print(f"{obj:>{max_len}}", end=" ")
                for j in range(size):
                    print(f"{evcg_matrix[index, j]:>{max_len}}", end=" ")
                print()

    def pad_sbert(self, input_arr):
        """Pad array to max seq length"""
        arr = np.zeros(self.env_spec.max_seq_len, dtype=int)
        if len(input_arr) > self.env_spec.max_seq_len:
            input_arr = input_arr[:self.env_spec.max_seq_len]
        arr[:len(input_arr)] = input_arr
        return arr

    def combine_obs_action_str(self, obs, action):
        full_info = obs + "\n" + "Player's action:<" + action + ">"
        return full_info

    def find_all_paths(self, G, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in G:
            return []
        paths = []
        for node in G[start]:
            if node not in path:
                new_paths = self.find_all_paths(G, node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def save_complex_graph(self, G):
        dir = ''
        labels = {v: k for k, v in self.evcg_objects.items()}

        # Collect nodes that have edges
        nodes_with_edges = set()
        for u, v in G.edges():
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)

        # Create a subgraph that only includes these nodes
        sub_G = G.subgraph(nodes_with_edges)

        # Compute layout for the subgraph
        pos = graphviz_layout(sub_G, prog="dot")

        # Filter labels to include only nodes in sub_G
        sub_labels = {k: labels[k] for k in sub_G.nodes}

        node_size = 200
        font_size = 8
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(sub_G, pos, node_color='skyblue', node_size=node_size)
        nx.draw_networkx_edges(sub_G, pos, edge_color='k')
        nx.draw_networkx_labels(sub_G, pos, labels=sub_labels, font_size=font_size)
        plt.savefig(dir)

    def extract_complex_relation(self):
        G = nx.DiGraph()
        rows, cols = np.where(self.evcg_matrix == 1)

        # Add all nodes to ensure they are included
        for node in range(len(self.evcg_objects)):
            if not G.has_node(node):
                G.add_node(node)

        for i, j in zip(rows, cols):
            if not G.has_edge(i, j):
                G.add_edge(i, j)

        # Only consider nodes with edges
        nodes_with_edges = set()
        for u, v in G.edges():
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)

        sub_G = G.subgraph(nodes_with_edges)

        all_paths = []
        for start in nodes_with_edges:
            for end in nodes_with_edges:
                if start != end:
                    paths = self.find_all_paths(sub_G, start, end)
                    all_paths.extend(paths)

        index_to_object = {v: k for k, v in self.evcg_objects.items()}

        complex_relation = []
        for path in all_paths:
            complex_relation.append([index_to_object[v] for v in path])

        complex_relation = [item for item in complex_relation if len(item) >= 2]

        return complex_relation, sub_G

    def extract_relevant_achievements(self, causality, achievement):
        # Define a list to store relevant achievements
        relevant_achievements = []

        # Create a set of relevant achievement keys based on causality
        relevant_keys = set()
        for item in causality:
            cause, effect = item
            relevant_keys.add(cause)
            relevant_keys.add(effect)

        # Check achievements for relevance
        for key, value in achievement.items():
            if any(relevant_key in key for relevant_key in relevant_keys):
                relevant_achievements.append(key)

        return relevant_achievements

    def query(self, full_obs, epoch, achievement=None):

        if (epoch+1) % 2 == 0:
            complex_relation, G = self.extract_complex_relation()
            if complex_relation is not None:
                print(f'find complex relation: {complex_relation}')
            self.save_complex_graph(epoch, G)
            for index, item in enumerate(full_obs):
                text_obs = item['text_obs_backup']
                inv_status = item['inv_status_backup']

                goal_str = self.lm.query_goal_for_complex_relation(complex_relation,
                                                                  {'obs': text_obs,
                                                                   "past_goals": self.past_goals_str[index],
                                                                   "past_action": self.past_actions_str[index],
                                                                   **inv_status},achievement[index])
                self.past_goals_str[index] = goal_str
                item['goal'] = goal_str
        else:
            for index, item in enumerate(full_obs):
                text_obs = item['text_obs_backup']
                inv_status = item['inv_status_backup']
                str_graph = self.lm.query({'text_obs': text_obs, **inv_status})
                if self.evcg_matrix is None or self.evcg_objects is None:
                    first_causal_dict = parse_causal_description(str_graph, text_obs, inv_status)
                    self.evcg_matrix, self.evcg_objects = build_causal_matrix(first_causal_dict)
                else:
                    self.evcg_matrix, self.evcg_objects = update_causal_matrix(self.evcg_matrix, self.evcg_objects,
                                                                               str_graph, text_obs, inv_status)
                relation = causal_matrix_to_text(self.evcg_matrix, self.evcg_objects)
                if achievement is not None:
                    related_achieve = self.extract_relevant_achievements(relation, achievement[index])
                    goal_str = self.lm.predict_options({'obs': text_obs, "causality": relation,   # TODO LLM2
                                                        "past_goals": self.past_goals_str[index],
                                                        "past_action": self.past_actions_str[index], **inv_status}, related_achieve)
                else:
                    goal_str = self.lm.predict_options({'obs': text_obs, "causality": relation,
                                                        "past_goals": self.past_goals_str[index],
                                                        "past_action": self.past_actions_str[index], **inv_status})
                self.past_goals_str[index] = goal_str
                item['goal'] = goal_str
        return full_obs
    def extract_info_from_obs(self, full_obs, type='muti'):
        obs = []
        text_obs_emd = []
        text_obs_des = []
        goals_emd = []
        goals_str = []
        if type == 'muti':
            for item in full_obs:
                obs.append(item['obs'])
                text_obs_emd.append(item['text_obs'])
                inv_status = item['inv_status_backup']
                goals_str.append(item['goal'])
                text_obs_des.append(item['text_obs_backup'] + "\n" + inv_status['status'] + "\n" + inv_status['inv'])
                goals_emd.append(self.pad_sbert(np.array(self.tokenizer(item['goal'])['input_ids'])))
        else:
            obs.append(torch.tensor(full_obs['obs'], dtype=torch.float32).permute(2, 0, 1))
            text_obs_emd.append(full_obs['text_obs'])
            inv_status = full_obs['inv_status_backup']
            goals_str.append(full_obs['goal'])
            text_obs_des.append(full_obs['text_obs_backup'] + "\n" + inv_status['status'] + "\n" + inv_status['inv'])
            goals_emd.append(self.pad_sbert(np.array(self.tokenizer(full_obs['goal'])['input_ids'])))
        obs = torch.stack(obs, dim=0)
        text_obs_emd = torch.from_numpy(np.array(text_obs_emd))
        goals_emd = torch.from_numpy(np.array(goals_emd))
        return obs, text_obs_emd, goals_emd, text_obs_des, goals_str



    def valid_relation_through_invention(self, model, need_valid_relation_list, epoch):

        sees = ['grass', 'tree', 'lava', 'path', 'sand']
        damage = ['zombie', 'skeleton']
        health = ['cow', 'water', 'plant']
        state = ['health', 'food', 'drink', 'energy']
        matrial = ['wood', 'stone', 'coal', 'iron', 'sapling', 'diamond']
        see_obj = ['table', 'furnace', 'plant']
        tool = ['wood_pickaxe', 'wood_sword', 'stone_pickaxe', 'stone_sword', 'iron_pickaxe', 'iron_sword']
        success_verify_list = [False for _ in range(len(need_valid_relation_list))]

        seed = np.random.randint(0, 2 ** 31 - 1)
        env = BaseTextEnv(seed=seed, use_sbert=self.cfg.env_spec.use_sbert,
                          max_seq_len=self.cfg.env_spec.max_seq_len, tokenizer=self.tokenizer)

        for index, (cause, effect) in enumerate(need_valid_relation_list):

            full_obs, init_inventory, local_canvas = env.reset_for_specific_env(cause=cause, effect=effect,
                                                                                stage='valid')
            cause_number = init_inventory[cause] if cause in init_inventory else 0
            effect_number = init_inventory[effect] if effect in init_inventory else 0

            for i in range(200):
                current_obs, current_status, current_inventory = full_obs['text_obs_backup'], \
                full_obs['inv_status_backup']['status'], full_obs['inv_status_backup']['inv']

                input = current_obs + "\n" + current_status + "\n" + current_inventory + "\n"
                input += f"Uncertain relation: {cause} -> {effect}" + '\n'
                input += "\nBased on the input, provide one goal from Available goals that helps the player verify the uncertain relation."
                input += "\nDo not add or explain any additional words!!"
                goal_i = self.lm.query_goal(input, 'init')
                if goal_i is None:
                    goal_i = "find cause"

                full_obs['goal'] = goal_i

                model.eval()
                obs, text_obs_emd, goals_emd, text_obs_des, _ = self.extract_info_from_obs(full_obs, type='single')

                obs = obs.to(self.device)
                text_obs_emd = text_obs_emd.to(self.device)
                goals_emd = goals_emd.to(self.device)
                # get action
                action = model.act({'obs': obs, "text_obs_emd": text_obs_emd, "goals_emd": goals_emd})['actions']
                current_action = ACTION_NAMES[action.item()]


                if current_action.startswith('place_') and {cause, effect}.issubset(constants.place) and cause in constants.place[effect]['uses']:
                    cost = constants.place[effect]['uses'][cause]
                elif current_action.startswith('make_') and effect in constants.make and cause in constants.make[effect]['uses']:
                    cost = constants.make[effect]['uses'][cause]
                else:
                    cost = 10

                # take action in the environment
                full_obs, reward, dones, info = env.step(action)

                if (cause in health and effect in state and cause == info['facing_obj_before'] and info['inventory'][effect] > effect_number) or \
                        (cause in sees and effect in matrial and cause == info['facing_obj_before'] and info['inventory'][effect] > effect_number) or \
                        (cause in matrial and effect in tool and cause_number - info['inventory'][cause] == cost and info['inventory'][effect] > effect_number) or \
                        (cause in matrial and effect in see_obj and cause_number - info['inventory'][cause] == cost and effect == info['facing_obj_after']) or\
                        (cause in damage and effect in state and info['inventory'][effect] < effect_number) or \
                        (cause in tool and effect in matrial and effect not in current_inventory and effect in info['facing_obj_before'] and info['inventory'][effect] > effect_number):
                    print(f'relation: {cause}->{effect} is verified')
                    print(f'cost {i} steps')
                    success_verify_list[index] = True

                    break

                if cause in matrial:
                    cause_number = info['inventory'][cause]

                if effect in matrial or effect in tool or effect in state:
                    effect_number = info['inventory'][effect]

        print('-*/-*/-*/-*/-*/-*/-*/ valid over -*/-*/-*/-*/-*/-*/-*/ ')
        return success_verify_list

    def train(self):
        total_successes = np.zeros((0, len(TASKS)), dtype=np.int32)

        ckpt_dir = ''
        start_epoch = 0

        if os.path.exists(ckpt_dir) and self.cfg.use_ckpt:
            checkpoint = torch.load(ckpt_dir)
            self.model.load_state_dict(checkpoint)
            start_epoch = (re.search(r'agent-e(\d+)\.pt', ckpt_dir)).group(1)
            start_epoch = int(start_epoch) + 1
            print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        else:
            print("Without checkpoint, starting from scratch")

        # past_new_list = []
        past_true_list = []
        past_false_list = []
        error_count = {}
        total_reward = 0


        for epoch in range(start_epoch, self.ppo_config["nepoch"] + 1):

            print(f"\nepoch {epoch}:")

            # Sample episodes
            print('START SAMPLE')
            start_time = time.time()
            rollout_stats = self.sample_rollouts(epoch, self.venv, self.model, self.storage)
            print('episode_rewards', rollout_stats['episode_rewards'])
            mean_reward = np.mean(rollout_stats['episode_rewards'])
            print('mean reward:', mean_reward)
            total_reward += mean_reward
            print("sample_rollouts cost time:", time.time() - start_time)

            # Compute returns
            self.storage.compute_returns(self.ppo_config["gamma"], self.ppo_config["gae_lambda"])

            # Update models
            start_time = time.time()
            train_stats = self.algorithm.update(self.storage)
            print("ppo update cost time:", time.time() - start_time)

            confusion_causal_relation = rollout_stats['confusion_causal_relations']
            print(f'confusion causal relation before: {confusion_causal_relation}')

            for true in past_true_list:
                confusion_causal_relation.remove(true) if true in confusion_causal_relation else None

            for false in past_false_list:
                confusion_causal_relation.remove(false) if false in confusion_causal_relation else None

            if (epoch+1) % 5 == 0:
                for item in past_false_list:
                    confusion_causal_relation.append(item) if item not in confusion_causal_relation else None

            for item in confusion_causal_relation:
                if item is not None and (error_count.get(item) or 0) > 5:
                    confusion_causal_relation.remove(item)
                    print(f'remove {item} because error count is {error_count.get(item)}')

            print(f'confusion causal relation after: {confusion_causal_relation}')

            if len(confusion_causal_relation) > 0:

                self.save_causal_graph(self.evcg_matrix, self.evcg_objects, need_output=True, stage='before')

                print('START VALID')
                start_time = time.time()
                res = self.valid_relation_through_invention(self.model,confusion_causal_relation,epoch)

                print('valid causal relation in env cost time:', time.time() - start_time)

                self.evcg_matrix, self.evcg_objects, self.already_verified_causal_relations, true_list, false_list = parse_causal_relation_with_index(
                    res, confusion_causal_relation,
                    self.evcg_matrix, self.evcg_objects, self.already_verified_causal_relations)
                print(f'true list:{true_list} \n')
                print(f'false list:{false_list} \n')


                for item in true_list:
                    past_true_list.append(item) if item not in past_true_list else None
                for item in false_list:
                    past_false_list.append(item) if item not in past_false_list  else None
                    if item not in error_count:
                        error_count[item] = 1
                    else:
                        error_count[item] += 1

                self.save_causal_graph(self.evcg_matrix, self.evcg_objects, need_output=True, stage='after')

            else:
                print('No relation need to be verified')

            # Reset storage
            self.storage.reset()

            # Compute score
            successes = rollout_stats["successes"]
            total_successes = np.concatenate([total_successes, successes], axis=0)
            success_rate = 100 * np.mean(total_successes, axis=0)
            score = np.exp(np.mean(np.log(1 + success_rate))) - 1

            # Get eval stats
            eval_stats = {
                "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
                "score": score,
            }

            # Print stats
            print(json.dumps(train_stats, indent=2))
            print(json.dumps(eval_stats, indent=2))
            print('total reward:', total_reward)

            # Log stats
            if self.ppo_config['log_stats']:
                # W&B
                self.logger.log(train_stats, epoch)
                self.logger.log(eval_stats, epoch)

            # Save checkpoint
            if self.ppo_config['save_ckpt'] and epoch % self.ppo_config["save_freq"] == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"agent-e{epoch:03}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

            print("score:", score)
            print("reward:", mean_reward)

    def sample_rollouts(self,
                        epoch,
                        env,
                        model: BaseModel,
                        storage: RolloutStorage,
                        ) -> Dict[str, np.ndarray]:
        # Set model to eval model
        model.eval()

        # Sample rollouts
        episode_rewards = []
        episode_lengths = []
        achievements = []
        successes = []

        # 记录需要验证的因果关系
        confusion_causal_relations = []
        time_complex = 0

        for step in range(storage.nstep):

            # Pass through model
            inputs = storage.get_inputs(step)
            outputs = model.act(inputs)

            actions = outputs["actions"]

            for index, action in enumerate(actions):
                self.past_actions_str[index] = ACTION_NAMES[action.cpu().item()]

            # Step environment
            full_obs, rewards, dones, infos, achieve_list = env.step(actions)
            for index, item in enumerate(full_obs):
                current_obs = (item['obs'] * 255).cpu().numpy().astype(np.uint8)
                current_obs = current_obs.transpose(1, 2, 0)
                current_obs_img = Image.fromarray(current_obs)


                current_goal = self.past_goals_str[index]
                img_embedding = self.clip.forward({'img': current_obs_img})['img_embedding']
                goal_embedding = self.clip.forward({'text': current_goal})['text_embedding']
                cos_scores = F.cosine_similarity(img_embedding, goal_embedding)
                if cos_scores > 0.5:
                    rewards[index] += cos_scores
                    print(f'cos_score for env {index}: {cos_scores.item()}')

            transition_text_desc_list = []
            for index, item in enumerate(full_obs):
                transition_text_desc = self.combine_obs_action_str(inputs['text_obs_des'][index],
                                                                   self.past_actions_str[index])
                transition_text_desc_list.append(transition_text_desc)
            storage.insert_transition_text_obs(step, transition_text_desc_list)


            if self.step_num % self.query_interval == 0:
                start_query_time = time.time()
                self.query(full_obs, epoch, achieve_list)   # TODO LLM1
                time_complex += time.time() - start_query_time
            else:
                current_sub_goals = []
                for index, item in enumerate(full_obs):
                    goal_str = self.past_goals_str[index]
                    item['goal'] = goal_str
                    current_sub_goals.append(goal_str)
            self.step_num += 1
            obs, text_obs_emd, goals_emd, text_obs_des, goals_str = self.extract_info_from_obs(full_obs)

            outputs["obs"] = obs
            outputs["rewards"] = rewards
            outputs["masks"] = 1.0 - dones
            outputs["text_obs_emd"] = text_obs_emd
            outputs["goals_emd"] = goals_emd
            outputs["text_obs_des"] = text_obs_des
            outputs['goal_str'] = goals_str
            outputs["successes"] = infos["successes"]

            # Update storage
            storage.insert(**outputs, model=model)

            # Update stats
            for i, done in enumerate(dones):
                if done:
                    # Episode lengths
                    episode_length = infos["episode_lengths"][i].cpu().numpy()
                    episode_lengths.append(episode_length)

                    # Episode rewards
                    episode_reward = infos["episode_rewards"][i].cpu().numpy()
                    episode_rewards.append(episode_reward)

                    # Achievements
                    achievement = infos["achievements"][i].cpu().numpy()
                    achievements.append(achievement)

                    # Successes
                    success = infos["successes"][i].cpu().numpy()
                    successes.append(success)

                    causal_matrix = copy.deepcopy(self.evcg_matrix)

                    str_graph = causal_matrix_to_text(causal_matrix, self.evcg_objects)

                    for item in str_graph:
                        confusion_causal_relations.append(item) if item is not None and item not in confusion_causal_relations else None

        print(f'query complex cost time: {time_complex}')

        inputs = storage.get_inputs(step=-1)
        # policy output
        outputs = model.act(inputs)
        vpreds = outputs["vpreds"]

        # Update storage
        storage.vpreds[-1].copy_(vpreds)

        # Stack stats
        episode_lengths = np.stack(episode_lengths, axis=0).astype(np.int32)
        episode_rewards = np.stack(episode_rewards, axis=0).astype(np.float32)
        achievements = np.stack(achievements, axis=0).astype(np.int32)
        successes = np.stack(successes, axis=0).astype(np.int32)

        confusion_causal_relations = list(set(confusion_causal_relations))
        for (obj1, obj2) in confusion_causal_relations:
            idx1 = self.evcg_objects.get(obj1)
            idx2 = self.evcg_objects.get(obj2)
            if idx1 is not None and idx2 is not None and self.evcg_matrix[idx1, idx2] == 0:
                confusion_causal_relations.remove((obj1, obj2))

        # Define rollout stats
        rollout_stats = {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "achievements": achievements,
            "successes": successes,
            "confusion_causal_relations": confusion_causal_relations,
        }

        return rollout_stats

root_dir = Path.cwd()


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    from train import Workspace as W
    if cfg.env_spec.lm_spec.api_key is None and not cfg.env_spec.lm_spec.use_local_llm:
        raise ValueError('Please provide an LLM API key')

    workspace = W(cfg)

    if cfg.stage == 'train':
        print("start train")
        workspace.train()
    else:
        print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-start test-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
        checkpoint = torch.load(cfg.expl_agent_path)
        workspace.test(checkpoint)


if __name__ == '__main__':
    dashscope.api_key = ""
    main()
    print('Done!')
