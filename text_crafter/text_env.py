"""Crafter environment with text observations."""

import string
import numpy as np
import gym
from transformers import AutoTokenizer

from text_crafter import constants
from text_crafter import engine
from text_crafter import objects
from text_crafter.env import Env

STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
VERB_ONLY = ['do nothing', 'move left', 'move right', 'move up', 'move down', 'sleep']
STATUS_THRESHOLD = 9


class BaseTextEnv(Env):
    """Base text environment for running baselines, where we can get text observations"""

    def __init__(self, use_sbert=False, max_seq_len=100,tokenizer=None, **kwargs):
        super().__init__(**kwargs)

        # Tokenizer to encode all strings
        self.use_sbert = use_sbert
        if use_sbert:
            self.tokenizer = tokenizer

        # Obs configuration
        view = self._view
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._text_view = engine.EmbeddingView(self.world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant], [view[0], view[1] - item_rows])
        self._max_seq_len = max_seq_len
        self._vocab = self._get_action_vocab()

    @property
    def action_space(self):
        """Define the discrete action space.
        With the harder env, each discrete action represents a unique (verb, noun) pair"""
        return super().action_space

    def reset(self):
        """ Reset the env, return a dictionary of tokenized string observations."""
        obs = super().reset()
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs': text_obs,
            'inv_status': inv_status,
            'success': False
        }
        return self.tokenize_obs(obs)
        # return obs

    def reset_for_specific_env(self, stage, cause, effect):
        """ Reset the env, return a dictionary of tokenized string observations."""
        obs, player = super().reset_for_specific(stage, cause, effect)
        text_obs, inv_status, local_canvas= self.text_obs_for_specific()
        obs = {
            'obs': obs,
            'text_obs': text_obs,
            'inv_status': inv_status,
            'success': False
        }
        return self.tokenize_obs(obs), player.inventory, local_canvas

    def step(self, action):
        """ Step the env. Action is passed in as an int."""
        obs, reward, done, info = super().step(action)
        info['env_reward'] = reward
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs': text_obs,
            'inv_status': inv_status,
        }
        tokenize_obs = self.tokenize_obs(obs)
        tokenize_obs['action'] = info['player_action']
        return tokenize_obs, reward, done, info

    def step_specific(self, action, canvas, cause):
        """ Step the env. Action is passed in as an int."""
        obs, reward, done, info = super().step_specific(action, canvas, cause)
        info['env_reward'] = reward
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs': text_obs,
            'inv_status': inv_status,
        }
        tokenize_obs = self.tokenize_obs(obs)
        tokenize_obs['action'] = info['player_action']
        return tokenize_obs, reward, done, info

    @property
    def action_names(self):
        """List of strings, one for each action (including invalid actions)."""
        return super().action_names

    @property
    def good_action_names(self):
        """Return good actions - i.e. actions which are valid in the environment."""
        return constants.good_actions

    def get_action_name(self, action):
        """Get action name. Input is either an int (the action index) or a tuple (verb index, noun index)"""
        return self.action_names[action]

    def check_actions_same(self, action_str_1, action_str_2):
        """Check if two actions are the same. Input is a string"""
        if action_str_1 == 'eat cow' and action_str_2 == 'attack cow' or action_str_1 == 'attack cow' and action_str_2 == 'eat cow':
            return True
        if action_str_1 == 'make crafting table' and action_str_2 == 'place crafting table' or action_str_1 == 'place crafting table' and action_str_2 == 'make crafting table':
            return True
        return action_str_1 == action_str_2

    def text_obs(self):
        """Return a dictionary of text observations"""
        inv, status = self._inventory_to_text()
        obs, _ = self._text_view.local_sentence_view(self.player)
        return obs, {'inv': inv, 'status': status}

    def text_obs_for_specific(self):
        """Return a dictionary of text observations"""
        inv, status = self._inventory_to_text()
        # obs, local_canvas = self._text_view.local_sentence_view_specific(self.player, obj)
        obs, local_canvas = self._text_view.local_sentence_view(self.player)
        return obs, {'inv': inv, 'status': status}, local_canvas

    def _inventory_to_text(self):
        """
        返回库存字符串列表和玩家状态字符串列表。 否则返回一个由库存列表组成的句子： "你有斧头、木材......"，以及状态列表 "你感到饥饿、困倦......"
        Returns a list of strings for the inventory, and list of strings for player status.
        else returns a sentence formed from the inventory lists: "You have axe, wood..", and status lists "You feel hungry, sleepy..."
        """
        inv = []
        status = []
        status_str = "Player status: <"
        inventory_str = "Player inventory: <"
        # Text description only mentions low status items and inventory items the player currently has
        for k, v in self.player.inventory.items():
            if k in STATUS_ITEMS:  # First four elements are status items. Only include status item if it is low.
                status_str += f"{v} {k} , "
                status.append(k)
            elif k not in STATUS_ITEMS and v > 0:  # Only add to inv if we have 1 or more
                inventory_str += f"{v} {k} , "
                inv.append(k)

        status_str += ">"
        inventory_str += ">"
        if len(inv) == 0:
            inventory_str = "Player inventory: <null>"
        return inventory_str, status_str

    def tokenize_str(self, s):
        """Tokenize a string using the vocab index"""
        if self.use_sbert:  # Use SBERT tokenizer
            return np.array(self.tokenizer(s)['input_ids'])
        # Use the vocab index
        arr = np.zeros(self._max_seq_len, dtype=int)
        if " " in s:
            word_list = [w.strip(string.punctuation + ' ').lower() for w in s.split()]
            word_list = [w for w in word_list if len(w) > 0]
        else:
            word_list = [s.lower()]
        assert len(
            word_list) <= self._max_seq_len, f"word list length {len(word_list)} too long; increase max seq length: {self._max_seq_len}"

        for i, word in enumerate(word_list):
            if len(word) == 0:
                continue
            assert word in self._vocab, f"Invalid vocab word: |{word}|. {s}"
            arr[i] = self._vocab.index(word)
        return arr

    def pad_sbert(self, input_arr):
        """Pad array to max seq length"""
        arr = np.zeros(self._max_seq_len, dtype=int)
        if len(input_arr) > self._max_seq_len:
            input_arr = input_arr[:self._max_seq_len]
        arr[:len(input_arr)] = input_arr
        return arr

    def tokenize_obs(self, obs_dict):
        """
        Takes in obs dict and returns a dict where all strings are tokenized.
        """
        new_obs = {}
        new_obs['text_obs_backup'] = obs_dict['text_obs']
        new_obs['inv_status_backup'] = obs_dict['inv_status']
        if self.use_sbert and isinstance(obs_dict['inv_status'], dict):
            inv_status = ""
            for k, v in obs_dict['inv_status'].items():
                if v != '.' and 'null' not in v:
                    inv_status += v + " "
            obs_dict['text_obs'] = obs_dict['text_obs'] + " " + inv_status

        for k, v in obs_dict.items():
            # If the value is a dictionary of strings, concatenate them into a single string
            if isinstance(v, dict):
                v = " ".join([str(val) for val in v.values()])
            # If the value is a string, tokenize it
            if isinstance(v, str):
                arr = self.tokenize_str(v)
                new_obs[k] = arr
            else:
                # Value is already tokenized (int, array, etc)
                new_obs[k] = v
        if self.use_sbert:
            new_obs['text_obs'] = self.pad_sbert(new_obs['text_obs'])
        return new_obs

    def untokenize_arr(self, arr):
        """Takes in an array of tokenized words and returns a string"""
        if self.use_sbert:
            # Trim off zero padding
            arr = arr[:np.argmax(arr == 0)]
            # Trim off the [CLS] token at the beginning and the [SEP] token at the end
            arr = arr[1:-1]
            return self.tokenizer.decode(arr)
        else:
            # 0 is the padding token
            return " ".join([self._vocab[token] for token in arr.tolist() if not token == 0])

    def untokenize_obs(self, obs):
        """" Takes in either a tokenized array or an obs_dict (same as output of tokenize_obs)
        Turns input into strings (or an obs_dict of strings) """
        if isinstance(obs, np.ndarray):
            return self.untokenize_arr(obs)
        assert isinstance(obs, dict)
        new_obs = {}
        for k, v in obs.items():
            if not k == 'obs':
                v = self.untokenize_arr(v)
            new_obs[k] = v
        return new_obs

    def _get_action_vocab(self):
        """Create a list of all possible vocab words."""
        # split string is the transformers library split token
        self.split_str = ' [SEP] '
        vocab = {self.split_str}
        vocab.update("you have in your inventory".split())
        vocab.update("you feel hurt hungry thirsty sleepy".split())
        vocab.update("you see".split())
        vocab.update("you are targeting".split())
        vocab.update('arrow player and'.split())
        vocab.update(constants.materials)

        split_actions = [ac.split() for ac in constants.actions]
        split_actions = [item for sublist in split_actions for item in sublist]

        vocab.update(split_actions)
        vocab.update(constants.walkable)
        vocab.update(constants.items.keys())
        vocab.update(constants.collect.keys())
        vocab.update(constants.place.keys())
        vocab.update(constants.make.keys())
        vocab.update(constants.achievements)

        vocab_list = ['null'] + sorted(list(vocab))
        vocab_list.append('sep')
        return vocab_list


class CustomTextEnv(BaseTextEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, confusion_relation):
        obs = super().reset()
        self.add_confusion_relation_objects(confusion_relation)
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs': text_obs,
            'inv_status': inv_status,
            'success': False
        }
        return self.tokenize_obs(obs)

    def add_confusion_relation_objects(self, confusion_relation):
        objects_to_add = set()
        for relation in confusion_relation:
            objects_to_add.update(relation)

        for obj in objects_to_add:
            if obj in self.player.inventory:
                self.player.inventory[obj] += 1
            else:
                self.player.inventory[obj] = 1

        # 更新环境状态，确保obs和text_obs一致
        self.update_environment_with_inventory()

    def update_environment_with_inventory(self):
        # 更新视觉渲染（具体实现根据需求调整）
        self._render_inventory()

        # 更新文本描述
        # 直接在这里处理文本更新逻辑
        self._text_obs = self._generate_text_obs()

    def _render_inventory(self):
        # 这里实现更新视觉渲染逻辑
        pass

    def _generate_text_obs(self):
        # 生成文本观察
        inv, status = self._inventory_to_text()
        obs = self._text_view.local_sentence_view(self.player)
        return obs, {'inv': inv, 'status': status}





