""" Env wrapper which adds goals and rewards """

import pickle as pkl
import time
import copy
import os
import pathlib
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util as st_utils
import language_model as lm
import fasteners
# from captioner import get_captioner
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class CrafterGoalWrapper:
    """ Goal wrapper for baselines. Used for baselines and single-goal eval. """

    def __init__(self, env, env_reward, single_task=None, single_goal_hierarchical=False):
        self.env = env
        self._single_task = self.set_single_task(single_task)
        self._single_goal_hierarchical = single_goal_hierarchical
        self._use_env_reward = env_reward
        self.prev_goal = ""
        self.use_sbert_sim = False
        self.goals_so_far = dict.fromkeys(self.env.action_names)  # for Eval purposes only
        self._cur_subtask = 0
        self.custom_goals = ['plant row', 'make workshop', 'chop grass with wood pickaxe', 'survival', 'vegetarianism',
                             'deforestation',
                             'work and sleep', 'gardening', 'wilderness survival']

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)

    def set_env_reward(self, use_env_reward):
        """ If this is true, we use the env reward, not the text reward."""
        self._use_env_reward = use_env_reward

    def on_final_subtask(self):
        return self._cur_subtask == len(self.goal_compositions()[self._single_task]) - 1

    def get_subtask(self):
        if not self._single_task:
            return None
        if self._single_goal_hierarchical:
            try:
                return self.goal_compositions()[self._single_task][self._cur_subtask]
            except:
                print(f'Error finding subtask {self._cur_subtask} for task {self._single_task}')
                import pdb;
                pdb.set_trace()
        else:
            return self._single_task

    def goal_compositions(self):
        """ Returns a dictionary with each goal, and the prereqs needed to achieve it. """
        goal_comps = {
            'eat plant': ['chop grass', 'place plant', 'eat plant'],
            'attack zombie': ['attack zombie'],
            'attack skeleton': ['attack skeleton'],
            'attack cow': ['attack cow'],
            'chop tree': ['chop tree'],
            'mine stone': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone'],
            'mine coal': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine coal'],
            'mine iron': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                          'make stone pickaxe', 'mine iron'],
            'mine diamond': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                             'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron',
                             'make iron pickaxe', 'mine diamond'],
            'drink water': ['drink water'],
            'chop grass': ['chop grass'],
            'sleep': ['sleep'],
            'place stone': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                            'place stone'],
            'place crafting table': ['chop tree', 'place crafting table'],
            'place furnace': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                              'mine stone', 'mine stone', 'mine stone', 'place furnace'],
            'place plant': ['chop grass', 'place plant'],
            'make wood pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe'],
            'make stone pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                                   'make stone pickaxe'],
            'make iron pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                                  'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron',
                                  'make iron pickaxe'],
            'make wood sword': ['chop tree', 'chop tree', 'place crafting table', 'make wood sword'],
            'make stone sword': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                                 'make stone sword'],
            'make iron sword': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone',
                                'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron',
                                'make iron sword'],
            'plant row': ['chop grass', 'place plant', 'chop grass', 'plant grass'],
            'chop grass with wood pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe',
                                             'chop grass with wood pickaxe'],
            'vegetarianism': ['drink water', 'chop grass'],
            'make workshop': ['chop tree', 'place crafting table', 'chop tree', 'place crafting table'],
            'survival': ['survival'],
            'deforestation': ['chop tree', 'chop tree', 'chop tree', 'chop tree', 'chop tree'],
            'work and sleep': ['chop tree', 'sleep', 'place crafting table'],
            'gardening': ['chop grass', 'chop tree', 'place plant'],
            'wilderness survival': ['sleep', 'chop grass', 'attack zombie'],

        }
        if self.env.action_space_type == 'harder':
            return self.filter_hard_goals(goal_comps)
        else:
            return goal_comps

    def check_multistep(self, action):
        """Check if a given action has prereqs"""
        if isinstance(action, str):
            action_name = action
        else:
            action_name = self.action_names[action]
        return action_name not in ['attack zombie', 'attack skeleton', 'attack cow', 'chop tree', 'drink water',
                                   'chop grass', 'sleep']

    def _tokenize_goals(self, new_obs):
        if self.use_sbert:  # Use SBERT tokenizer
            new_obs['goal'] = self.env.pad_sbert(new_obs['goal'])
        return new_obs

    def reset(self):
        """Reset the environment, adding in goals."""
        # Parametrize goal as enum.
        obs, info = self.env.reset()
        self.goal_str = ""
        self.oracle_goal_str = ""
        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        obs['goal_success'] = np.array(0)  # 0 b/c we can't succeed on the first step
        obs = self.tokenize_obs(obs)
        self._cur_subtask = 0
        # self._reset_custom_task()
        return self._tokenize_goals(obs), info

    def _reset_custom_task(self):
        self.drank_water = False
        self.trees_chopped = 0

    def set_single_task(self, task):
        """ When single_task is set, we only give the agent this goal."""
        self._single_task = task
        return task

    def set_end_on_success(self, end_on_success):
        """ When end_on_success is set, we end the episode when the agent succeeds."""
        self._end_on_success = end_on_success
        return end_on_success

    def set_single_goal_hierarchical(self, single_goal_hierarchical):
        """
        There are 3 options for single_goal_hierarchical:
        - False/None: don't use hierarchical goals
        - 'reward_last': use hierarchical goals, but don't reward the agent for each subtask
        - True: use hierarchical goals, and reward the agent for each subtask
        """
        self._single_goal_hierarchical = single_goal_hierarchical
        return single_goal_hierarchical

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Replace env reward with text reward
        goal_reward = 0
        # if not self._use_env_reward:
        #     if info['action_success']:  # Only reward on an action success
        #         action_name = self.env.get_action_name(action)
        #         if self._single_task is not None:  # If there's a single task, check if we've achieved it
        #             # If reward_last is true, don't reward for intermediate tasks
        #             task = self.get_subtask()
        #             achieved_task = action_name == task
        #             goal_reward = int(achieved_task)
        #             if self._single_goal_hierarchical == 'reward_last' and (not self.on_final_subtask()):
        #                 goal_reward = 0
        #             if achieved_task and self._single_goal_hierarchical:
        #                 self._cur_subtask = min(self._cur_subtask + 1,
        #                                         len(self.goal_compositions()[self._single_task]) - 1)
        #         else:
        #             goal_reward = 1  # reward for any env success if no task is specified
        #             reward = info['health_reward'] + goal_reward
        #     else:
        #         reward = 0  # Don't compute reward if action failed.

        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        obs['goal_success'] = info['eval_success'] and goal_reward > 0
        obs = self.tokenize_obs(obs)
        return self._tokenize_goals(obs), reward, done, info

    def filter_hard_goals(self, inputs):
        good_actions = self.env.good_action_names + self.custom_goals
        if isinstance(inputs, dict):
            return {k: v for k, v in inputs.items() if k in good_actions}
        elif isinstance(inputs, list):
            return [v for v in inputs if v in good_actions]
        else:
            raise NotImplementedError


class CrafterLMGoalWrapper(CrafterGoalWrapper):

    def __init__(self, env, lm_spec, env_reward, device=None, threshold=.5, debug=True, single_task=None,
                 single_goal_hierarchical=False,
                 use_state_captioner=False, use_transition_captioner=False, check_ac_success=True):
        super().__init__(env, env_reward, single_task, single_goal_hierarchical)
        self.env = env
        self.debug = debug
        self.goal_str = ""  # Text describing the goal, e.g. "chop tree"
        self.oracle_goal_str = ""
        self.prev_goal = ""
        self.goals_so_far = {}
        self.sbert_time = 0
        self.cache_time = 0
        self.cache_load_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_successes = 0
        self.step_num = 1
        self.query_interval = 100
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'embedding_cache.pkl'
        self.rw_lock = fasteners.InterProcessReaderWriterLock(self.cache_path)

        # Language model setup.
        prompt_format = getattr(lm, lm_spec['prompt'])()
        lm_class = getattr(lm, lm_spec['lm_class'])
        self.check_ac_success = check_ac_success
        if 'Baseline' in lm_spec['lm_class']:
            lm_spec['all_goals'] = self.action_names.copy()
            self.check_ac_success = False
        self.lm = lm_class(prompt_format=prompt_format, **lm_spec)
        # self.oracle_lm = lm.SimpleOracle(prompt_format=prompt_format, **lm_spec)
        self.use_sbert_sim = True
        self.device = device
        assert self.device is not None
        self.threshold = threshold
        # hugging face 下载被墙
        self.embed_lm = SentenceTransformer('/home/amax/zhp/code/paraphrase-MiniLM-L3-v2')
        self.device = torch.device(device)
        # 存储str对应的embedding
        self.cache = {}
        self.suggested_goals = []
        self.all_suggested_actions = []
        self.oracle_suggested_actions = []
        self.past_actions = None
        self.past_goals_str = None
        self.past_goals_list = None
        self.all_frac_valid, self.all_frac_covered, self.all_frac_correct = [], [], []
        self.unit_cache_time, self.unit_query_time = [], []
        self._end_on_success = False
        # Get the captioner model
        self._use_state_captioner = use_state_captioner
        self._use_transition_captioner = use_transition_captioner
        # if use_state_captioner or use_transition_captioner:
        #     self.transition_captioner, self.state_captioner, self.captioner_logging = get_captioner()
        self.transition_caption = self.state_caption = None
        self.prev_info = None

    def _save_caches(self):
        if self.debug: pass
        start_time = time.time()
        # The cache will be used by multiple processes, so we need to lock it.
        # We will use the file lock to ensure that only one process can write to the cache at a time.

        self.rw_lock.acquire_write_lock()
        with open(self.cache_path, 'wb') as f:
            pkl.dump(self.cache, f)
        self.rw_lock.release_write_lock()
        self.cache_load_time += time.time() - start_time

    def load_and_save_caches(self):
        new_cache = self._load_cache()
        # Combine existing and new cache
        self.cache = {**new_cache, **self.cache}
        self._save_caches()

    def _load_cache(self):
        if self.debug:
            self.cache = {}
            return {}
        start_time = time.time()
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                self.rw_lock.acquire_read_lock()
                with open(self.cache_path, 'rb') as f:
                    cache = pkl.load(f)
                self.rw_lock.release_read_lock()
            except FileNotFoundError:
                cache = {}
        self.cache_load_time += time.time() - start_time
        return cache

    def text_reward(self, action_embedding, rewarding_actions, update_suggestions=True):
        """
            Return a sparse reward based on how close the task is to the list of actions
            the  LM proposed.
        """
        text_rew = 0
        best_suggestion = None

        # If there are no suggestions, there is no reward
        if len(rewarding_actions) == 0:
            return 0, None

        if self.device is None:
            raise ValueError("Must specify device for real LM")

        # Cosine similarity reward
        suggestion_embeddings, suggestion_strs, updated_cache = self._get_model_embeddings(rewarding_actions)

        # action_name = self.get_action_name(action_embedding)
        action_name = action_embedding
        action_embedding, updated_cache_action = self._get_model_embedding(action_name)

        # Compute the cosine similarity between the action embedding and the suggestion embeddings
        cos_scores = st_utils.pytorch_cos_sim(action_embedding, suggestion_embeddings)[0].detach().cpu().numpy()

        # Compute reward for every suggestion over the threshold
        for suggestion, cos_score in zip(suggestion_strs, cos_scores):
            # print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
            if cos_score > self.threshold:
                print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
                if suggestion in self.all_suggested_actions and update_suggestions:
                    self.all_suggested_actions.remove(suggestion)
                text_rew = max(cos_score, text_rew)
        if text_rew > 0:
            best_suggestion = suggestion_strs[np.argmax(cos_scores)]
            print(text_rew, best_suggestion)
        return text_rew, best_suggestion

    def _get_model_embeddings(self, str_list):
        assert isinstance(str_list, list)
        # Split strings into those in cache and those not in cache
        strs_in_cache = []
        strs_not_in_cache = []
        for str in str_list:
            if str in self.cache:
                strs_in_cache.append(str)
            else:
                strs_not_in_cache.append(str)
        all_suggestions = strs_in_cache + strs_not_in_cache

        # Record how many strings are in/not in the cache
        self.cache_hits += len(strs_in_cache)
        self.cache_misses += len(strs_not_in_cache)

        # Encode the strings which are not in cache
        if len(strs_not_in_cache) > 0:
            start_time = time.time()
            embeddings_not_in_cache = self.embed_lm.encode(strs_not_in_cache, convert_to_tensor=True,
                                                           device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            assert embeddings_not_in_cache.shape == (len(strs_not_in_cache), 384)  # size of sbert embeddings
            # Add each (action, embedding) pair to the cache
            for suggestion, embedding in zip(strs_not_in_cache, embeddings_not_in_cache):
                self.cache[suggestion] = embedding
            updated_cache = True
        else:
            embeddings_not_in_cache = torch.FloatTensor([]).to(self.device)
            updated_cache = False

        # Look up the embeddings of the strings which are in cache
        if len(strs_in_cache) > 0:
            start_time = time.time()
            embeddings_in_cache = torch.stack([self.cache[suggestion] for suggestion in strs_in_cache]).to(self.device)
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            assert embeddings_in_cache.shape == (len(strs_in_cache), 384)  # size of sbert embeddings
        else:
            embeddings_in_cache = torch.FloatTensor([]).to(self.device)

        # Concatenate the embeddings of the suggestions in the cache and the suggestions not in the cache
        suggestion_embeddings = torch.cat((embeddings_in_cache, embeddings_not_in_cache), dim=0)
        return suggestion_embeddings, all_suggestions, updated_cache

    def _get_model_embedding(self, action_name):
        " return the embedding for the action name, and a boolean indicating if the cache was updated"
        if action_name in self.cache:
            start_time = time.time()
            embedding = self.cache[action_name]
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            return embedding, False
        else:
            start_time = time.time()
            embedding = self.embed_lm.encode(action_name, convert_to_tensor=True, device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            self.cache[action_name] = embedding
            return embedding, True


    def reset(self, evcg_matrix, evcg_objects):
        obs, info = self.env.reset()
        self._cur_subtask = 0
        self.goal_str = None
        self.oracle_goal_str = None
        self.lm.reset()
        # self.oracle_lm.reset()
        # 根据当前text obs更新因果图
        text_obs, inv_status = self.env.text_obs()
        str_graph = self.lm.query({'obs': text_obs, **inv_status})
        # 解析文本，构建因果图
        if evcg_matrix is None or evcg_objects is None:
            first_causal_dict = self.parse_causal_description(str_graph)
            evcg_matrix, evcg_objects = self.build_causal_matrix(first_causal_dict)
        else:
            evcg_matrix, evcg_objects = self.update_causal_matrix(evcg_matrix, evcg_objects, str_graph)
        info['evcg_matrix'] = evcg_matrix
        info['evcg_objects'] = evcg_objects
        self.prev_info = info
        goals = self._make_predictions(str_graph)
        goals_str = " ".join([s.lower().strip() for s in goals])
        if self.use_sbert:
            goals_str = 'Your goal is: ' + goals_str + '.'
        obs['goal'] = goals_str
        obs['success'] = False
        obs['goal_success'] = np.array(0)
        obs = self.env.tokenize_obs(obs)
        # self._reset_custom_task()
        return self._tokenize_goals(obs), info

    def parse_causal_description(self, description):
        # 使用正则表达式匹配文本描述中的因果关系
        pattern = re.compile(r'- object (\w+) -> object (\w+)')
        # 找到所有匹配的因果对，并构建从效果到原因的映射
        causal_relationships = pattern.findall(description)
        # 创建一个字典来存储每个对象的因果关系
        causal_dict = {}
        for match in causal_relationships:
            cause, effect = match
            # 将效果对象添加到字典中，如果没有则初始化为一个集合
            if effect not in causal_dict:
                causal_dict[effect] = set()
            if cause not in causal_dict:
                causal_dict[cause] = set()
            # 添加直接原因
            causal_dict[effect].add(cause)

        return causal_dict

    def build_causal_matrix(self, causal_dict):
        # 确定所有对象的集合
        all_objects = set(causal_dict.keys())
        # 为每个对象创建索引
        object_to_index = {obj: idx for idx, obj in enumerate(all_objects)}

        # 初始化因果矩阵，所有条目都为0
        num_objects = len(object_to_index)
        causal_matrix = [[0] * num_objects for _ in range(num_objects)]

        # 遍历因果关系字典填充矩阵
        for effect, causes in causal_dict.items():
            effect_index = object_to_index[effect]
            for cause in causes:
                cause_index = object_to_index[cause]
                causal_matrix[cause_index][effect_index] = 1

        return causal_matrix, object_to_index

    def update_causal_matrix(self, causal_matrix, object_to_index, new_description):
        # 解析新的描述以获取新的因果关系
        new_causal_dict = self.parse_causal_description(new_description)
        # 找出新描述中所有涉及的对象，并添加到现有对象集合中
        all_objects = set(object_to_index.keys()) | set(new_causal_dict.keys())
        all_objects |= set(cause for causes in new_causal_dict.values() for cause in causes)
        # 更新对象到索引的映射
        updated_object_to_index = object_to_index.copy()  # 复制现有映射以保留现有索引
        if len(all_objects - set(object_to_index.keys())) == 0:
            return causal_matrix, object_to_index  # 如果没有新对象，直接返回原因果矩阵
        for obj in all_objects - set(object_to_index.keys()):  # 仅对新对象进行索引
            updated_object_to_index[obj] = len(updated_object_to_index)
        # 更新因果矩阵的大小并填充新对象的因果关系
        num_nodes = len(updated_object_to_index)
        updated_causal_matrix = np.array([[0] * num_nodes for _ in range(num_nodes)])

        # 复制现有因果关系到更新的矩阵中
        updated_causal_matrix[:len(causal_matrix), :len(causal_matrix)] = causal_matrix

        # 添加新的因果关系到矩阵中
        for effect, causes in new_causal_dict.items():
            effect_index = updated_object_to_index[effect]
            for cause in causes:
                cause_index = updated_object_to_index[cause]
                updated_causal_matrix[cause_index][effect_index] = 1

        return updated_causal_matrix, updated_object_to_index

    def causal_matrix_to_text(self, causal_matrix, object_to_index):
        # 初始化一个空列表来保存文本描述的每一行
        descriptions = []

        # 获取对象名称列表，以便可以通过索引访问
        objects = list(object_to_index.keys())

        # 遍历因果矩阵的每一行
        for cause_index, row in enumerate(causal_matrix):
            # 找出所有值为1的列，这些列的索引表示结果对象
            effect_indices = [index for index, value in enumerate(row) if value == 1]

            # 对于每个结果对象，构建描述语句
            for effect_index in effect_indices:
                cause = objects[cause_index]  # 原因对象
                effect = objects[effect_index]  # 结果对象
                description = f"- object {cause} -> object {effect}."
                descriptions.append(description)
        # 将描述列表转换为文本描述，描述之间用换行符分隔
        return "\n".join(descriptions)

    def _make_predictions(self, causality_graph_text):
        text_obs, inv_status = self.env.text_obs()
        if self._use_state_captioner:
            state_caption = self.state_captioner(self.prev_info)
            self.state_caption = state_caption
            caption = state_caption
        else:
            caption = text_obs
        # TODO:根据因果图来做planning
        suggest_goals = self.lm.predict_options(
            {'obs': caption, "causality": causality_graph_text,
             "past_action": "noop" if self.past_actions is None else self.past_actions,
             "past_goals": "null" if self.past_goals_str is None else self.past_goals_str, **inv_status}, self)
        self.past_goals_str = " - " + "\n - ".join(suggest_goals)
        self.past_goals_list = suggest_goals
        return suggest_goals

    def step(self, action, evcg_matrix, evcg_objects):
        # 每隔一定步数，根据观察到的信息更新因果图 （√）
        obs, reward, done, info = self.env.step(action)   # 执行环境中的动作，获取新的观察结果、奖励、是否结束标志及额外信息
        if self.step_num % self.query_interval == 0:
            # 再次根据当前text obs更新因果图
            text_obs, inv_status = self.env.text_obs()
            str_graph = self.lm.query({'obs': text_obs, **inv_status})
            # 解析文本，构建因果图
            if evcg_matrix is None or evcg_objects is None:      # 判断是否首次构建因果图
                first_causal_dict = self.parse_causal_description(str_graph)    # 解析文本描述，得到因果关系字典
                evcg_matrix, evcg_objects = self.build_causal_matrix(first_causal_dict)     # 根据字典构建因果图的矩阵和对象表示
            else:
                evcg_matrix, evcg_objects = self.update_causal_matrix(evcg_matrix, evcg_objects, str_graph)  # 否则更新现有因果图的矩阵和对象表示
            info['evcg_matrix'] = evcg_matrix   # 将更新后的因果图矩阵存入信息字典
            info['evcg_objects'] = evcg_objects   # 将更新后的因果图对象存入信息字典
            causality_graph_text = self.causal_matrix_to_text(evcg_matrix, evcg_objects)   # 将因果图转为文本表示
            goals = self._make_predictions(causality_graph_text)   # 根据因果图文本生成子目标列表
            goals_str = " ".join([s.lower().strip() for s in goals])   # 将子目标列表连接成一个字符串，每个子目标首字母小写且去前后空格
        else:
            goals_str = " ".join([s.lower().strip() for s in self.past_goals_list])   # 非更新步数时，使用历史子目标列表生成字符串
        if self.use_sbert:
            goals_str = 'Your goal is: ' + goals_str + '.'
        obs['goal'] = goals_str
        self.step_num += 1

        # TODO: 计算transition和LLM给出sub-goals之间的相似度，将相似度作为rewar
        #  这里由于使用原动作空间，对于其do的动作，其可能对于不同的物体会有不同的意义，比如do+tree 就是sub-goal chop tree


        # TODO--------------------------------------------------------------
        #  暂时没有把多任务和特殊任务的情况放进来

        info['env_reward'] = reward
        health_reward = info['health_reward']
        text_reward = 0
        closest_suggestion = None
        info['goal_achieved'] = None

        # 动作空间处理，do分解 动词-名词
        if self.action_space_type == 'harder':
            verb, noun = self.unflatten_ac(action)  # text_crafter/text_env.py
            action = tuple([verb, noun])

        # 若动作可以成功执行
        if (info['action_success'] and self.check_ac_success) or not self.check_ac_success:
            action_name = self.get_action_name(action)
            if self._use_transition_captioner:
                caption = self.transition_captioner(self.prev_info, info)  # 当前状态描述
                self.transition_caption = caption
                caption_meaningful = caption.strip() != 'you did nothing.' and caption.strip() != 'nothing happened' # 描述是否有意义
                if caption_meaningful:  #若描述有意义，则使用该描述与LLM给出的sub_goal进行相似度匹配
                    # TODO: 源代码是caption与past_goals_list进行相似度匹配，不确定这里是不是要跟“goals”还是"goals_str"进行匹配
                    text_reward, closest_suggestion = self.text_reward(caption, self.past_goals_list, update_suggestions=True)
            else:  # 若不使用transition_captioner，则直接使用动作名与LLM给出的sub_goal进行相似度匹配
                text_reward, closest_suggestion = self.text_reward(action_name, self.past_goals_list, update_suggestions=True)

            if not self._use_env_reward:
                reward = health_reward + text_reward

            # 语言模型动作
            self.lm.take_action(closest_suggestion)
            self.oracle_lm.take_action(action_name)

            # 信息更新
            info['goal_achieved'] = closest_suggestion

        # 若动作执行失败
        else:
            if not self._use_env_reward:
                reward = health_reward  # Don't compute lm reward if action failed

        # 信息更新
        info['text_reward'] = text_reward
        self.prev_info = info
        self._make_predictions()

        # 观察更新
        obs = self._get_full_obs(obs)
        obs['success'] = info['action_success']
        obs['goal_success'] = int(info['eval_success'] and text_reward > 0)

        # 打印信息
        if text_reward > 0:
            print(f"Goal success {obs['goal_success']}, {info['action_success']}, {text_reward}")

        # TODO--------------------------------------------------------------


        # 信息更新
        obs = self.env.tokenize_obs(obs)
        obs = self._tokenize_goals(obs)
        return obs, reward, done, info


    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)
