import collections
import os
import pickle as pkl
import time
import wandb
import pathlib
import fcntl
import numpy as np
from http import HTTPStatus
import dashscope
import re
import torch
import str_utils
from openai import OpenAI


class PromptFormat:
    def format_prompt(self, state_dict):
        raise NotImplementedError

    def parse_response(self, response):
        raise NotImplementedError


class BulletPrompt(PromptFormat):
    def __init__(self):

        self.messages = [{'role': 'system',
                          'content': str_utils.QUERY_SUB_GOALS_SYS_PROMPT},
                         {'role': 'user',
                          'content': ''}]

    def format_prompt(self, state_dict, acheivement):
        past_goals = state_dict.get('past_goals', "")
        if past_goals != "<null>":
            past_goals = f'- {past_goals}'
        else:
            past_goals = '- <null>'

        # Capitalize first letter of each element of state_dict
        input = state_dict['obs'] + "\n" + state_dict['status'] + "\n" + state_dict['inv'] + "\n"
        input = input + "Past action: " + state_dict.get('past_action', "<null>") + "\n"
        input = input + "Past goals:\n" + past_goals + "\n"
        input = input + "Player's understanding of the causal relationship between objects in the game environment:\n"
        if state_dict['causality'] != 'null':
            for item in state_dict['causality']:
                cause, effect = item
                input += f"{cause} -> {effect}\n"
        else:
            input += 'Null'
        input = input + "\n" + "Achievements related to causality: \n"
        if acheivement is not None:
            for item in acheivement:
                input += f"{item}\n"
        else:
            input = input + 'None'
        self.messages[1]['content'] = input
        return self.messages

    def parse_response(self, response):
        """
        response: string, probably contains suggestions. Each suggestion starts with a dash.
        """
        # 定义合法动作的集合
        valid_actions = {"sleep", "eat", "attack", "chop", "drink", "place", "make", "mine"}
        if response[-4:] == '\n"""' or response[-4:] == '\n```':
            response = response[:-4]
        # 使用正则表达式匹配
        goals = re.findall(r'- goal \d+: (.+)', response)

        return goals


class LanguageModel:

    def __init__(self, **kwargs):
        super().__init__()
        self.achievements = set()
        self.verbose = kwargs.get('verbose', False)

    def reset(self):
        self.achievements = set()

    def take_action(self, suggestion):
        """
        action: action taken, in the form used in the constants file
        """
        if suggestion is not None:
            # Don't double count same suggestion
            if suggestion == 'place crafting table':
                self.achievements.add('make crafting table')
            elif suggestion == 'make crafting table':
                self.achievements.add('place crafting table')
            elif suggestion == 'eat cow':
                self.achievements.add('attack cow')
            elif suggestion == 'attack cow':
                self.achievements.add('eat cow')
            self.achievements.add(suggestion)

    def log(self, step):
        pass

    def predict_options(self, _, _2):
        raise NotImplementedError

    def load_and_save_cache(self):
        pass

    #
    def prereq_map(self, env='yolo'):
        prereqs = {  # values are [inv_items], [world_items]
            'eat plant': ([], ['plant']),
            'attack zombie': ([], ['zombie']),
            'attack skeleton': ([], ['skeleton']),
            'attack cow': ([], ['cow']),
            'eat cow': ([], ['cow']),
            'chop tree': ([], ['tree']),
            'mine stone': (['wood_pickaxe'], ['stone']),
            'mine coal': (['wood_pickaxe'], ['coal']),
            'mine iron': (['stone_pickaxe'], ['iron']),
            'mine diamond': (['iron_pickaxe'], ['diamond']),
            'drink water': ([], ['water']),
            'chop grass': ([], ['grass']),
            'sleep': ([], []),
            'place stone': (['stone'], []),
            'place crafting table': (['wood'], []),
            'make crafting table': (['wood'], []),
            'place furnace': (['stone', 'stone', 'stone', 'stone'], []),
            'place plant': (['sapling'], []),
            'make wood pickaxe': (['wood'], ['table']),
            'make stone pickaxe': (['stone', 'wood'], ['table']),
            'make iron pickaxe': (['wood', 'coal', 'iron'], ['table', 'furnace']),
            'make wood sword': (['wood'], ['table']),
            'make stone sword': (['wood', 'stone'], ['table']),
            'make iron sword': (['wood', 'coal', 'iron'], ['table', 'furnace']),
        }
        if env.action_space_type == 'harder':
            return env.filter_hard_goals(prereqs)
        else:
            return prereqs


class GPTLanguageModel(LanguageModel):
    """Use LLM Model"""

    def __init__(self, lm: str = 'text-curie-001',
                 prompt_format: PromptFormat = None,
                 max_tokens: int = 100,
                 temperature: float = .7,
                 stop_token=['\n\n'],
                 novelty_bonus=True,
                 use_local_llm=False,
                 **kwargs):
        """
        lm: which language model to use
        prompt format: ID of which prompt format to use
        max_tokens: maximum number of tokens returned by the lm
        temperature: temperature in [0, 1], higher is more random
        logger: optional logger
        """
        super().__init__(**kwargs)

        assert 0 <= temperature <= 1, f"invalid temperature {temperature}; must be in [0, 1]"
        self.total_token = 0
        self.lm = lm
        self.prompt_format = prompt_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.novelty_bonus = novelty_bonus
        self.cached_queries = 0
        self.all_queries = 0
        self.num_parse_errors = 0
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'lm_cache.pkl'
        self.cache = self.load_cache()
        self.prices = []
        self.attempts = 0
        self.api_key_idx = 0
        self.stop = stop_token
        model_id = kwargs.get("local_lm_path")
        self.use_qwen = kwargs.get("use_qwen", False)
        self.use_deepseek = kwargs.get("use_deepseek", False)

        print(f"model id:  {model_id}")
        self.use_local_llm = use_local_llm
        self.client = OpenAI(api_key="", base_url="")
        if use_local_llm:
            if self.use_qwen:
                print("using local qwen1.5 LLM")
                from modelscope import AutoModelForCausalLM, AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    "qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-7B-Chat-GPTQ-Int4")
            else:
                print("using local LLama3 LLM")
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            print("model loaded")
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            if self.use_qwen:
                print("using qwen api LLM")
                import dashscope
                dashscope.api_key = ""
            elif self.use_deepseek:
                print('using deppseek api')
                import requests
                self.client = OpenAI(api_key="", base_url="")
            elif self.use_kimi:
                print('using kimi api')
                import requests
                self.client = OpenAI(api_key="", base_url="",)




    def load_cache(self):
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                with open(self.cache_path, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    cache = pkl.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
            except:
                cache = {}
        return cache

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            # Lock file while saving cache so multiple processes don't overwrite it.
            fcntl.flock(f, fcntl.LOCK_EX)
            pkl.dump(self.cache, f)
            fcntl.flock(f, fcntl.LOCK_UN)

    def check_in_cache(self, inputs):
        return inputs in self.cache

    def retrieve_from_cache(self, inputs):
        return self.cache[inputs]

    def valid_causality(self, inputs):
        messages = [{'role': 'system', 'content': str_utils.JUDGEMENT_CAUSAL_RELATION_SYS_PROMPT},
                    {'role': 'user', 'content': inputs}]
        response = self.query_general(messages)
        return response

    def query(self, inputs):
        prefix = inputs['text_obs'] + "\n" + inputs['status'] + "\n" + inputs['inv'] + "\n"

        messages = [{'role': 'system', 'content': str_utils.QUERY_CAUSAL_RELATION_SYS_PROMPT},  # TODO prompt update
                    {'role': 'user', 'content': prefix}]

        response = self.query_general(messages)
        return response

    def query_for_confusion_relations(self, history_data, confusion_relation):
        inputs = "Player's historical information:\n"
        for index, item in enumerate(history_data):
            inputs += f"t = {index}:\n" + item + "\n"
        inputs += "#end\n" + "Player's Confusion causality:\n"
        cause, effect = confusion_relation
        inputs += f"- {cause} -> {effect}\n"
        messages = [{'role': 'system', 'content': str_utils.JUDGEMENT_CONFUSION_CAUSALITY_SYS_PROMPT},
                    {'role': 'user', 'content': inputs}]
        response = self.query_general(messages)
        if response == "Correct!":
            return True
        else:
            return False

    def query_goal(self, inputs, stage=None):

        if stage=='init':
            messages = [{'role': 'system', 'content': str_utils.QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT_INIT},
                        {'role': 'user', 'content': inputs}]
        else:
            messages = [{'role': 'system', 'content': str_utils.QUERY_SUB_GOALS_WITH_CONFUSION_SYS_PROMPT},
                        {'role': 'user', 'content': inputs}]
        response = self.query_general(messages)
        response = response.replace("-", "").strip()
        if len(response) == 0 or response == "":
            return None
        return response

    def query_general(self, inputs):

        input_content = inputs[1]['content']
        if self.check_in_cache(input_content):
            response = self.retrieve_from_cache(input_content)
            self.cached_queries += 1
            return response
        else:
            if self.use_local_llm:
                input_ids = self.tokenizer.apply_chat_template(
                    inputs,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.model.device)
                if self.use_qwen:
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=self.max_tokens,
                    )
                else:
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        eos_token_id=self.terminators,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.9,
                    )
                response = outputs[0][input_ids.shape[-1]:]
                response = self.tokenizer.decode(response, skip_special_tokens=True)
            else:
                if self.use_qwen:
                    response = dashscope.Generation.call(
                        'qwen-plus',
                        messages=inputs,
                        result_format="message",
                    )
                    if response.status_code == HTTPStatus.OK:
                        response = response.output.choices[0]['message']['content']
                    else:
                        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                            response.request_id, response.status_code,
                            response.code, response.message
                        ))
                        response = None
                elif self.use_deepseek:
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=inputs,
                        max_tokens=1024,
                        temperature=0.7,
                        stream=False
                    )
                    self.total_token += response.usage.total_tokens
                    print('query token cost:', self.total_token)
                    response = response.choices[0].message.content

                elif self.use_kimi:
                    response = self.client.chat.completions.create(
                        model="moonshot-v1-8k",
                        messages=inputs,
                        max_tokens=1024,
                        temperature=0.7,
                        stream=False
                    )
                    self.total_token += response.usage.total_tokens
                    print('query token cost:', self.total_token)
                    response = response.choices[0].message.content

            # 存入cache
            self.store_in_cache(input_content, response)
            return response

    def predict_options(self, state_dict, acheivement=None, env=None):
        """
        state_dict: a dictionary with language strings as values. {'inv' : inventory, 'status': health status, 'actions': actions, 'obs': obs}
        """
        prompt = self.prompt_format.format_prompt(state_dict, acheivement)
        inputs = prompt[1]['content']
        if self.check_in_cache(inputs):  # 如果遇到过一样的情况
            if self.verbose: print("Fetching from cache", prompt[-2000:-50])
            response = self.retrieve_from_cache(inputs)
            self.cached_queries += 1
            new_api_query = False
        else:
            if self.verbose: print("Fetching new inputs and response", prompt[-200:])
            response = None
            max_attempts = float('inf')  # 1000
            attempts = 0
            while response is None:
                try:
                    response = self.query_general(prompt)
                except Exception as e:
                    if attempts > max_attempts or not 'code' in self.lm:
                        if attempts > max_attempts:
                            print('max attempts exceeded')
                        raise e
                    attempts += 1
                    print('attempts:{}, prompt:{}'.format(attempts, self.prompt_format.messages))
                    time.sleep(4)
            self.attempts = .99 * self.attempts + .01 * attempts
            new_api_query = True
        self.all_queries += 1
        if new_api_query:
            self.store_in_cache(inputs, response)

        response = response.replace("-", "").strip()

        return response

    def store_in_cache(self, inputs, response):
        self.cache[inputs] = response

    def query_goal_for_complex_relation(self, verified_relation, state_dict, acheivement):
        past_goals = state_dict.get('past_goals', "")
        if past_goals != "<null>":
            past_goals = '-' + past_goals
        else:
            past_goals = '- <null>'

        # Capitalize first letter of each element of state_dict
        input = state_dict['obs'] + "\n" + state_dict['status'] + "\n" + state_dict['inv'] + "\n"
        input = input + "Past action: " + state_dict.get('past_action', "<null>") + "\n"
        input = input + "Past goals:\n" + past_goals + "\n"
        input = input + "Player's understanding of the causal relationship between objects in the game environment:\n"
        if verified_relation is None:
            input = input + "None"
        for relation in verified_relation:
            input += "- " + "-> ".join(relation) + "\n"
        input = input + "\n" + "Players have unlocked achievements: \n"
        if acheivement is not None:
            for key, value in acheivement.items():
                if value != 0:
                    input = input + f"{key}\n"
        else:
            input = input + 'None'
        input = input + "\n" + "Based on the information provided, providing an goal allows the player to unlock the achievement as much as possible."

        messages = [{'role': 'system', 'content': str_utils.QUERY_SUB_GOALS_SYS_PROMPT},
                    {'role': 'user', 'content': input}]

        response = self.query_general(messages)
        return response



