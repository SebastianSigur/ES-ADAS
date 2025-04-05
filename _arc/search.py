import argparse
import copy
import json
import os
import pickle
from collections import namedtuple,OrderedDict
from concurrent.futures import ThreadPoolExecutor
import re
import scipy as sp
import backoff
import numpy as np
import openai
from tqdm import tqdm
from google import genai
from google.genai import types
import random

from arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt, get_task_mutated_instruction, get_prompt_mutated, TASK_MUTATOR_PROMPTS as INITIAL_TASK_MUTATOR_PROMPTS, get_initial_task_mutators

# For the filename change the 'expr_name' arg at the bottom

### Experiment 1 ###
# Settings to change below:
# GENERATE_TASK_MUTATORS = False (Create new mutators OFF)
# TASK_MUTATORS_PERFORMANCE_SAMPLING = False (Uniform Sampling)
# Filename for each run:
# Filename run 1: arc_openai_gemini_task_mutator_experiment_1_run_1_results
# Filename run 2: arc_openai_gemini_task_mutator_experiment_1_run_2_results
# Filename run 3: arc_openai_gemini_task_mutator_experiment_1_run_3_results

### Experiment 2 ###
# Settings to change below:
# GENERATE_TASK_MUTATORS = True (Create new mutators ON)
# TASK_MUTATORS_PERFORMANCE_SAMPLING = False (Uniform Sampling)
# Filename for each run:
# Filename run 1: arc_openai_gemini_task_mutator_experiment_2_run_1_results
# Filename run 2: arc_openai_gemini_task_mutator_experiment_2_run_2_results
# Filename run 3: arc_openai_gemini_task_mutator_experiment_2_run_3_results

### Experiment 3 ###
# Settings to change below:
# GENERATE_TASK_MUTATORS = True (Create new mutators ON)
# TASK_MUTATORS_PERFORMANCE_SAMPLING = True (Performance Sampling)
# Filename for each run:
# Filename run 1: arc_openai_gemini_task_mutator_experiment_3_run_1_results
# Filename run 2: arc_openai_gemini_task_mutator_experiment_3_run_2_results
# Filename run 3: arc_openai_gemini_task_mutator_experiment_3_run_3_results

# Generator of Task Mutators Hyperparams
GENERATE_TASK_MUTATORS = True
N_TASK_MUTATORS = 10

# Task Performance Performance Sampling Hyperparams
TASK_MUTATORS_PERFORMANCE_SAMPLING = True
SAMPLING_TEMP = 0.3
PERFORMANCE_METRIC = 'mean'

# MIN ARCHIVE PERFORMANCE
MIN_PERFORMANCE_FLAG = True
MIN_PERCENTILE_ARCHIVE = 0.25


# Set the global seeds per run
random.seed(42)
np.random.seed(42)

openai_client = openai.OpenAI()
gemini_client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'))

from utils import random_id, format_arc_data, eval_solution, list_to_string, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        system_message,
        temperature=0.5
):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": msg},
    ]
    
    combined_prompt = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            combined_prompt += f"System: {content}\n\n"
        elif role == "user":
            combined_prompt += f"User: {content}\n\n"
        elif role == "assistant":
            combined_prompt += f"Assistant: {content}\n\n"

    response = gemini_client.models.generate_content(
        model='gemini-1.5-flash-8b',
        contents=combined_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1024,
            stop_sequences=None,
            response_mime_type='application/json'
            
        )
    )
    content = response.text
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        temperature=0.8
):
    print('Calling GPT-4o')
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        code_output = False

        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `transform` function. DO NOT return anything else. (It will be tested later.)"
                code_output = True
        system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue

            if isinstance(content, list):
                try:
                    content = list_to_string(content)
                except:
                    pass

            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except Exception as e:
                return gen_output(f"Error during function execution: {e}"), correct_examples, wrong_examples

            if transformed_grid == output_grid:
                feedback += f"Your transform function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    transformed_grid = list_to_string(transformed_grid)
                except:
                    pass
                feedback += f"Your transform function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {transformed_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        return gen_output(feedback), correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
        except Exception as e:
            return gen_output(f"Error during function execution: {e}")

        return gen_output(transform_output)


def search(args):
    # Generates Task Mutators if flag is ON
    if GENERATE_TASK_MUTATORS:
        mutators_filename = os.path.join(args.save_dir, f"{args.expr_name}_task_mutators.json")
        # Reloading mutators of run if they already exist
        if os.path.exists(mutators_filename):
            with open(mutators_filename, 'r') as mut_json_file:
                TASK_MUTATOR_PROMPTS_JSON = json.load(mut_json_file)
                TASK_MUTATOR_PROMPTS = TASK_MUTATOR_PROMPTS_JSON['task_mutators']
        else:
            # Generating Mutators if they have not been created
            sys_prompt_task_generator, prompt_task_generator=  get_initial_task_mutators(N_TASK_MUTATORS, INITIAL_TASK_MUTATOR_PROMPTS)

            msg_list_task_generator = [
                        {"role": "system", "content": sys_prompt_task_generator},
                        {"role": "user", "content": prompt_task_generator},
                    ]
            task_mutators_generated = get_json_response_from_gpt_reflect(msg_list_task_generator)
            # Each candidate returns the task mutators with different keys
            if 'instruction' in task_mutators_generated:
                assert type(task_mutators_generated['instruction'])==list, 'Invalid output from Task Mutator Generator'
                assert len(task_mutators_generated['instruction'])>=N_TASK_MUTATORS, 'Invalid length from Task Mutator Generator'
                TASK_MUTATOR_PROMPTS = task_mutators_generated['instruction'][:N_TASK_MUTATORS]
            else:
                assert type(task_mutators_generated['instruction_mutators'])==list, 'Invalid output from Task Mutator Generator'
                assert len(task_mutators_generated['instruction_mutators'])>=N_TASK_MUTATORS, 'Invalid length from Task Mutator Generator'
                TASK_MUTATOR_PROMPTS = task_mutators_generated['instruction_mutators'][:N_TASK_MUTATORS]

            print('-'*30)
            print('Task Mutators Generated:')
            for i,mut_task_prompt in enumerate(TASK_MUTATOR_PROMPTS): print(i+1,mut_task_prompt)
            print('-'*30)

            print('Saving Generated Mutators...')
            TASK_MUTATOR_PROMPTS_JSON = {'task_mutators' : TASK_MUTATOR_PROMPTS}
            with open(mutators_filename, 'w') as file:
                json.dump(TASK_MUTATOR_PROMPTS_JSON, file, indent=4)
    
    else:
        TASK_MUTATOR_PROMPTS = INITIAL_TASK_MUTATOR_PROMPTS[:]


    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    # Creating OrderedDict to keep track of Task Mutators
    if TASK_MUTATORS_PERFORMANCE_SAMPLING:
        # Getting the Avg. Fitness of Initial archive
        median_initial_accs = []
        for sol in archive:
            if 'initial' == sol['generation']:
                initial_acc = get_median_fitness(sol['fitness']) / 100
                median_initial_accs.append(initial_acc)
        if PERFORMANCE_METRIC=='median':
            initial_avg_acc = np.median(np.array(median_initial_accs)).item()
        else:
            initial_avg_acc = np.mean(np.array(median_initial_accs)).item()
        print('Initial Avg/Median Performance:',initial_avg_acc)

        # Initializing Mutators avg accuracy w the avg of the initial archive
        task_mutators_dict = OrderedDict()
        for i in range(len(TASK_MUTATOR_PROMPTS)):
            task_mutators_dict[i] = {
                'task_mutator' : TASK_MUTATOR_PROMPTS[i],
                'n' : 0,
                'results' : [],
                'generation': [],
                'avg_accuracy' : initial_avg_acc
            }
        
        # If there is already runs, the results, and avg. accuracies need to be updated
        # Iterating through the archive, and if a generation exists, then we add its fitness
        for sol in archive:
            if 'initial' != sol['generation']:
                acc = get_median_fitness(sol['fitness']) / 100
                mut_id = TASK_MUTATOR_PROMPTS.index(sol['task_mutator'])
                task_mutators_dict[mut_id]['results'].append(acc)
        # Re-iterating through the dict to update the avg. accuracies
        for i in range(len(TASK_MUTATOR_PROMPTS)):
            if len(task_mutators_dict[i]['results'])>0:
                if PERFORMANCE_METRIC=='median':
                    new_mutator_avg = np.median(np.array(task_mutators_dict[i]['results'])).item()
                else:
                    new_mutator_avg = np.mean(np.array(task_mutators_dict[i]['results'])).item()
                task_mutators_dict[i]['avg_accuracy'] = new_mutator_avg

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        archive_for_prompt = copy.deepcopy(archive)
        for sol in archive_for_prompt:
            # Removing the Code Mutator description so that it does not influence future runs
            if 'code_mutator' in sol:
                del sol['code_mutator']
            # Removing the Task Mutator description so that it does not influence future runs
            if 'task_mutator' in sol:
                del sol['task_mutator']
            # Removing the Task Mutated Instruction so that it does not influence future runs
            if 'mutated_instruction' in sol:
                del sol['mutated_instruction']

        if MIN_PERFORMANCE_FLAG:
            # Getting results of proposed agents (valid results, so no 0%)
            n_valid_agents = 0
            agents_results = []
            for sol in archive_for_prompt:
                n_valid_agents += 1 #???
                agent_performance = get_median_fitness(sol['fitness']) # MEDIAN FUNC !!!
                # should I include 0 agents?, I'd say best if I don't...
                if agent_performance>0:
                    agents_results.append(agent_performance)
            if n_valid_agents>=2:
                agents_results = np.array(agents_results)
                min_perf = np.percentile(agents_results, MIN_PERCENTILE_ARCHIVE*100).item()/100
            else: 
                min_perf = 0.0
            print('Min Performance for Archive:', round(min_perf,4)) 

        # Removing agents that have 0% accuracy (only from the archive given as context, not from the saved archive)
        archive_for_prompt_tmp = []        
        for sol in archive_for_prompt:
            if 'fitness' in sol:
                if get_upper_bound(sol['fitness']) == 0.:
                    continue
                if MIN_PERFORMANCE_FLAG:
                    if (get_median_fitness(sol['fitness'])/100)<=min_perf:
                        continue
            archive_for_prompt_tmp.append(sol)
        archive_for_prompt = archive_for_prompt_tmp
        # Removing mutator info from the archive for prompt
        for sol in archive_for_prompt:
            if 'mutated_instruction' in sol:
                del sol['mutated_instruction']
            if 'task_mutator' in sol:
                del sol['task_mutator']

        # If Performance Sampling is on for Task mutators it calculates de probs using Softmax
        if TASK_MUTATORS_PERFORMANCE_SAMPLING:
            # Get Performances and use Softmax to get p
            mutators_accuracies = [task_mutators_dict[k]['avg_accuracy'] for k in task_mutators_dict.keys()]
            p = sp.special.softmax(np.array(mutators_accuracies) / SAMPLING_TEMP)
        else:
            p = np.ones(len(TASK_MUTATOR_PROMPTS)) / len(TASK_MUTATOR_PROMPTS)
        # Task mutator only applied to new agents, not to initial ones
        system_prompt_task, task_mutator_prompt, task_mutator_id, task_mutator = get_task_mutated_instruction(TASK_MUTATOR_PROMPTS, p)
        msg_list_task_mut_query = [
            {"role": "system", "content": system_prompt_task},
            {"role": "user", "content": task_mutator_prompt},
        ]
        task_mutator_instruction = get_json_response_from_gpt_reflect(msg_list_task_mut_query)
        task_mutator_instruction = task_mutator_instruction['instruction']

        system_prompt, prompt = get_prompt_mutated(archive_for_prompt, task_mutator_instruction)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive_for_prompt[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        # Adding the Task Mutator Instruction:
        next_solution['task_mutator'] = task_mutator
        next_solution['mutated_instruction'] = task_mutator_instruction

        # Updating Task Mutator Dict (if available)
        if TASK_MUTATORS_PERFORMANCE_SAMPLING:
            task_mutators_dict[task_mutator_id]['n'] += 1
            task_mutators_dict[task_mutator_id]['generation'].append(n+1)
            task_mutators_dict[task_mutator_id]['results'].append(get_median_fitness(next_solution['fitness'])/100)
            if PERFORMANCE_METRIC=='median':
                new_mutator_avg = np.median(np.array(task_mutators_dict[task_mutator_id]['results'])).item()
            else:
                new_mutator_avg = np.mean(np.array(task_mutators_dict[task_mutator_id]['results'])).item()
            task_mutators_dict[task_mutator_id]['avg_accuracy'] = new_mutator_avg

        # Reporting Sampling Dist. at the end of each generation
        if TASK_MUTATORS_PERFORMANCE_SAMPLING:
            # Get Performances and use Softmax to get p
            mutators_accuracies = [task_mutators_dict[k]['avg_accuracy'] for k in task_mutators_dict.keys()]
            p = sp.special.softmax(np.array(mutators_accuracies) / SAMPLING_TEMP)
            print(p)

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

def get_upper_bound(upper_bound_string):
    match = re.search(r'\(([\d.]+)%,\s*([\d.]+)%\)', upper_bound_string)
    if match:
        return float(match.group(2))
    else:
        return 0.0

def get_median_fitness(fitness_string):
    match = re.search(r'Median:\s*([\d.]+)%$', fitness_string)
    if match:
        return float(match.group(1))
    else:
        return 0.0

def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    evaluation_candidates = [] # only choosing top agents to evaluate
    print(f'len(archive): {len(archive)}')
    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        
        sorted_archive = sorted(archive, key=lambda x: get_upper_bound(x['fitness']), reverse=True)
        
        count = 0
        max_agents = args.max_agents
        initial_count = 0
        for archived_agent in archive:
            if archived_agent['generation'] == "initial":
                evaluation_candidates.append(archived_agent)
                initial_count += 1
        for archived_agent in sorted_archive:
            if archived_agent['generation'] == "initial":
                continue
            if count >= max_agents:
                break
            evaluation_candidates.append(archived_agent)
            count += 1
            
        if len(eval_archive) - initial_count >= args.max_agents:
             break
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = evaluation_candidates[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1


def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    if SEARCHING_MODE:
        arc_dir = args.val_data_path
    else:
        arc_dir = args.test_data_path
    print(arc_dir)
    with open(arc_dir, 'rb') as pickle_file:
        arc_data_queue = pickle.load(pickle_file)

    print(f"problem length: {len(arc_data_queue) * args.n_repreat}")
    max_workers = min(len(arc_data_queue) * args.n_repreat, args.max_workers) if args.multiprocessing else 1

    agent_task_queue = []
    for arc_data in arc_data_queue:
        task_str, examples, test_input = format_arc_data(arc_data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data)] * args.n_repreat)

    def call_forward(agent_task_queue):
        agent, taskInfo, arc_data = agent_task_queue
        res = agent.forward(taskInfo)
        origin_res = res
        try:
            if isinstance(res, Info):
                res = res.content
            if isinstance(res, str):
                res = eval(res)
            hard_score = eval_solution(res, arc_data, soft_eval=False)
            return hard_score
        except Exception as e:
            # print(e)
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acc_list = list(tqdm(executor.map(call_forward, agent_task_queue), total=len(agent_task_queue)))

    print("acc:", bootstrap_confidence_interval(acc_list))
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, default='dataset/sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='dataset/sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default='arc_openai_gemini_task_mutator_experiment_3_run_3_results')
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--max_agents', type=int, default=5)

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
