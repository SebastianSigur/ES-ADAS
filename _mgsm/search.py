import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import re
import backoff
import numpy as np
import openai
from tqdm import tqdm
from google import genai
from google.genai import types


from mgsm_prompt import get_init_archive, get_prompt, get_reflexion_prompt

openai_client = openai.OpenAI()
gemini_client = genai.Client(api_key=os.getenv('GOOGLE_AI_API_KEY'))

from utils import get_all_examples, random_id, bootstrap_confidence_interval, score_mgsm

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

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
    try:
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
    except Exception as e:
        print(e)
        raise e
    content = response.text
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        temperature=0.8
):
    print('Messaging openai')
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
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
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
    def __init__(self) -> None:
        pass


#-------------------------------------------------------------------------------------------------------#
def create_map_elites_structure_api(
    archive,
    candidate_labels=None,
    bins_api=2,
    min_api=None,
    max_api=None
):
    """
    Creates a map elites grid from the candidate archive based on two dimensions:
      - Structure label: taken from each solution's "structure_label" field.
      - API calls: the number of API calls (agent['api_calls']).

    Parameters:
        archive (list): List of candidate dictionaries.
        candidate_labels (list, optional): List of allowed structure labels. If None, they are derived from the archive.
        bins_api (int): Number of bins for the API calls dimension (now set to 2).
        min_api (int, optional): Minimum API calls value. If None, computed from the archive.
        max_api (int, optional): Maximum API calls value. If None, computed from the archive.

    Returns:
        dict: A dictionary mapping cell keys (as strings, e.g. "Chain-of-Thought,0")
              to the best candidate (elite) in that cell (the one with the highest fitness).
    """
    # Derive candidate labels from the archive if not provided.
    if candidate_labels is None:
        candidate_labels = list({sol.get("structure_label") for sol in archive if "structure_label" in sol})
    
    # Collect all API call counts.
    api_calls_values = [sol['api_calls'] for sol in archive if 'api_calls' in sol]
    if not api_calls_values:
        return {}
    if min_api is None:
        min_api = min(api_calls_values)
    if max_api is None:
        max_api = max(api_calls_values)
    
    # Initialize grid with keys as "structure_label,api_bin"
    grid = {f"{label},{i}": None for label in candidate_labels for i in range(bins_api)}
    
    for sol in archive:
        if 'structure_label' not in sol or 'api_calls' not in sol or 'fitness' not in sol:
            continue
        
        label = sol["structure_label"]
        # If the label is not in the candidate labels, skip this solution.
        if label not in candidate_labels:
            continue
        
        api_val = sol["api_calls"]
        norm_api = (api_val - min_api) / (max_api - min_api) if max_api != min_api else 0
        api_bin = min(int(norm_api * bins_api), bins_api - 1)
        cell_key = f"{label},{api_bin}"
        new_fitness = get_upper_bound(sol['fitness'])
        
        if grid[cell_key] is None:
            grid[cell_key] = sol
        else:
            current_sol = grid[cell_key]
            current_fitness = get_upper_bound(current_sol['fitness'])
            if new_fitness > current_fitness:
                grid[cell_key] = sol

    return grid
#-------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------#
def count_api_calls(forward_code):
    """
    Uses Gemini to analyze the given code and returns the number of API calls made.
    Each call to `LLMAgentBase(...)(...)` counts as one API call.
    The analysis accounts for loops, multiple agents, or recursive calls.
    Returns a JSON object with the 'api_calls' field as an integer.
    """
    system_prompt = ("You are a code analysis expert. Given a Python function, predict the number of API calls it makes to an LLM. "
                     "Each call to `LLMAgentBase(...)(...)` counts as one API call. Account for loops, multiple agents, or recursive calls. "
                     "Return a JSON object with the 'api_calls' field as an integer.")
    user_message = f"Analyze the following code and predict the number of API calls per execution:\n\n```python\n{forward_code}\n```"
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash-8b',
            contents=system_prompt + user_message,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                stop_sequences=None,
                response_mime_type='application/json'
            )
        )
        content = response.text
        api_json = json.loads(content)
        return int(float(api_json.get("api_calls", 0)))
    except Exception as e:
        print("Error assessing API calls:", e)
        return 0
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# Defines a function that takes the solution thought and name as details as input, and outputs a label for the structure.
# The label is created via using a BERT model and gemini
def recheck_label_with_gemini(agent_name, agent_thought, agent_code, candidate_labels):
    system_prompt = (
        "You are an expert in classification and label verification. You are given an agent’s name, its detailed \"thought\" description, "
        "and a set of candidate labels. The initial classification from a BERT model is highly uncertain (confidence ≤ 0.75). Your task is to "
        "reassess the agent’s approach and decide which candidate label best fits the agent’s design. Below are the candidate labels with their descriptions:\n\n"
        "1. Chain-of-Thought Reasoning: Generates a single, linear, step-by-step reasoning process with every intermediate step explicitly shown.\n"
        "2. Multi-Agent Reasoning: Runs several independent reasoning modules in parallel and aggregates their outputs to form the final answer.\n"
        "3. Self-Reflection Reasoning: Produces an initial answer, then internally critiques and refines it through iterative self-review.\n"
        "4. Abstraction to Principles Reasoning: First abstracts the problem’s details into high-level principles, then uses these abstractions to guide the solution.\n"
        "Do all the reasoning internally and output only the final label prediction (which must exactly match one of the provided candidate labels) with no additional explanation or text."
    )
    
    user_prompt = (
        f"Agent Name: {agent_name}\n"
        f"Agent Thought: {agent_thought}\n"
        f"Agent Code: {agent_code}\n"
        f"Candidate Labels: {', '.join(candidate_labels)}\n\n"
        "Please evaluate which single candidate label fits the agent best based on the agent thought and agent code. If the agent structure would fit to multiple labels, choose the one label it fits to the most. Output only the final label prediction and nothing else."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash-8b',
            contents=system_prompt + "\n\n" + user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                stop_sequences=None,
                response_mime_type='text/plain'
            )
        )
        final_label = response.text.strip()
        return final_label
    except Exception as e:
        print("Error during Gemini reclassification:", e)
        return None


def get_structure_label(solution):
    """
    Determines the structure label for a candidate solution based on its 'thought' field.
    Uses a zero-shot classification pipeline with a set of candidate labels.
    If the classifier confidence is low (≤ 0.75), it rechecks using Gemini via recheck_label_with_gemini().
    
    Returns:
        A string representing the structure label.
    """
    # Define the candidate labels for structure classification.
    candidate_labels = [
        "Chain-of-Thought Reasoning",
        "Multi-Agent Reasoning",
        "Self-Reflection Reasoning",
        "Abstraction to Principles Reasoning"
    ]
    
    # Initialize the zero-shot classification pipeline.
    # This uses the MoritzLaurer/mDeBERTa-v3-base-mnli-xnli model.
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    
    thought_text = solution.get("thought", "")
    if not thought_text:
        return None
    
    code_text = solution.get("code", "")
    if not code_text:
        return None
    
    # Get the classification result.
    output = classifier(thought_text, candidate_labels, multi_label=False)
    predicted_label = output['labels'][0]
    score = output['scores'][0]
    
    # If the confidence is low (≤ 0.75), recheck using Gemini.
    if score <= 0.75:
        # Use .get() to safely retrieve "name", providing a default if missing.
        agent_name = solution.get("name", "Unknown Agent")
        new_label = recheck_label_with_gemini(agent_name, thought_text, code_text, candidate_labels)
        return new_label if new_label is not None else predicted_label
    else:
        return predicted_label
#-------------------------------------------------------------------------------------------------------#



def search(args):

    ## Initializes and loads archive (uses save_dir & expr_name as locations to load and save archive )
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
    
    #-------------------------------------------------------------------------------------------------------#    
    ## Call Gemini to assess the code of each agent in the archive.
    ## Gemini's response should include only the number of API calls made. Add that information to each solution.
    for solution in archive:
        solution["api_calls"] = count_api_calls(solution["code"])
    
        ## Add a structure label to each solution
        solution["structure_label"] = get_structure_label(solution)

    # ## Ensure each candidate has an 'api_calls' field (for map elites dimension). Default to 0 if missing.
    # for solution in archive:
    #     if 'api_calls' not in solution:
    #         solution['api_calls'] = 0
    #------------------------------------------------------------------------------------------------------#

    ## Loops over every solution in the archive and ensures that they have a fitness score
    ## Note: Computes fitness using evaluate_forward_fn() and bootstrap_confidence_interval()
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

    # ------------------------------------------------------------
    # Compute current map elites using predefined function create_map(). Get fitness scores from above
    map_elites = create_map_elites_structure_api(archive,
                                                 candidate_labels=None,  # Derived from archive if None
                                                 bins_api=args.bins_dim2,
                                                 min_api=args.min_dim2, max_api=args.max_dim2)
    # ------------------------------------------------------------
    
    ## Generates n new solutions (n_generation parameter)
    for n in range(start, args.n_generation):

        # Randomly choses a cell in the map. Then passes stores the respective agent of that cell and the category in two new variables agent & category
        # ------------------------------------------------------------
        if map_elites:
            # Randomly choose a cell from the map elites.
            cell = random.choice(list(map_elites.keys()))
            parts = cell.split(',')
            structure_label = parts[0]
            api_bin = int(parts[1])
            # Define the mapping for API calls bins using only 2 bins.
            api_calls_mapping = {0: "few API calls", 1: "many API calls"}
            category_text = f"{structure_label}, {api_calls_mapping.get(api_bin, 'unknown API calls')}"
            
            selected_agent = map_elites[cell]
            # If the selected cell is empty, search for another cell with the same structure_label.
            if selected_agent is None:
                for key, candidate in map_elites.items():
                    if key.startswith(structure_label + ",") and candidate is not None:
                        selected_agent = candidate
                        break
        else:
            selected_agent = None
            category_text = None
        # ------------------------------------------------------------


        ## Uses pre-defined system prompt to generate new solution
        ## Performs two relfexions to improve quality of new solution
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive, selected_agent, category_text)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:

            next_solution = get_json_response_from_gpt_reflect(msg_list)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None, category=category_text)
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
            n -= 1
            continue
        
        ## Re-evaluation of new solution (debug_max number of re-evaluations)
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
            n -= 1
            continue
        
        ## Fitness computation & adding to archive (if evaluation is passed, fitness score is computed) 
        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        # ------------------------------------------------------------
        ## Call Gemini again similar to before to assess the number of API calls made and add it to next_solution
        if "code" in next_solution:
            next_solution["api_calls"] = count_api_calls(next_solution["code"])
        else:
            print("Warning: next_solution is missing the 'code' field, skipping API call count.")
        # ------------------------------------------------------------
        ## Give Structure label to newly generated solution
        if "thought" in next_solution:
            next_solution["structure_label"] = get_structure_label(next_solution)
        else:
            print("Warning: next_solution is missing the 'thought' field, skipping structure label assignment.")
        # ------------------------------------------------------------


        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
        
        # ------------------------------------------------------------
        # Updates map of elites in case of better performance achieved for that specific cell.
        map_elites = create_map_elites_structure_api(archive,
                                                     bins_api=args.bins_dim2,
                                                     candidate_labels=None,
                                                     min_api=args.min_dim2, max_api=args.max_dim2)
               
        # Store the map of elites after every generation as a new file.
        map_file_path = os.path.join(args.save_dir, f"{args.expr_name}_map_elites_gen{n+1}.json")
        with open(map_file_path, 'w') as f:
            json.dump(map_elites, f, indent=4)
        # ------------------------------------------------------------


def get_upper_bound(upper_bound_string):
    match = re.search(r'\(([\d.]+)%,\s*([\d.]+)%\)', upper_bound_string)
    if match:
        return float(match.group(2))
    else:
        return 0.0

## Used to thoroughly evaluate the new archive
def evaluate(args):

    ## Loads archive and previous evaluaiton results. Prepares file for storing new, thorough evaluations
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    ## Iterative evaluation of candidate agents
    ## Note: Selects candidate agents for evaluation (only include initial + max_agents top performing agents)
    evaluation_candidates = [] # only choosing top agents to evaluate
    print(f"len(archive): {len(archive)}")
    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        
        ## Selects candidates agent (only initial + top performing)            
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
        current_idx += 1
        
        ## Evaluates candidate performance (uses evaluate_forward_fn)
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        ## Saves result of evaluation in new file path
        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

## Used to evaluate a single agent (during search() and evaluate())
def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py

    ## Creates callable function: Converts text description into executable function
    ## and assigns it as an attribute to the agent
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    ## Preparing Evaluation Examples: Creates n number of examples of the environment task
    ## Note 1: Example shuffled using shuffle_seed
    ## Note 2: valid_size is number of examples used for evaluation of new agent during search
    ## Note 3: valide_size + test_size is used during evaluate() for more thorough evaluation
    ## Note 4: This is then repeated n_repeat times

    # set seed 0 for valid set
    examples = get_all_examples()
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    ## Creates Queue of example tasks to solve (formatts them in a proper way)
    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    ## Parallel evaluation of examples (max_workers defines how many examples processed in parallel)
    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    ## Scores the performance as confidence interval
    ## Note: The score is either 0 or 1 and results in a percentage then on average
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            correct_answer = answers[q_idx]
            correct = score_mgsm(correct_answer, extracted_answer)
        except Exception as e:
            acc_list.append(0)
            continue

        acc_list.append(1 if correct else 0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results_mgsm_config2/')
    parser.add_argument('--expr_name', type=str, default="mgsm_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=20)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--max_agents', type=int, default=5)

    # ------------------------------------------------------------
    # Map elites arguments:
    parser.add_argument('--bins_dim1', type=int, default=3, help="Number of bins for performance dimension (default 3)")
    parser.add_argument('--bins_dim2', type=int, default=2, help="Number of bins for API calls dimension (default 2)")
    parser.add_argument('--min_dim1', type=float, default=None, help="Minimum performance value (if not provided, computed from archive)")
    parser.add_argument('--max_dim1', type=float, default=None, help="Maximum performance value (if not provided, computed from archive)")
    parser.add_argument('--min_dim2', type=int, default=None, help="Minimum api_calls value (if not provided, computed from archive)")
    parser.add_argument('--max_dim2', type=int, default=None, help="Maximum api_calls value (if not provided, computed from archive)")
    # ------------------------------------------------------------

    # Arguments for multiple runs to test variance
    parser.add_argument('--num_runs', type=int, default=1, help="Number of runs to execute")
    parser.add_argument('--base_seed', type=int, default=42, help="Base seed value for the first run")



    args = parser.parse_args()

    # Store the original expr_name for later prefixing.
    original_expr_name = args.expr_name

    for run in range(args.num_runs):
        # Update the seed for each run (for example, add the run number to the base_seed)
        args.shuffle_seed = args.base_seed + run
        
        # Set the seed for Python's random module and NumPy at the beginning of each run.
        random.seed(args.shuffle_seed)
        np.random.seed(args.shuffle_seed)

        # Modify expr_name to include the run prefix (run1_, run2_, etc.)
        args.expr_name = f"run{run+1}_{original_expr_name}"
        print(f"Starting run {run+1} with seed {args.shuffle_seed} and expr_name {args.expr_name}")
        
        # Run the search phase with SEARCHING_MODE turned on.
        SEARCHING_MODE = True
        search(args)
        
        # Then perform the evaluation phase.
        SEARCHING_MODE = False
        evaluate(args)
