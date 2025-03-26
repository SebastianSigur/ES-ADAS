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
        candidate_labels = [
            "Linear Chain-of-Thought",
            "Iterative Refinement",
            "Tree-of-Thought",
            "Decompositional Reasoning",
            "Multi-Agent Reasoning",
            "Abstraction to Principles Reasoning"
        ]
    
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
        # Use a simple condition: if API calls are <= 5, assign bin 0 ("few API calls"), else bin 1 ("many API calls").
        if api_val <= 5:
            api_bin = 0
        else:
            api_bin = 1
        
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
    system_prompt = """You are a code analysis expert specializing in LLM agent architectures. Analyze the Python function to count API calls according to these STRICT RULES:

    1. Counting Principles:
    - ONLY count AGENT METHOD CALLS (agent())
    - DO NOT count LLMAgentBase instantiations
    - Count ALL calls regardless of nesting or scope

    2. Loop Handling:
    - FOR/WHILE loops: Multiply counts by iterations
    - List comprehensions: Treat as implicit loops
    - Nested loops: Multiply counts at all levels

    3. Special Cases:
    - Recursive calls: Count each stack frame
    - Conditional calls: Count ALL possible paths
    - Reused agents: Count each execution separately

    4. Calculation Steps:
    1. Find all agent() method calls
    2. Analyze loop structures
    3. Multiply counts by loop iterations
    4. Sum all execution counts

    5. REVISION STEP - CHECK FOR COMMON MISTAKES:
    BEFORE FINALIZING, VERIFY:
    1. LOOP ITERATIONS: Multiply nested loops (rounds×agents)
    2. CONDITIONAL PATHS: Take MAX branch
    3. REUSED AGENTS: Count per execution
    4. SHADOWED AGENTS: Catch loop-redefines
    5. ROUTING LOGIC: Only executed agents
    6. GENERATORS: Treat as implicit loops
    7. PHASE SEPARATION: No init counts

    EXAMPLES:

    === Complex Loop (Original) ===
    for _ in range(3):
        agent(); helper() → 6 calls

    === Variable-Length Loop (New) ===
    N = 3  # Dynamic iterations
    for _ in range(N): agent() → 3 calls

    === Hybrid Agent Pattern (New) ===
    base = LLMAgentBase()
    for _ in range(2):
        base()          # 2
        LLMAgentBase()() # 2 → Total 4

    === Multi-Branch Conditional (New) ===
    if X: a() 
    elif Y: a(); a() 
    else: pass → Count 2

    === Variable Shadowing (New) ===
    for _ in range(2):
        agent = LLMAgentBase()  # New each iter
        agent() → 2 calls

    === Generator Pattern (New) ===
    agents = (LLMAgentBase() for _ in range(3))
    sum(agent() for agent in agents) → 3

    === Routing Logic (Original) ===
    router(); experts[choice]() → 2

    === Nested Debate (Original) ===
    2 rounds × 2 agents = 4

    === Collaborative Refinement (Original) ===
    3 agents × 2 phases = 6

    === Phase Separation (Original) ===
    agent1 = LLMAgentBase() → 0
    agent1() → 1

    === Recursive Pattern (Original) ===
    run(2) → 3 calls

    Return JSON: {"api_calls": integer}"""
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
        "You are a specialist in LLM agent architecture analysis. Your task is to classify agents by their fundamental reasoning structure through rigorous code examination. "
        "Follow this analytical process:\n\n"
        "1. STRUCTURAL ANALYSIS: Identify key architectural patterns in the code related to:\n"
        "  - Control Flow:\n"
        "       * Linear: No loops/conditionals altering reasoning path\n"
        "       * Iterative: Look for FOR/WHILE loops with:\n"
        "           - Feedback incorporation (previous answers/feedback as input)\n"
        "           - Same LLMAgentBase instance reused\n"
        "       * Branching: IF/ELSE or SWITCH that create distinct reasoning paths\n"
        "   - Agent Count:\n"
        "       * Count ALL LLMAgentBase() instantiations\n"
        "       * When counting LLMAgentBase instances, only consider unique instantiations; if a single instance is reused within a loop, this does not count as multiple agents but is a sign for Iterative Refinement."
        "       * Multi-Agent = 2+ unique instances (not reused in loops)\n"
        "   - Coordination Mechanisms:\n"
        "       * Voting/Consensus: Aggregation of multiple agent outputs\n"
        "       * Debate: Agents directly reference each other's outputs\n"
        "       * Parallel Execution: Simultaneous agent calls (list comprehensions)\n"
        "   - Abstraction Markers:\n"
        "       * Explicit principle extraction (e.g., 'principle_agent')\n"
        "       * Problem decomposition into sub-tasks\n"
        "   - Self-Reflection:\n"
        "       * Critique/feedback loops modifying inputs\n"
        "       * Explicit correctness checking\n\n"
        "2. LABEL MAPPING: Match patterns to these EXCLUSIVE categories:\n"
        "   1. Linear Chain-of-Thought: The agent produces its final answer in a single, linear chain-of-thought without any iterative self-refinement or use of multiple agents.\n"
        "   2. Iterative Refinement: The agent continually repcrosses its chain-of-thought, revising, re-evaluating, and self-assessing its intermediate steps - to progressively converge on a robust final answer.\n"
        "       - REQUIRED: Single LLMAgentBase instance reused in loop\n"
        "       - REQUIRED: Modified inputs between iterations (feedback/previous answers)\n"
        "       - Even if generating diverse answers, STILL Iterative if single agent\n"
        "       - Common misclassification trap: Voting over single-agent outputs ≠ Multi-Agent\n\n"
        "   3. Tree-of-Thought: The agent creates a tree-of-thought by dynamically branches out at key decision points, exploring multiple reasoning paths and selectively following the most promising branch to arrive at the final answer.\n"
        "   4. Decompositional Reasoning: The agent breaks down a complex problem into independent sub-problems, solves each one separately, and then integrates these solutions into a cohesive final answer.\n"
        "   5. Multi-Agent Reasoning: The agent concurrently creates several LLM instances that interact with one another and create different reasoning trajectories. The agent aggreates the outcome from the different LLM instances - such as through voting or consensus - to produce the final decision. Common mistake: A single agent generating multiple responses  is NOT multi-agent reasoning. Multi-agent reasoning requires multiple LLMAgentBase instances with coordination.\n"
        "       - MUST HAVE ALL:\n"
        "        * 2+ unique LLMAgentBase instances (NOT reused in loops)\n"
        "        * Explicit coordination between agents\n"
        "        * Parallel execution (not sequential)\n" 
        "      - FORBIDDEN: Any agent instance reuse\n"
        "   6. Abstraction to Principles Reasoning: First abstracts the problem’s details into high-level principles, then uses these abstractions to guide the solution.\n"
        "3. Check for common classification mistakes:\n"
        "   - Do not misclassify a single LLMAgentBase instance used in a loop as Multi-Agent Reasoning—this pattern should be identified as Iterative Refinement.\n"
        "   - Do not classify agents that loop over a single, reused LLMAgentBase instance as Multi-Agent Reasoning. Multi-Agent Reasoning requires multiple unique LLMAgentBase instances with inter-agent coordination.\n"
        "   - If principle extraction is present, even in a single API call, the agent must be labeled as Abstraction to Principles Reasoning.\n\n"
        "4. DECISION RULES:\n"
        "   - Prioritize structural implementation over described intent\n"
        "   - If multiple patterns exist, choose the DOMINANT structural paradigm\n"
        "   - Reject any hybrid categories - force into the single best fit\n"
        "   - Check your decision for the common mistakes above\n"
        "5. Examples of correct classification:\n"
        "   === Example 1 ===\n"
        "   Agent Name: Quality-Diversity\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')\n"
        "       for i in range(3):\n"
        "           thinking, answer = cot_agent(...)  # Same instance reused\n"
        "       return final_decision_agent(answers)\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Control Flow: Iterates over a list of unique agent instances to generate diverse outputs\n"
        "   - Agent Count: 3 unique LLMAgentBase instances for answer generation plus 1 separate final decision agent\n"
        "   - Coordination: Independent outputs are aggregated via a final decision agent using a consensus mechanism\n"
        "   - Key Pattern: Multiple independent agents generate diverse answers that are coordinated to produce the final decision\n"
        "   Rationale: This architecture instantiates multiple unique agents with explicit coordination, which meets all the criteria for Multi-Agent Reasoning.\n"
        "   Correct Label: Multi-Agent Reasoning\n\n"

        "   === Example 2 ===\n"
        "   Agent Name: Self-Refine (Reflexion)\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       cot_agent = LLMAgentBase(...)\n"
        "       critic_agent = LLMAgentBase(...)\n"
        "       for i in range(5):\n"
        "           feedback = critic_agent(...)\n"
        "           thinking, answer = cot_agent(...)  # Sequential refinement\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Control Flow: Explicit feedback loop with 5 iterations\n"
        "   - Agent Count: 2 agents but ONLY cot_agent refined\n"
        "   - Key Pattern: Iterative input modification (cot_inputs.extend())\n"
        "   Rationale: Main refinement loop focuses on cot_agent with growing input context, making Iterative Refinement the dominant pattern despite critic presence.\n"
        "   Correct Label: Iterative Refinement\n\n"

        "   === Example 3 ===\n"
        "   Agent Name: Dynamic Assignment of Roles\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       expert_agents = [LLMAgentBase(...) for role in roles]\n"
        "       choice = routing_agent(...)\n"
        "       return expert_agents[choice](taskInfo)\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Control Flow: Sequential execution with routing\n"
        "   - Agent Count: Multiple agents but only ONE used\n"
        "   - Key Pattern: Problem decomposition into selection + execution\n"
        "   Rationale: Agents work independently - no coordination between experts, pure decomposition into sub-tasks.\n"
        "   Correct Label: Decompositional Reasoning\n\n"

        "   === Example 4 ===\n"
        "   Agent Name: LLM Debate\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       debate_agents = [LLMAgentBase(...), LLMAgentBase(...), LLMAgentBase(...)]\n"
        "       answers = [agent(taskInfo) for agent in debate_agents]\n"
        "       return consensus(answers)\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Agent Count: 3 unique instances\n"
        "   - Coordination: Parallel execution + consensus\n"
        "   - Key Pattern: No instance reuse, true parallelism\n"
        "   Rationale: Multiple independent agents with aggregated output fulfills all Multi-Agent requirements.\n"
        "   Correct Label: Multi-Agent Reasoning\n\n"

        "   === Example 5 ===\n"
        "   Agent Name: Step-back Abstraction\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       principle_agent = LLMAgentBase(...)\n"
        "       principle = principle_agent(...)\n"
        "       answer = solver_agent([principle])\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Abstraction Markers: Explicit principle extraction\n"
        "   - Control Flow: Two-stage process (principles → solution)\n"
        "   - Key Pattern: Principle output directly guides solver\n"
        "   Rationale: Clear separation of abstraction and application phases, even with multiple agents.\n"
        "   Correct Label: Abstraction to Principles Reasoning\n\n"

        "   === Example 6 ===\n"
        "   Agent Name: Chain-of-Thought\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       cot_agent = LLMAgentBase(...)\n"
        "       thinking, answer = cot_agent(taskInfo)\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Control Flow: Single pass, no loops\n"
        "   - Agent Count: 1 instance\n"
        "   - Key Pattern: Straight-line execution\n"
        "   Rationale: No iteration, branching, or coordination - pure linear processing.\n"
        "   Correct Label: Linear Chain-of-Thought\n\n"

        "   === Example 7 ===\n"
        "   Agent Name: Principle Evaluation and Solution Agent\n"
        "   Agent's Code Structure:\n```python\n"
        "   def forward(self, taskInfo):\n"
        "       combined_agent = LLMAgentBase(...)\n"
        "       output = combined_agent([taskInfo], instruction)\n"
        "       return output[3]\n"
        "   ```\n"
        "   Structural Analysis:\n"
        "   - Abstraction Markers: Explicit principle identification in instruction\n"
        "   - Control Flow: Single call with integrated phases\n"
            "   - Key Pattern: Abstraction guides solution in one step\n"
        "   Rationale: Combined process still emphasizes principle-guided solving as core paradigm.\n"
        "   Correct Label: Abstraction to Principles Reasoning\n\n"  
        "6. Output Format: Output only the final label prediction (which must exactly match one of the provided candidate labels) with no additional explanation or text."        
    )
    
    user_prompt = (
        f"AGENT ANALYSIS REQUEST\n"
        f"Agent Name: {agent_name}\n"
        f"Agent's Code Structure:\n```python\n{agent_code}\n```\n\n"
        f"Required Classification: Select SOLELY from these structural categories - {', '.join(candidate_labels)}\n\n"
        "Focus exclusively on the code's architectural patterns, NOT the problem domain or description."
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
    If the classifier confidence is low (≤ 0.9), it rechecks using Gemini via recheck_label_with_gemini().
    
    Returns:
        A string representing the structure label.
    """
    # Define the candidate labels for structure classification.
    candidate_labels = [
        "Linear Chain-of-Thought",
        "Iterative Refinement",
        "Tree-of-Thought",
        "Decompositional Reasoning",
        "Multi-Agent Reasoning",
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
    
    # If the confidence is low (≤ 0.9), recheck using Gemini.
    if score <= 0.9:
        # Use .get() to safely retrieve "name", providing a default if missing.
        agent_name = solution.get("name", "Unknown Agent")
        new_label = recheck_label_with_gemini(agent_name, thought_text, code_text, candidate_labels)
        return new_label if new_label is not None else predicted_label
    else:
        return predicted_label
#-------------------------------------------------------------------------------------------------------#


def validate_agent(agent: dict) -> bool:
    """Ensure agent has all required fields and valid structure label"""
    required_fields = {
        'thought': str,
        'name': str,
        'code': str,
        'fitness': (int,str),
        'generation': (int, str),  # Allow "initial" or int
        'api_calls': (int, str),
        'structure_label': str
    }
    
    valid_structure_labels = [
        "Linear Chain-of-Thought",
        "Iterative Refinement",
        "Tree-of-Thought",
        "Decompositional Reasoning",
        "Multi-Agent Reasoning",
        "Abstraction to Principles Reasoning"
    ]

    # Check required fields
    for field, field_type in required_fields.items():
        if field not in agent:
            print(f"Agent missing required field: {field}")
            return False
            
        if not isinstance(agent[field], field_type):
            print(f"Invalid type for {field}: {type(agent[field])}")
            return False

    # Validate structure label
    if agent['structure_label'] not in valid_structure_labels:
        print(f"Invalid structure label: {agent['structure_label']}")
        return False

    # Validate code content
    if not agent['code'].strip().startswith('def forward('):
        print("Invalid code format - missing forward function")
        return False

    return True

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
    ## Gemini's response should include only the number of API calls made. Add that information to each.
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

        # ------------------------------------------------------------
        # # First sampling: select parent cell for agent
        # while True:  # Keep searching until we find a valid agent
        #     parent_keys = list(map_elites.keys())
        #     parent_random_index = random.randint(0, len(parent_keys) - 1)
        #     parent_cell = parent_keys[parent_random_index]
        #     parent_parts = parent_cell.split(',')
        #     parent_structure_label = parent_parts[0]
        #     parent_api_bin = int(parent_parts[1])
        #     parent_api_calls_mapping = {0: "few API calls", 1: "many API calls"}
        #     parent_api_label = parent_api_calls_mapping.get(parent_api_bin, "few API calls")
            
        #     selected_agent = map_elites[parent_cell]
          
        #     # First fallback: try same structure agents
        #     if selected_agent is None:
        #         same_structure_agents = [agent for key, agent in map_elites.items() 
        #                                 if key.startswith(f"{parent_structure_label},") 
        #                                 and agent is not None]
        #         if same_structure_agents:
        #             selected_agent = max(same_structure_agents, 
        #                             key=lambda x: get_upper_bound(x['fitness']))
            
        #     # Second fallback: if still None, search entire archive
        #      # Second fallback: if still None, search entire archive
        #     if selected_agent is None:
        #         non_empty_agents = [agent for agent in map_elites.values() if agent is not None]
        #         if non_empty_agents:
        #             selected_agent = random.choice(non_empty_agents)
            
        #     # If we found a valid agent, break the loop
        #     if selected_agent is not None:
        #         break

        # First sampling: select parent cell for agent
        while True:  # Keep searching until we find a valid agent
            # Create a list of valid parent cells (cells with non-None agents)
            valid_cells = [key for key in map_elites if map_elites[key] is not None]
            if not valid_cells:
                raise RuntimeError("No valid agent found in map_elites")
            
            # Compute raw weights from each agent's fitness using get_upper_bound
            raw_weights = [get_upper_bound(map_elites[key]['fitness']) for key in valid_cells]
            
            # Apply softmax transformation using numpy
            temperature = 1.0  # Adjust as needed
            exp_weights = np.exp(np.array(raw_weights) / temperature)
            total_exp = np.sum(exp_weights)
            softmax_weights = exp_weights / total_exp
            
            # Sample one parent cell weighted by the softmax probabilities
            parent_cell = random.choices(valid_cells, weights=softmax_weights.tolist(), k=1)[0]
            parent_parts = parent_cell.split(',')
            parent_structure_label = parent_parts[0]
            parent_api_bin = int(parent_parts[1])
            parent_api_calls_mapping = {0: "few API calls", 1: "many API calls"}
            parent_api_label = parent_api_calls_mapping.get(parent_api_bin, "few API calls")
            
            selected_agent = map_elites[parent_cell]
            
            # First fallback: try same structure agents if the selected cell is empty
            if selected_agent is None:
                same_structure_agents = [
                    agent for key, agent in map_elites.items()
                    if key.startswith(f"{parent_structure_label},") and agent is not None
                ]
                if same_structure_agents:
                    selected_agent = max(same_structure_agents, key=lambda x: get_upper_bound(x['fitness']))
            
            if selected_agent is not None:
                break


        # # Second sampling: select target structure and API labels
        # possible_structure_labels = list(set([cell.split(',')[0] for cell in map_elites.keys()]))
        # possible_api_bins = list(range(args.bins_dim2))  # 0 to bins_dim2-1
        # target_structure_label = random.choice(possible_structure_labels)
        # target_api_bin = random.choice(possible_api_bins)
        # api_calls_mapping = {0: "few API calls", 1: "many API calls"}
        # target_api_label = api_calls_mapping.get(target_api_bin, "few API calls")

        # Second sampling: select target structure and API labels,
        # weighted by inverted fitness (cells with lower fitness get higher probability),
        # and include cells with a null agent by assigning them the minimum raw fitness.

        all_keys = list(map_elites.keys())

        # Compute raw fitness values for non-null cells
        non_null_raws = [get_upper_bound(map_elites[k]['fitness']) for k in all_keys if map_elites[k] is not None]
        min_raw = min(non_null_raws) if non_null_raws else 1  # fallback to 1 if all cells are null

        # Compute raw weights for all cells: if cell is null, assign min_raw
        raw_weights_target = []
        for key in all_keys:
            if map_elites[key] is not None:
                raw_weights_target.append(get_upper_bound(map_elites[key]['fitness']))
            else:
                raw_weights_target.append(min_raw)

        # Apply softmax transformation on inverted raw weights using numpy
        temperature = 2.0  # Adjust as needed
        inverted_weights = np.exp(-np.array(raw_weights_target) / temperature)
        total_inverted = np.sum(inverted_weights)
        softmax_inv = inverted_weights / total_inverted  # NumPy array of probabilities

        # Sample one target cell weighted by the inverted softmax probabilities
        target_cell = random.choices(all_keys, weights=softmax_inv.tolist(), k=1)[0]
        target_parts = target_cell.split(',')
        target_structure_label = target_parts[0]
        target_api_bin = int(target_parts[1])
        api_calls_mapping = {0: "few API calls", 1: "many API calls"}
        target_api_label = api_calls_mapping.get(target_api_bin, "few API calls")
        # ------------------------------------------------

        ## Uses pre-defined system prompt to generate new solution
        ## Performs two relfexions to improve quality of new solution
        print(f"============Generation {n + 1}=================")

        print(f"Parent Cell: {parent_cell} (Structure: {parent_structure_label}, API: {parent_api_bin})")
        print(f"Mutation Target: Structure {target_structure_label}, API {target_api_label}")  
        
        if selected_agent is None or selected_agent == "Take inspiration from an agent with similar architecture in the archive":
            print(f"Selected Agent: {selected_agent}")
        else:
            print(f"Selected Agent: {selected_agent.get('name', 'Unnamed Agent') if selected_agent else 'No agent found'}")

        system_prompt, prompt = get_prompt(archive, selected_agent, target_structure_label, target_api_label)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:

            next_solution = get_json_response_from_gpt_reflect(msg_list)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None, target_structure_label, target_api_label)
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
        
        # Add validation before adding to archive
        if not validate_agent(next_solution):
            print(f"Skipping invalid agent from generation {n+1}")
            continue  # Skip this agent

        # Check if the new agent is at least as good as the worst in the archive
        current_upper_bounds = [get_upper_bound(agent['fitness']) for agent in archive]
        min_current_upper = min(current_upper_bounds) if current_upper_bounds else 0.0
        new_upper = get_upper_bound(next_solution['fitness'])

        if new_upper >= min_current_upper:
            archive.append(next_solution)
            print(f"Added new agent with fitness {next_solution['fitness']} to archive.")

            # Save results
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as json_file:
                json.dump(archive, json_file, indent=4)
            
            # Update map of elites
            map_elites = create_map_elites_structure_api(archive,
                                                        bins_api=args.bins_dim2,
                                                        candidate_labels=None,
                                                        min_api=args.min_dim2, max_api=args.max_dim2)
            
            # Store the map of elites after every generation as a new file
            map_file_path = os.path.join(args.save_dir, f"{args.expr_name}_map_elites_gen{n+1}.json")
            with open(map_file_path, 'w') as f:
                json.dump(map_elites, f, indent=4)
        else:
            print(f"New agent with fitness {next_solution['fitness']} not added; below archive minimum of {min_current_upper}.")
            continue  # Skip this agent and proceed to next generation
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

    # Shuffle with local random seed
    local_random = random.Random(args.shuffle_seed)
    examples = get_all_examples()
    local_random.shuffle(examples)

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
    parser.add_argument('--save_dir', type=str, default='results_mgsm_softmax_sampling_direction_3_no_archive/')
    parser.add_argument('--expr_name', type=str, default="mgsm_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=30)
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
        run_seed = args.base_seed + run
        
        # Set the global seeds per run
        random.seed(run_seed)
        np.random.seed(run_seed)


        # Modify expr_name to include the run prefix (run1_, run2_, etc.)
        args.expr_name = f"run{run+1}_{original_expr_name}"
        print(f"Starting run {run+1} with seed {run_seed} and expr_name {args.expr_name}")
        
        # Run the search phase with SEARCHING_MODE turned on.
        SEARCHING_MODE = True
        search(args)
        
        # Then perform the evaluation phase.
        SEARCHING_MODE = False
        evaluate(args)
