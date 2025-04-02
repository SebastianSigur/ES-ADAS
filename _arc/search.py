import argparse
import copy
import json
import os
import pickle
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import re
import backoff
import numpy as np
import random
import openai
from tqdm import tqdm
from google import genai
from google.genai import types

from arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt

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

#-------------------------------------------------------------------------------------------------------#
# Function to create and update MAP of Elites after every generation based on archive
def create_map_elites_structure_api(
    archive
):
    """
    Creates a map elites grid from the candidate archive based on two dimensions:
      - Structure label: taken from each solution's "structure_label" field.
      - API calls: the number of API calls (agent['api_calls']).
    Returns:
        dict: A dictionary mapping cell keys (as strings, e.g. "Chain-of-Thought,0")
              to the best candidate (elite) in that cell (the one with the highest fitness).
    """
    # Set number of API call bins
    bins_api = 2

    # Extract API Call values
    api_calls_values = [sol['api_calls'] for sol in archive if 'api_calls' in sol]
    if not api_calls_values:
        return {}
    
    # Determine structure labels
    candidate_labels = [
            "Linear Chain-of-Thought",
            "Iterative Refinement",
            "Tree-of-Thought",
            "Decompositional Reasoning",
            "Multi-Agent Reasoning",
            "Abstraction to Principles Reasoning"
        ]
    
    # Initialize grid with keys as "structure_label,api_bin"
    grid = {f"{label},{i}": None for label in candidate_labels for i in range(bins_api)}
    
    # Go through every agent in archive
    for sol in archive:

        # Skip invalid agents
        if 'structure_label' not in sol or 'api_calls' not in sol or 'fitness' not in sol:
            continue
        
        # Determine structure label
        label = sol["structure_label"]

        # Skip invalid labels
        if label not in candidate_labels:
            continue
        
        # Extract API value
        api_val = sol["api_calls"]

        # Assign API calls to bins (<= 5 for few API calls to 0; 6+ for many API calls to bin 1)
        if api_val <= 5:
            api_bin = 0
        else:
            api_bin = 1
        
        # Allocate agents to niches
        cell_key = f"{label},{api_bin}"
        new_fitness = get_upper_bound(sol['fitness'])
        
        if grid[cell_key] is None:
            grid[cell_key] = sol
        else:
            current_sol = grid[cell_key]
            current_fitness = get_upper_bound(current_sol['fitness'])
            if new_fitness > current_fitness:
                grid[cell_key] = sol

    # Return final grid
    return grid
#-------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------#
# Create API Call label with Gemini
def count_api_calls(forward_code):
    """
    Use Gemini to determine the API count label
    Returns JSON object with 'api_calls' field from agent as integer
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
    
    # Use Gemini to classify API count
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
# Create structure label with Gemini (unnecessary complex as relict from earlier version with included an
# initial check with a BERT model before then passing it to Gemini)
def recheck_label_with_gemini(agent_name, agent_code, candidate_labels):
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
    
    # Use Gemini to determine the structure label
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

# Create structure label with Gemini (unnecessary complex as relict from earlier version with included an
# initial check with a BERT model before then passing it to Gemini)
def get_structure_label(solution):
    """
    Uses Gemini via recheck_label_with_gemini().
    
    Returns:
        A string representing the structure label.
    """
    # Define structure labels
    candidate_labels = [
        "Linear Chain-of-Thought",
        "Iterative Refinement",
        "Tree-of-Thought",
        "Decompositional Reasoning",
        "Multi-Agent Reasoning",
        "Abstraction to Principles Reasoning"
    ]
    
    # Extract code text
    code_text = solution.get("code", "")
    if not code_text:
        return None
        
    # Extract agent name
    agent_name = solution.get("name", "Unknown Agent")

    # Determine structure label
    new_label = recheck_label_with_gemini(agent_name, code_text, candidate_labels)

    # Return structure label
    return new_label
#-------------------------------------------------------------------------------------------------------#

# Ensure that all generated agent have the correct formatting (would otherwise constantly break code)
def validate_agent(agent: dict) -> bool:
    """Ensure agent has all required fields and valid structure label"""
    required_fields = {
        'thought': str,
        'name': str,
        'code': str,
        'fitness': (int,str),
        'generation': (int, str),
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

    ## Initializes and loads archive (uses save_dir & expr_name as locations to load and save archive)
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
    # Determine API call and structure label
    for solution in archive:
        solution["api_calls"] = count_api_calls(solution["code"])
        solution["structure_label"] = get_structure_label(solution)
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
    # Compute initial map of elites
    map_elites = create_map_elites_structure_api(archive)
    # ------------------------------------------------------------
    
    ## Generates n new solutions (n_generation parameter)
    for n in range(start, args.n_generation):

        # ------------------------------------------------------------

        # Sampling of the selected agent (i.e., agent to mutate/inspiration agent)
        if args.agent_sampling == "fitness":
            
            # Fitness-based sampling of selected agent
            while True:
                # Create a list of valid parent cells (cells with non-None agents)
                valid_cells = [key for key in map_elites if map_elites[key] is not None]
                if not valid_cells:
                    raise RuntimeError("No valid agent found in map_elites")
                
                # Compute raw weights from each agent's fitness using get_upper_bound
                raw_weights = [get_upper_bound(map_elites[key]['fitness']) for key in valid_cells]
                
                # Apply softmax transformation using numpy
                temperature = 1.0
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
                
                # If cell empty, use agent from same structure label with different API label
                if selected_agent is None:
                    same_structure_agents = [
                        agent for key, agent in map_elites.items()
                        if key.startswith(f"{parent_structure_label},") and agent is not None
                    ]
                    if same_structure_agents:
                        selected_agent = max(same_structure_agents, key=lambda x: get_upper_bound(x['fitness']))
                
                # If no agent in either cell for this structure, sampling process is restarted
                if selected_agent is not None:
                    break
        
        else:
            # Uniform sampling of selected agent
            while True:
                parent_keys = list(map_elites.keys())
                parent_random_index = random.randint(0, len(parent_keys) - 1)
                parent_cell = parent_keys[parent_random_index]
                parent_parts = parent_cell.split(',')
                parent_structure_label = parent_parts[0]
                parent_api_bin = int(parent_parts[1])
                parent_api_calls_mapping = {0: "few API calls", 1: "many API calls"}
                parent_api_label = parent_api_calls_mapping.get(parent_api_bin, "few API calls")
                
                selected_agent = map_elites[parent_cell]
            
                # If cell empty, use agent from same structure label with different API label
                if selected_agent is None:
                    same_structure_agents = [agent for key, agent in map_elites.items() 
                                            if key.startswith(f"{parent_structure_label},") 
                                            and agent is not None]
                    if same_structure_agents:
                        selected_agent = max(same_structure_agents, 
                                        key=lambda x: get_upper_bound(x['fitness']))
                
                # If no agent in either cell for this structure, sampling process is restarted
                if selected_agent is None:
                    non_empty_agents = [agent for agent in map_elites.values() if agent is not None]
                    if non_empty_agents:
                        selected_agent = random.choice(non_empty_agents)
                
                # If selected agent is found
                if selected_agent is not None:
                    break
        
        # Sampling of the mutation direction
        if args.direction_sampling == "fitness":

            # Fitness-based: weighted by inverted fitness (cells with lower fitness get higher probability)
            # Note: cells with a null agent included by assigning minimum raw fitness

            all_keys = list(map_elites.keys())

            # Compute raw fitness values for non-null cells
            non_null_raws = [get_upper_bound(map_elites[k]['fitness']) for k in all_keys if map_elites[k] is not None]
            min_raw = min(non_null_raws) if non_null_raws else 1

            # Compute raw weights for all cells
            raw_weights_target = []
            for key in all_keys:
                if map_elites[key] is not None:
                    raw_weights_target.append(get_upper_bound(map_elites[key]['fitness']))
                else:
                    raw_weights_target.append(min_raw)

            # Transformation into probability distribution via softmax
            temperature = 1.0
            inverted_weights = np.exp(-np.array(raw_weights_target) / temperature)
            total_inverted = np.sum(inverted_weights)
            softmax_inv = inverted_weights / total_inverted

            # Sample one target cell
            target_cell = random.choices(all_keys, weights=softmax_inv.tolist(), k=1)[0]
            target_parts = target_cell.split(',')
            target_structure_label = target_parts[0]
            target_api_bin = int(target_parts[1])
            api_calls_mapping = {0: "few API calls", 1: "many API calls"}
            target_api_label = api_calls_mapping.get(target_api_bin, "few API calls")        

        else:
            # Uniform sampling of mutation direction
            possible_structure_labels = list(set([cell.split(',')[0] for cell in map_elites.keys()]))
            possible_api_bins = list(range(2))
            target_structure_label = random.choice(possible_structure_labels)
            target_api_bin = random.choice(possible_api_bins)
            api_calls_mapping = {0: "few API calls", 1: "many API calls"}
            target_api_label = api_calls_mapping.get(target_api_bin, "few API calls")

        # # ------------------------------------------------

        ## Generation of new agents
        print(f"============Generation {n + 1}=================")

        # Print statements for tracking of generations
        print(f"Parent Cell: {parent_cell} (Structure: {parent_structure_label}, API: {parent_api_bin})")
        print(f"Mutation Target: Structure {target_structure_label}, API {target_api_label}")  
        
        if selected_agent is None or selected_agent == "Take inspiration from an agent with similar architecture in the archive":
            print(f"Selected Agent: {selected_agent}")
        else:
            print(f"Selected Agent: {selected_agent.get('name', 'Unnamed Agent') if selected_agent else 'No agent found'}")

        system_prompt, prompt = get_prompt(archive, map_elites, args.past_agents, selected_agent, target_structure_label, target_api_label)
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
        ## Determine API count and structure label
        if "code" in next_solution:
            next_solution["api_calls"] = count_api_calls(next_solution["code"])

        if "thought" in next_solution:
            next_solution["structure_label"] = get_structure_label(next_solution)
        # ------------------------------------------------------------


        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        
        # Validation if agent complete
        if not validate_agent(next_solution):
            print(f"Skipping invalid agent from generation {n+1}")
            continue


        # Check if agent is misproduced
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
            map_elites = create_map_elites_structure_api(archive)
            
            # Store the map of elites after every generation as a new file
            map_file_path = os.path.join(args.save_dir, f"{args.expr_name}_map_elites_gen{n+1}.json")
            with open(map_file_path, 'w') as f:
                json.dump(map_elites, f, indent=4)
        
        # Skip misproduced agents
        else:
            print(f"New agent with fitness {next_solution['fitness']} not added; below archive minimum of {min_current_upper}.")
            continue  
        # ------------------------------------------------------------




def get_upper_bound(upper_bound_string):
    match = re.search(r'\(([\d.]+)%,\s*([\d.]+)%\)', upper_bound_string)
    if match:
        return float(match.group(2))
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
    
    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        
        sorted_archive = sorted(archive, key=lambda x: get_upper_bound(x['fitness']), reverse=True)
        
        count = 0
        max_agents = args.max_agents

        for archived_agent in archive:
            if archived_agent['generation'] == "initial":
                evaluation_candidates.append(archived_agent)
        for archived_agent in sorted_archive:
            if archived_agent['generation'] == "initial":
                continue
            if count >= max_agents:
                break
            evaluation_candidates.append(archived_agent)
            count += 1
            
            
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
    parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default='arc_gpt3.5_results')
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--max_agents', type=int, default=5)

    # Arguments for configuration
    parser.add_argument('--agent_sampling', type=str, default='fitness', help="Fitness or uniform sampling of selected agent")
    parser.add_argument('--direction_sampling', type=str, default='fitness', help="Fitness or uniform sampling of mutation direction")
    parser.add_argument('--past_agents', type=str, default='MAP', help="Inclusion of past agents into system prompt. Values are either MAP, Archive, Agent)")

    
    # Arguments for multiple runs to test variance
    parser.add_argument('--num_runs', type=int, default=3, help="Number of runs to execute (default: 3)")
    parser.add_argument('--base_seeds', nargs='+', type=int, default=[42, 45, 47], help="List of seeds for each run. Length must match num_runs")

    args = parser.parse_args()

    # Validate seed configuration
    if len(args.base_seeds) != args.num_runs:
        args.base_seeds = args.base_seeds[:args.num_runs]

    original_expr_name = args.expr_name

    for run_idx, seed in enumerate(args.base_seeds):
        # Generate consistent folder name based on parameters
        args.save_dir = f"run_{args.past_agents}_{args.agent_sampling}_{args.direction_sampling}_gen{args.n_generation}_seed{seed}"
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Update experiment name
        args.expr_name = f"run{run_idx+1}_{original_expr_name}"
        
        print(f"Starting run {run_idx+1}/{args.num_runs} with seed {seed}")
        print(f"Save directory: {args.save_dir}")
        print(f"Experiment name: {args.expr_name}\n")

        # Run search and evaluation
        SEARCHING_MODE = True
        search(args)
        
        SEARCHING_MODE = False
        evaluate(args)