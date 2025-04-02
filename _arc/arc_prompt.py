import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
"""
}

COT_code = {
    "thought": "Directly formatting the output can be challenging. A good practice is to allow the LLM to write the transformation code and then evaluate it to generate the output. This ensures that the output is derived from executable code, improving reliability.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach with code generation
    cot_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instantiate a new LLM agent specifically for CoT with code output
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')
    
    # Get the CoT agent's response, which includes both thinking steps and code
    thinking, code = cot_agent([taskInfo], cot_instruction)
    
    # Evaluate the generated code to get the output
    answer = self.get_test_output_from_code(code)
    
    # Return the final output derived from the code execution
    return answer
    """
}

COT_SC = {
    "thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling. Note that we need to collect only the ones that pass the examples, preventing the context length from becoming too long.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for step-by-step reasoning and code generation
    cot_instruction = "Please think step by step and then solve the task by writing the code."
    N = 5  # Number of CoT agents
    
    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent', temperature=0.7) for _ in range(N)]

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    
    possible_answers = []
    
    # Collect reasoning and answers from each CoT agent
    for i in range(N):
        thinking, code = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.extend([thinking, code])
    
    # Make a final decision based on all collected reasoning and answers
    thinking, code = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    
    return answer
    """
}

Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. After each answer, testing on the examples to provide feedback, and the LLM uses insights from previous attempts and feedback to refine its answer. It is very good practice to use `self.run_examples_and_get_feedback` to get feedback. One should consider trying to use this feedback in future agent design.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    cot_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you went wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    
    # Instantiate a Chain-of-Thought (CoT) agent
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')
    
    N_max = 3  # Maximum number of attempts
    
    # Initial attempt
    thinking, code = cot_agent([taskInfo], cot_initial_instruction, 0)
    
    # Iteratively refine the answer based on feedback
    for i in range(N_max):
        # Get feedback by testing the code on examples
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)  
        
        # Add feedback to the inputs for the next iteration
        attempt = [thinking, code, feedback]

        # Reflect on previous attempts and refine the answer
        # Only consider the latest attempts to control context length. You can try to increase the N_max.
        # The input to LLMAgentBase should be a list of Info.
        thinking, code = cot_agent([taskInfo] + attempt, cot_reflect_instruction, i + 1)  

    # Get the final answer after refinement
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

LLM_debate = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    debate_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer by writing the code."
    
    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'code'], 'Debate Agent', temperature=0.6, role=role) for role in ['Puzzle Game Designer', 'Expert Logician']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)

    max_round = 2  # Maximum number of debate rounds
    all_results = [[] for _ in range(max_round)]
    
    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, code = debate_agents[i]([taskInfo], debate_initial_instruction)
                answer = self.get_test_output_from_code(code)
            else:
                input_infos = [taskInfo] + all_results[r-1]
                thinking, code = debate_agents[i](input_infos, debate_instruction)
                answer = self.get_test_output_from_code(code)
            all_results[r].extend([thinking, answer])
    
    # Make the final decision based on all debate results and solutions
    thinking, code = final_decision_agent([taskInfo] + all_results[max_round-1], final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

QD = {
    "thought": "Similar to Quality-Diversity methods, allowing the LLM to generate multiple diverse and interesting solutions could be beneficial.",
    "name": "Quality-Diversity",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    cot_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for generating another interesting way to solve the task based on previous attempts
    cot_QD_instruction = "Given previous attempts, try to come up with another interesting way to solve the task by writing the code."
    
    # Initialize the Chain-of-Thought (CoT) agent
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')

    # Instruction for final decision-making based on all solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3  # Maximum number of attempts
    qd_inputs = [taskInfo]  # Initialize inputs with the task information

    possible_answers = []
    
    # Generate multiple diverse solutions
    # Different from generating multiple answers through repeated questioning, we generate interestingly new solutions based on previous attempts
    for i in range(N_max):
        # Generate a solution based on the instruction (initial or QD)
        # Also control the context length.
        thinking, code = cot_agent(qd_inputs[-3:], cot_initial_instruction if i == 0 else cot_QD_instruction, i)
        # Get feedback by testing the code on examples
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
        # Add the solution to inputs for the next iteration
        qd_inputs.extend([thinking, code, feedback])  
        # Collect all possible answers
        possible_answers.append({
            'thinking': thinking,
            'code': code,
            'feedback': feedback,
            'correct_count': len(correct_examples)
        })

    # Sort the possible answers based on the number of correct examples in descending order
    sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)
    
    # Select the top solutions (e.g., top 2 solutions)
    top_solutions = sorted_answers[:2]

    # Prepare inputs for the final decision agent
    final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]

    # Make the final decision based on all solutions
    thinking, code = final_decision_agent(final_inputs, final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the ARC (Abstraction and Reasoning Corpus) challenge.
In this challenge, each task consists of three demonstration examples, and one test example. Each Example consists of an “input grid” and an “output grid”. Test-takers need to use the transformation rule learned from the examples to predict the output grid for the test example.
# An example task from ARC challenge:

## Task Overview:
You will be given some number of paired example inputs and outputs grids. The outputs were produced by applying a transformation rule to the input grids. In addition to the paired example inputs and outputs, there is also one test input without a known output.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color. 0 is black.
Your task is to determine the transformation rule from examples and find out the answer, involving determining the size of the output grid for the test and correctly filling each cell of the grid with the appropriate color or number.

The transformation only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the grid visually, and try to find the pattern.

## Examples:
### Example 0:
input = [[0,0,0,0,5,0,0,0,0],[0,0,0,0,5,0,0,0,0],[0,0,0,4,5,0,0,0,0],[0,0,0,4,5,4,4,0,0],[0,0,3,3,5,0,0,0,0],[0,0,0,3,5,0,0,0,0],[0,0,0,3,5,3,3,3,0],[0,0,0,3,5,0,0,0,0],[0,0,0,0,5,0,0,0,0],[0,0,0,0,5,0,0,0,0]]
output = [[0,0,0,0],[0,0,0,0],[0,0,0,4],[0,0,4,4],[0,0,3,3],[0,0,0,3],[0,3,3,3],[0,0,0,3],[0,0,0,0],[0,0,0,0]]

### Example 1:
input = [[0,0,0,0,5,0,0,0,0],[0,0,0,2,5,0,0,0,0],[0,0,0,2,5,2,6,0,0],[0,0,0,2,5,0,0,0,0],[0,0,0,2,5,2,2,2,0],[0,0,6,6,5,6,0,0,0],[0,0,0,2,5,0,0,0,0],[0,2,2,0,5,2,0,0,0],[0,0,0,2,5,0,0,0,0],[0,0,0,0,5,0,0,0,0]]
output = [[0,0,0,0],[0,0,0,2],[0,0,6,2],[0,0,0,2],[0,2,2,2],[0,0,6,6],[0,0,0,2],[0,2,2,2],[0,0,0,2],[0,0,0,0]]

### Example 2:
input = [[0,0,0,0,5,0,0,0,0],[0,0,0,0,5,7,0,0,0],[0,0,0,8,5,0,0,0,0],[0,0,0,8,5,0,0,0,0],[0,7,8,8,5,0,0,0,0],[0,0,0,0,5,8,8,0,0],[0,0,0,8,5,0,0,0,0],[0,0,0,8,5,0,0,0,0],[0,0,0,0,5,8,7,0,0],[0,0,0,0,5,0,0,0,0]]
output = [[0,0,0,0],[0,0,0,7],[0,0,0,8],[0,0,0,8],[0,7,8,8],[0,0,8,8],[0,0,0,8],[0,0,0,8],[0,0,7,8],[0,0,0,0]]

## Test Problem:
Given input:
[[0,0,0,0,5,0,0,0,0],[0,0,0,1,5,0,0,0,0],[0,0,0,1,5,1,0,0,0],[0,1,1,1,5,1,1,1,6],[0,0,0,6,5,6,6,0,0],[0,0,0,0,5,1,1,1,0],[0,0,0,1,5,0,0,0,0],[0,0,0,1,5,1,6,0,0],[0,0,0,0,5,6,0,0,0],[0,0,0,0,5,0,0,0,0]]

Analyze the transformation rules based on the provided Examples and determine what the output should be for the Test Problem.

# The utility code:

```python
from collections import namedtuple, Union
import numpy as np
import json

from utils import random_id, format_arc_data, eval_solution, list_to_string, bootstrap_confidence_interval
import openai
import backoff

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
def FORMAT_INST(request_keys):
    return f\"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n\"""

# Role description for the LLM
def ROLE_DESC(role):
    return f"You are a {role}."

# Instruction for the code function
CODE_INST = \"""You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should ensure that you implement a version of the transformation that works for both example and test inputs.\"""

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

class LLMAgentBase:
    \"""
    Base class for an LLM agent.

    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        An example of a generated prompt:
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.

        \"""
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
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt 

    def query(self, input_infos: list[Info], instruction: str, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.

        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.

        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list[Info], instruction: str, iteration_idx=-1) -> list[Info]:
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    def __init__(self, examples: list[dict], test_input: list[list[int]]) -> None:
        \"""
        Initializes the AgentArchitecture with examples and a test input.
        
        Args:
            examples (List[Dict[str, List[List[int]]]]): A list of dictionaries, where each dictionary contains an 'input' and 'output'.
                - 'input' (List[List[int]]): A 2D list representing the input grid.
                - 'output' (List[List[int]]): A 2D list representing the expected output grid for the corresponding input.
            test_input (List[List[int]]): The input grid for the test problem, which is a 2D list. The agent's task is to determine the correct output for this input.
        
        Note:
            You are free to use these data in any way that aids in solving the task.
        \"""
        self.examples = examples
        self.test_input = test_input
    
    def run_examples_and_get_feedback(self, code):
        \"""
        Runs provided code on examples and gets feedback. This is very useful to provide feedback to the generated transform code.

        Args:
        - code (Info/str): The CODE to evaluate.

        Returns:
        - Info: Feedback on the code whether it works on examples or not. The content is a string summarizing the success and failure on examples.
        - list(list[dict[str, list[list[int]]]]): list of Correct Examples
        - list(list[dict[str, list[list[int]]]]): list of Wrong Examples
            - keys for both correct and wrong example dict:
            - 'input' (list[list[int]]): A 2D list representing the input grid.
            - 'output' (list[list[int]]): A 2D list representing the expected output grid for the corresponding input.

        An example of feedback Info content:
        "Your transform function generates a WRONG answer in Example 0!
        Expect: xxx
        You got: yyy
        Observe the Example 0 carefully!

        Your transform function generates a CORRECT answer in Example 1!
        ..."
        \"""
        examples = self.examples

        #... (code to run the provided code on examples and get feedback)
            
        # return feedback, correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        \"""
        Gets the output from the code on the test input.

        Args:
        - code (Info/str): The code to evaluate.

        Returns:
        - Info: Output on the test input with the provided code, which is the answer to the task.
        \"""
        test_input = self.test_input

        #... (code to run the provided code on the test input and get the output)
            
        # return transform_output

    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str, list[list[int]]]:
        \"""
        Placeholder method for processing task information.

        Args:
        - taskInfo (Info): Task information.

        Returns:
        - Answer (Union[Info, str, list[list[int]]]): Your FINAL answer. Return either a named tuple Info or a string of answer or a list[list[int]].
        \"""
        pass


#... Code to implement the evaluation of the agent's performance on ARC tasks
```
[PAST_AGENTS]

# Agent's fitness value
The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".
The "generation" number indicates the sequential order of attempts made in designing the architecture. Each generation represents a distinct iteration or version, reflecting the evolution and refinement of the design.

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. DON'T try to use some function that doesn't exisit. In forward(), you need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it might be helpful to set the LLM’s role and temperature to further control the LLM’s response. It is also helpful to allow chain-of-thought thinking in addition to your required output fields. Note that the LLMAgentBase() will automatically parse the output and return a list of “Info”. And when you query LLMAgentBase(), it takes in a list of "Info". DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.
In this domain, because you have been given training examples, you could choose to test your proposed solution against those training examples to see if it is correct. One example way to use this domain’s API is:
```
possible_answers = []
# got some new answers
thinking, code = ...
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
# collect possible answers
possible_answers.append({
    'thinking': thinking,
    'code': code,
    'feedback': feedback,
    'correct_count': len(correct_examples)
})
# Sort the possible answers based on the number of correct examples in descending order
sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)

# Select the top solutions (e.g., top 3 solutions)
top_solutions = sorted_answers[:3]

# And then use the top_solutions anyway you want.
# One example is to use it for final decision
final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]
# Make a final decision based on the top solutions
thinking, code = final_decision_agent(final_inputs, final_decision_instruction)
```

## WRONG Implementation examples:
Here are some mistakes you may make:
1. This is WRONG: ```
thinking, code = code_generator_agent([taskInfo] + rules, code_generation_instruction)
feedback_info = verifier_agent([taskInfo, Info('thinking', 'Code Generator Agent', thinking, 0), Info('code', 'Code Generator Agent', code, 0)], verification_instruction)
```
First, it is wrong to use "Info('thinking', 'Code Generator Agent', thinking, 0)". The returned "thinking" and "code" is already Info.
Second, directly let a agent provide feedback to a generated code is not a good practice. You could use `self.run_examples_and_get_feedback(code)` to get feedback first, as it will test the code on examples. And then you could use this feedback anyway you like, including providing it to another critic agent.

2. This is WRONG: ```
# Debugging: Log the generated code
print('Generated Code:', dynamic_memory['code'])
feedback_info = verifier_agent([taskInfo, Info('thinking', 'Code Generator Agent', thinking, 0), Info('code', 'Code Generator Agent', code, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work. The only way to check how many examples are passed in the generated code is to get feedback from `self.run_examples_and_get_feedback` and check the correct_count.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.

3. This is WRONG: ```
# Verify the transformation rules using the feedback mechanism
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(rules)
```
`self.run_examples_and_get_feedback()` can only take in CODE input.

5. This is WRONG: ```
for thinking, code in module_solutions:
    # Implement actual integration logic
    integrated_solution.append(code)  # Placeholder
```
You should not put any unfinished code or placeholder in your code.

6. This is WRONG: ```
answer = code
return answer
```
You should always use self.get_test_output_from_code() to run the code and return the output answer.

7. This is Wrong: ```
queried_solutions = []
query_infos = active_query_agent(input_infos, active_query_instruction)
query_thinking, query_code = query_infos[0], query_infos[1]
# THIS IS VERY WRONG! YOU DON'T NEED TO MAKE INFO YOURSELF IN MOST CASES. JUST CONCATENATE THE PREVIOUS OUTPUTS!
queried_solutions.append(Info('queried_solution', 'Active Query Agent', {'thinking': query_thinking.content, 'code': query_code.content}, iteration_idx=solution.iteration_idx))
```
Correct way will be: `queried_solutions.extend([query_thinking, query_code])`

8. This is Wrong: ```# Generate sub-goals using the manager agent
manager_agent = LLMAgentBase(['thinking', 'sub_goals'], 'Manager Agent')
thinking, sub_goals = manager_agent([taskInfo], manager_instruction)
sub_goals_content = sub_goals.content[:len(actor_agents)]
```
Info.content is a string. You can not slice them. What you can do is to require more output fields like `LLMAgentBase(['thinking', 'subgoal1', 'subgoal2', ...], 'Manager Agent')`.
If you want to give multiple infos into other agents, please use "queried_solutions + [query_thinking, query_code]".

9. This is Wrong: ```# Collect solutions that pass at least one example
possible_answers = []
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) > 0:  
    possible_answers.extend([thinking, code, feedback])

10. This is NOT recommended: ```
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) == len(self.examples):
    break  # If all examples are correct, break the loop
```
Even if all examples are correct, it doesn't guarantee that the code is correct. Don't break the loop in your agent.

# Sort the possible answers based on the number of correct examples in descending order
sorted_answers = sorted(possible_answers, key=lambda x: x.content['correct_count'], reverse=True)
```
You can not sort in this way. The correct method is: ```# Collect solutions that pass at least one example
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) > 0:  
    possible_answers.append({
        'code': code,
        'feedback': feedback,
        'correct_count': len(correct_examples)
    })
sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)
```

DON'T make those mistakes.

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature.
Your goal is to maximize "fitness" by designing an improved agent that is specifically tailored to the category: [STRUCTURE_LABEL] and [API_LABEL]. You are provided with a selected agent as inspiration: [SELECTED_AGENT].
Your task is to mutate and refine this agent to create a better-performing variant that meets the structure [STRUCTURE_LABEL] and has [API_LABEL]. To achieve this, you will perform two mutations: A structure mutation and an API call mutation.

Observe the discovered architectures carefully and consider the insights, lessons, or stepping stones they provide.
Draw inspiration from related LLM agent papers or academic literature from other research areas. Focus on modifications that can enhance performance while optimizing resource usage in line with the specified category.
THINK OUTSIDE THE BOX.

I. STRUCTURE MUTATION:
Your mutated code must conform exactly to the targeted structure specified by [STRUCTURE_LABEL]. Follow these instructions precisely:

1. Understand the Target Structure:
   - "Linear Chain-of-Thought": A single, straight-line execution with one call to agent() and no loops, branches, or feedback mechanisms.
   - "Iterative Refinement": A loop-based design where the same agent() call is invoked repeatedly (with feedback or modified inputs) to refine the answer.
   - "Tree-of-Thought": A branching architecture where, at key decision points, multiple reasoning paths are generated, and one branch is selected for the final answer.
   - "Decompositional Reasoning": A design that breaks the problem into sub-tasks solved by distinct agent instances (or separate calls), with their results combined to produce the final answer.
   - "Multi-Agent Reasoning": A design that concurrently instantiates two or more unique agent() instances (not reusing the same one in a loop) and coordinates their outputs (e.g., via voting or consensus) to decide the final answer.
   - "Abstraction to Principles Reasoning": A two-phase process where the agent first abstracts the problem into high-level principles and then uses these abstractions to guide the solution.

2. Implementation Guidelines:
Step 1. Incorporate Core Features from the Selected Agent:
   - Analyze the selected agent ([SELECTED_AGENT]) and identify its most impactful reasoning components—those that have contributed significantly to its performance.
   - Prioritize and integrate only these high-performing features into your mutated design, rather than incorporating every minor detail.
   - For instance, if the selected agent employs a particular feedback loop or abstraction mechanism that correlates with high fitness, incorporate that feature while ignoring less impactful elements.
Step 2. Perform the structure mutation:
   - Modify the control flow to reflect the target structure. For example, if [STRUCTURE_LABEL] is "Iterative Refinement", introduce a clear loop that repeatedly calls the agent() method with updated inputs.
   - Ensure that agent instantiation patterns match the target:
       * For "Multi-Agent Reasoning", instantiate at least two unique LLMAgentBase objects (do not simply reuse one inside a loop).
       * For "Linear Chain-of-Thought", ensure there is only a single agent() call without any loops or branches.
       * For "Tree-of-Thought", incorporate conditional branches or multiple calls that represent divergent reasoning paths, followed by a selection step.
       * For "Decompositional Reasoning", structure the code to split the problem first into independent, smaller sub-tasks that are then in a second step solved independently (possibly by different agent instances), before then combining their outputs in a third step.
       * For "Abstraction to Principles Reasoning", structure the code into two phases: first, extract and process high-level principles; then, use these principles to generate the final answer.

3. Self-Review:
   - Before finalizing your mutated code, carefully review the overall control flow and agent instantiation pattern to ensure it matches the target structure [STRUCTURE_LABEL].

4. Example Structure Mutation Strategies:
▸ From Linear → Multi-Agent: Split monolithic calls into specialized agents  
▸ From Multi → Iterative: Add feedback gathering between rounds  
▸ From Iterative → Tree: Convert loop into conditional branches  
▸ From Abstraction → Tree: Create principle-specific reasoning paths 

II. API CALL MUTATION:
Follow these instructions precisely:

1. Stay within Target API Calls:
Your mutated code must meet the targeted API call count specified by [API_LABEL].
   - If [API_LABEL] is "few API calls", your final code must include between 1 and 5 calls to agent().
   - If [API_LABEL] is "many API calls", your final code must include more than 5 calls to agent().

2. Counting Rules:
   - Only count invocations of the agent() method (do NOT count LLMAgentBase instantiations).
   - Count every agent() call, regardless of its location (inside loops, conditionals, etc.).
   - Include concise inline comments indicating the number of calls per code block (e.g., "Loop: 3 iterations x 1 call = 3 calls").

3. Self-Review:
   - Before finalizing your code, carefully review it to ensure that the total number of agent() calls falls exactly within the required range.
   - If the agent does not meet the specified number of API calls, reiterate on the agent's code to achieve the required API calls.

4. Examples of API call mutations:
========= Mutation to Few API Call Category =================
# Original Agent (2 calls)
def forward():
    agent1 = LLMAgentBase()
    result1 = agent1()  # 1 call
    result2 = agent1()  # 1 call (Total: 2)

# Mutated Agent (4 calls - "Few API" compliant)
def forward():
    agent1 = LLMAgentBase()
    # Initial analysis phase
    for _ in range(2):  # 2 iterations × 1 call = 2 calls
        agent1()  
    
    # Final refinement
    inputs = [taskInfo, previous_results]
    agent1(inputs)  # 1 call
    
    # Validation step
    if needs_correction:  # Always True path
        agent1()  # 1 call (Total: 2+1+1=4)
========= Mutation to Many API Call Category =================
# Original Agent (3 calls)
def forward():
    agents = [LLMAgentBase() for _ in range(3)]
    for a in agents:
        a()  # 3 calls

# Mutated Agent (7 calls - "Many API" compliant)
def forward():
    # Parallel agent pool
    agents = [LLMAgentBase() for _ in range(3)]  # 0 calls (instantiation)
    
    # First debate round
    for a in agents:  # 3 iterations × 1 call = 3
        a()  
    
    # Second refinement round 
    for i in range(2):  # 2 iterations × 2 agents = 4
        for a in agents[:2]:  
            a()  # (Total: 3+4=7)

III. FITNESS MAXIMIZATION IMPROVEMENT:
Your final solution must not only meet the target structure and API call constraints but also achieve the highest possible performance (fitness) on the benchmark. To maximize fitness, you must:
   - Critically analyze the entire reasoning chain and computational steps to ensure they produce accurate, meaningful, and non-zero results.
   - Optimize key computations by streamlining feedback loops, refining arithmetic operations, and eliminating redundant data manipulations to reduce noise and enhance accuracy.
   - Adjust prompt phrasing, LLM roles, temperature settings, and hyperparameters to encourage deeper, more robust reasoning.
   - Prioritize and incorporate only high-impact features from the selected agent ([SELECTED_AGENT]); evaluate each inherited component for its contribution to fitness and discard minor or redundant elements.
   - Compare your mutated solution against high-performing benchmarks or baseline performance metrics, iterating until the performance is significantly improved.
   - Validate your design using simulated test cases to ensure that the logic effectively solves the benchmark.
   - Eliminate any default or constant outputs that do not represent genuine computation.

IV. CODE QUALITY ASSURANCE:
Your final mutated code must:
   - Be syntactically correct and runnable without errors.
   - Produce meaningful outputs (non-zero accuracy) rather than returning a constant or zero value.
   - Pass a self-review of code quality: double-check for potential syntax issues, logical errors, and ensure that all required functions execute as intended.
   - Data Type and Functional Correctness: Ensure that all variables and operations use the correct data types and that arithmetic and data structure manipulations are valid. Verify that your code’s evaluation produces realistic, non-zero values.
   - Focus on delivering a robust solution that not only meets the target structure and API call constraints but also avoids coding errors leading to 0 accuracy.

V. EVALUATION ASSURANCE:
Before finalizing your mutated code, ensure that the agent actually performs a computation to solve the benchmark:
   - Analyze the logic to confirm that the code processes the input taskInfo and produces a meaningful result.
   - Check that the code does not simply return a default value (e.g., 0 or constant) or an empty result.
   - Internally verify that all computations, data manipulations, and output assignments are correct and yield non-zero, plausible values.
   - Ensure that the solution would effectively solve the intended benchmark rather than just meeting structural or API constraints.

VI. OUTPUT FORMAT ENFORCEMENT:
Your final mutated code must:
   - Use the exact function interface specified.
   - Include the required instruction, input information, and output fields for the LLM agents.
   - Ensure the output format exactly matches the following example:
   
[EXAMPLE]

   - Set the LLM’s role and temperature as needed.
   - DO NOT FORGET the taskInfo input when calling the agent.

IMPORTANT RULES:
[RULES]
These rules MUST be followed strictly. Any solution that violates these rules will be rejected.
"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

Before providing your solution, you MUST verify:
1. Does your implementation follow this critical rule?:
Look through the code again independently. You have to detect if the code breaks the rules. REMEMBER the rules are as follows:
[RULES]

2. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

3. **Implementation Mistakes**:
Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.
REMEMBER checking "## WRONG Implementation examples" in the prompt. MAKE SURE TO FOLLOW THE RULES STRICTLY: [RULES]

4. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"rule_verification": Explain the rules you must follow, and explain how you verified rule compliance.

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Make sure to follow the rules strictly: [RULES]
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""

RULES = lambda api_threshold: (
    "RULE: In the forward() function, every single occurrence of LLMAgentBase(...)(...) counts as one usage. "
    "This means that if you call an LLMAgentBase instance more than once—even if it's the same instance—each call is counted separately. "
    f"The total number of such calls must be lower thannot exceed {api_threshold}. "
    "This includes calls made inside loops, conditionals, or any nested structures. "
    "No exceptions: every call is counted individually. "
    "Strict adherence to this rule is mandatory."
)

def get_init_archive():
    return [COT_code, Reflexion, LLM_debate, COT_SC, QD]

def get_prompt(current_archive, current_map, past_agent_parameter, selected_agent=None, structure_label=None, api_label=None, adaptive=False):
    # Convert the archive to a JSON string
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"

    # Convert the map to a JSON string
    map_str = ",\n".join([json.dumps(sol) for sol in current_map])
    map_str = f"[{map_str}]"

    # Replace [EXAMPLE]
    prompt = base.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    # Include past agents based on past_agent_parameter
    if past_agent_parameter == "MAP":
        template_str = """# Discovered Architectures 
Below are the discovered architectures:
 
[MAP_ELITES]"""
        prompt = prompt.replace("[PAST_AGENTS]", template_str)
        prompt = prompt.replace("[MAP_ELITES]", map_str)
    
    elif past_agent_parameter == "Archive":
        template_str = """# Discovered Architectures 
Below are the discovered architectures:
 
[ARCHIVE]"""
        prompt = prompt.replace("[PAST_AGENTS]", template_str)
        prompt = prompt.replace("[ARCHIVE]", archive_str)
    
    else:
        template_str = """# Selected Agent
Below is the architecture of the selected agent:
 
[SELECTED_AGENT]"""
        prompt = prompt.replace("[PAST_AGENTS]", template_str)

    # Generate rules for API calls
    rules = RULES(api_label if api_label is not None else "few API calls")
    prompt = prompt.replace("[RULES]", rules)
    
    # Add in mutation direction
    if structure_label is not None:
        prompt = prompt.replace("[STRUCTURE_LABEL]", structure_label)
    else:
        prompt = prompt.replace("[STRUCTURE_LABEL]", "")
    
    if api_label is not None:
        prompt = prompt.replace("[API_LABEL]", api_label)
    else:
        prompt = prompt.replace("[API_LABEL]", "")
    
    # Replace [LABEL DESCRIPTION] with corresponding description based on structure_label
    label_descriptions = {
        "Linear Chain-of-Thought": "The agent produces its final answer in a single, linear chain-of-thought without any iterative self-refinement or use of multiple agents.",
        "Iterative Refinement": "The agent continually repcrosses its chain-of-thought, revising, re-evaluating, and self-assessing its intermediate steps - to progressively converge on a robust final answer.",
        "Tree-of-Thought": "The agent creates a tree-of-thought by dynamically branches out at key decision points, exploring multiple reasoning paths and selectively following the most promising branch to arrive at the final answer.",
        "Decompositional Reasoning": "The agent breaks down a complex problem into independent sub-problems, solves each one separately, and then integrates these solutions into a cohesive final answer.",
        "Multi-Agent Reasoning": "The agent concurrently creates several LLM instances that interact with one another and create different reasoning trajectories. The agent aggreates the outcome from the different LLM instances - such as through voting or consensus - to produce the final decision. Common mistake: A single agent generating multiple responses  is NOT multi-agent reasoning. Multi-agent reasoning requires multiple LLMAgentBase instances with coordination.",
        "Abstraction to Principles Reasoning": "First abstracts the problem’s details into high-level principles, then uses these abstractions to guide the solution."
    }

    label_description = label_descriptions.get(structure_label, "")
    prompt = prompt.replace("[LABEL DESCRIPTION]", label_description)
    
    # Add in selected agent
    if selected_agent is not None:
        prompt = prompt.replace("[SELECTED_AGENT]", json.dumps(selected_agent))
    else:
        prompt = prompt.replace("[SELECTED_AGENT]", "")
    
    return system_prompt, prompt



def get_reflexion_prompt(prev_example, structure_label=None, api_label=None):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = (Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str)
          if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", ""))
    
    # Generate rules
    rules = RULES(api_label if api_label is not None else "")
    r1 = r1.replace("[RULES]", rules)
    
    reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
    Make sure to follow the rules strictly: [RULES]
    Your response should be organized as follows:
    Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
    """
    reflexion_prompt_2 = reflexion_prompt_2.replace("[RULES]", rules)
    
    return r1, reflexion_prompt_2
