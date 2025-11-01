import os
import json
import transformers
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

ARC_SYSTEM_PROMPT_TEMPLATE = """You are given initial example input-output grid pairs from the ARC (Abstraction and Reasoning Corpus) task.
    Each grid is represented as a 2D array of integers ranging from 0 to 9. Each integer corresponds to a specific color:

    0 - Black  
    1 - Blue  
    2 - Red  
    3 - Green  
    4 - Yellow  
    5 - Gray  
    6 - Magenta  
    7 - Orange  
    8 - Light Blue  
    9 - Dark Red  

    The tasks from ARC are based on the following priors:
    - Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
    - Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
    - Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
    - Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.
   
     Your task:
    1. Study the given initial example input-output pairs carefully.
    2. Some examples may be incorrect or noisy — identify the pattern that the *majority* of examples follow.
    3. Infer the correct transformation rule that maps the input grid to the output grid.
    4. Apply this inferred transformation to the provided test input grid to produce the correct output grid.

    Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
    Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
    Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
    Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.
"""
ARC_USER_PROMPT_TEMPLATE = """ Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
    {examples}

    Now, here is the test input grid:

    {test_input}

    Generate the output grid that correctly applies the inferred transformation to this test input.
"""
ARC_OUTPUT_PROMPT_TEMPLATE = """```grid
{output_grid}```"""

def prompt_training(training_challenges_path, training_solutions_path,save_path, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(training_challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(training_solutions_path, 'r') as f:
        solutions = json.load(f)


    prompts = []
    for challenge_id, challenge in challenges.items():
        solution = solutions.get(challenge_id, [])
        if solution == []:
            continue
        train_examples = challenge.get('train', [])
        test_inputs = challenge.get('test', [{}])
        if train_examples == []:
            continue
        train_examples_str = ""
        for i, ex in enumerate(train_examples):
            train_examples_str += f"Example {i+1}:\n"
            input = ex.get('input')
            output = ex.get('output')
            input_str = ""
            output_str = ""
            for ind in range(len(input)):
                for ele in input[ind]:
                    input_str += str(ele)
                input_str += "\n"
            
            for ind in range(len(output)):
                for ele in output[ind]:
                    output_str += str(ele)
                output_str += "\n"
            
            train_examples_str += f"Input:\n```grid\n{input_str}```\nOutput:\n```grid\n{output_str}```\n"

        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            test_output = solution
            if not test_input.get('input'):
                continue
            test_input_json = test_input.get('input', [])
            test_input_str = "Input:\n```grid\n"
            for ind in range(len(test_input_json)):
                for ele in test_input_json[ind]:
                    test_input_str += str(ele)
                test_input_str += "\n"
            test_input_str += "```"
            solution_str = ""
            for ind in range(len(test_output)):
                for ele in test_output[ind]:
                    solution_str += str(ele)
                solution_str += "\n"
            challenge_id_new = challenge_id
            user_prompt = ARC_USER_PROMPT_TEMPLATE.format(
                examples=train_examples_str,
                test_input=test_input_str
            )
            output_prompt = ARC_OUTPUT_PROMPT_TEMPLATE.format(output_grid=solution_str)
            messages = [
                {"role": "system", "content": ARC_SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output_prompt}
            ]
            prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            if(len(test_inputs)>1):
                challenge_id_new = f"{challenge_id}_test{i}"
            prompts.append({
                "challenge_id": challenge_id_new,
                "prompt": prompt_string,
                "output": test_output
            })
            print(f"Generated prompt for challenge_id: {challenge_id_new}")
            
    with open(save_path, 'w') as f:
        json.dump(prompts, f, indent=4)

def prompt_test(challenges_path,save_path, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    prompts = []
    
    for challenge_id, challenge in challenges.items():
        train_examples = challenge.get('train', [])
        test_inputs = challenge.get('test', [{}])
        if train_examples == []:
            continue
        train_examples_str = ""
        for i, ex in enumerate(train_examples):
            train_examples_str += f"Example {i+1}:\n"
            input = ex.get('input')
            output = ex.get('output')
            input_str = ""
            output_str = ""
            for ind in range(len(input)):
                for ele in input[ind]:
                    input_str += str(ele)
                input_str += "\n"
            
            for ind in range(len(output)):
                for ele in output[ind]:
                    output_str += str(ele)
                output_str += "\n"
            
            train_examples_str += f"Input:\n```grid\n{input_str}```\nOutput:\n```grid\n{output_str}```\n"

        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            if not test_input.get('input'):
                continue
            test_input_json = test_input.get('input', [])
            test_input_str = "Input:\n```grid\n"
            for ind in range(len(test_input_json)):
                for ele in test_input_json[ind]:
                    test_input_str += str(ele)
                test_input_str += "\n"
            test_input_str += "```"
            challenge_id_new = challenge_id
            user_prompt = ARC_USER_PROMPT_TEMPLATE.format(
                examples=train_examples_str,
                test_input=test_input_str
            )
            messages = [
                {"role": "system", "content": ARC_SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_prompt},
            ]
            prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if(len(test_inputs)>1):
                challenge_id_new = f"{challenge_id}_test{i}"
            prompts.append({
                "challenge_id": challenge_id_new,
                "prompt": prompt_string,
            })
            
    with open(save_path, 'w') as f:
        json.dump(prompts, f, indent=4)
    


if __name__ == "__main__":
    training_challenges_path = os.path.join("../data/arc-agi-2025/problem_augmented/arc-agi_training_challenges.json")
    training_solutions_path = os.path.join("../data/arc-agi-2025/problem_augmented/arc-agi_training_solutions.json")
    evaluation_challenges_path = os.path.join("../data/arc-agi-2025/test_time_finetuning/arc-agi_evaluation_challenges.json")
    evaluation_solutions_path = os.path.join("../data/arc-agi-2025/test_time_finetuning/arc-agi_evaluation_solutions.json")
    test_challenges_path = os.path.join("../data/arc-agi-2025/test_time_finetuning/arc-agi_test_challenges.json")    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print("Generating training prompts...")
    prompt_training(training_challenges_path,training_solutions_path,save_path="../data/arc-agi-2025/prompts/arc-agi_training_prompts.json", model_name=model_name)
    print("Generating evaluation prompts...")
    prompt_training(evaluation_challenges_path,evaluation_solutions_path,save_path="../data/arc-agi-2025/prompts/arc-agi_evaluation_prompts.json", model_name=model_name)
    print("Generating test prompts...")
    prompt_test(test_challenges_path,save_path="../data/arc-agi-2025/prompts/arc-agi_test_prompts.json", model_name=model_name)