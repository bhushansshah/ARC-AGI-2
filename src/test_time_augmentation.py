import os
import json
from matplotlib.pyplot import grid
import numpy as np
from src.utils.augmentation import change_color, padding_grid, horizontal_scale_grid, vertical_scale_grid
from src.utils.helpers import convert_ndarray
import argparse
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



def augment_grid_color(challenge, challenge_solution):
    """Apply color change to generate additional augmented grid according to type_to_augment."""
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    color_map = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        color_changed_input, color_map = change_color(training_sample_input, color_map=color_map)
        color_changed_output, color_map = change_color(training_sample_output, color_map=color_map)
        augmented_challenge['train'].append({'input': color_changed_input, 'output': color_changed_output})

    augmented_test_input_grid = change_color(challenge['test'][0]['input'], color_map=color_map)[0]
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = change_color(challenge_solution, color_map=color_map)[0]
    return augmented_challenge, augmented_solution

def augment_padding(challenge, challenge_solution):
    """Apply padding to generate additional augmented grid according to type_to_augment."""
    #randomly choose padding value between 0 and 9
    padding_value = np.random.randint(0, 10)
    #randomly choose whether to pad each side or not with 50% chance
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        padding_top = np.random.randint(0, 5) 
        padding_bottom = np.random.randint(0, 5)
        padding_left = np.random.randint(0, 5)
        padding_right = np.random.randint(0, 5)
        padded_input = padding_grid(training_sample_input, padding_top, padding_bottom, padding_left, padding_right, padding_value)
        augmented_challenge['train'].append({'input': padded_input, 'output': training_sample_output})
    
    padding_top = np.random.randint(0, 5)
    padding_bottom = np.random.randint(0, 5)
    padding_left = np.random.randint(0, 5)
    padding_right = np.random.randint(0, 5)
    augmented_test_input_grid = padding_grid(challenge['test'][0]['input'], padding_top, padding_bottom, padding_left, padding_right, padding_value)
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = challenge_solution
    return augmented_challenge, augmented_solution

def augment_horizontal_scaling(challenge, challenge_solution):
    """Apply horizontal scaling to generate additional augmented grid according to type_to_augment."""
    #each cell will be scaled in the horizontal direction by 2. meaning each cell will become two cells side by side with the same value
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        scaled_input = horizontal_scale_grid(training_sample_input)
        scaled_output = horizontal_scale_grid(training_sample_output)
        augmented_challenge['train'].append({'input': scaled_input, 'output': scaled_output})

    augmented_test_input_grid = horizontal_scale_grid(challenge['test'][0]['input'])
    augmented_test_output_grid = horizontal_scale_grid(challenge_solution)
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = augmented_test_output_grid
    return augmented_challenge, augmented_solution

def augment_vertical_scaling(challenge, challenge_solution):
    """Apply vertical scaling to generate additional augmented grid according to type_to_augment."""
    #each cell will be scaled in the vertical direction by 2. meaning each cell will become two cells on top of each other with the same value
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        scaled_input = vertical_scale_grid(training_sample_input)
        scaled_output = vertical_scale_grid(training_sample_output)
        augmented_challenge['train'].append({'input': scaled_input, 'output': scaled_output})

    augmented_test_input_grid = vertical_scale_grid(challenge['test'][0]['input'])
    augmented_test_output_grid = vertical_scale_grid(challenge_solution)
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = augmented_test_output_grid
    return augmented_challenge, augmented_solution

def augment_both_scaling(challenge, challenge_solution):
    """Apply both horizontal and vertical scaling to generate additional augmented grid according to type_to_augment."""
    #each cell will be scaled in both directions by 2. meaning each cell will become four cells in a 2x2 block with the same value
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        scaled_input = vertical_scale_grid(horizontal_scale_grid(training_sample_input))
        scaled_output = vertical_scale_grid(horizontal_scale_grid(training_sample_output))
        augmented_challenge['train'].append({'input': scaled_input, 'output': scaled_output})

    augmented_test_input_grid = vertical_scale_grid(horizontal_scale_grid(challenge['test'][0]['input']))
    augmented_test_output_grid = vertical_scale_grid(horizontal_scale_grid(challenge_solution))
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = augmented_test_output_grid
    return augmented_challenge, augmented_solution

def augment_dataset_for_test_time_finetuning(base_dir, challenges_filename, solutions_filename, save_base_dir):
    if save_base_dir is None or challenges_filename is None:
        return

    with open(os.path.join(base_dir, challenges_filename), 'r') as f:
        dataset = json.load(f)

    for challenge_id, challenge in dataset.items():
        for sample in challenge.get('train', []):
            sample['input'] = np.array(sample['input'])
            if 'output' in sample:
                sample['output'] = np.array(sample['output'])

    augmented_dataset = {}
    augmented_solutions = {}
    augmented_challenge_ids = set()
    counter = 0
    for challenge_id, challenge in dataset.items():
        if len(challenge.get('train', [])) == 0 or len(challenge.get('test', [])) == 0:
            print(f"Skipping challenge {challenge_id} due to insufficient data.")
            continue

        original_challenge_id = challenge_id.split('_')[0]
        if original_challenge_id in augmented_challenge_ids:
            continue

        count = 0
        training_samples = challenge['train']
        print(f"No. of training samples - ", len(training_samples))
        for i in range(7):
            # Randomly select a training sample
            sample_ind = np.random.randint(len(training_samples))
            sample = training_samples[sample_ind]
            new_challenge = {
                'train':[],
                "test": []
            }
            new_challenge_solution = None
            for indx in range(len(training_samples)):
                if indx == sample_ind:
                    new_challenge['test'].append({
                        'input': sample['input']
                    })
                    new_challenge_solution = sample['output']
                else:
                    new_challenge['train'].append(training_samples[indx])

            print(f"No. of training samples in new challenge - ", len(new_challenge['train']))
            augmented_challenge, augmented_solution = augment_grid_color(new_challenge, new_challenge_solution)
            new_challenge_id = f"{original_challenge_id}_{count}"
            augmented_dataset[new_challenge_id] = augmented_challenge
            augmented_solutions[new_challenge_id] = augmented_solution
            count += 1

        for i in range(4):
            # Randomly select a training sample
            sample_ind = np.random.randint(len(training_samples))
            sample = training_samples[sample_ind]
            new_challenge = {
                'train':[],
                "test": []
            }
            new_challenge_solution = None
            for indx in range(len(training_samples)):
                if indx == sample_ind:
                    new_challenge['test'].append({
                        'input': sample['input']
                    })
                    new_challenge_solution = sample['output']
                else:
                    new_challenge['train'].append(training_samples[indx])

            augmented_challenge, augmented_solution = augment_padding(new_challenge, new_challenge_solution)
            new_challenge_id = f"{original_challenge_id}_{count}"
            augmented_dataset[new_challenge_id] = augmented_challenge
            augmented_solutions[new_challenge_id] = augmented_solution
            count += 1
    
        for i in range(4):
            # Randomly select a training sample
            sample_ind = np.random.randint(len(training_samples))
            sample = training_samples[sample_ind]
            new_challenge = {
                'train':[],
                "test": []
            }
            new_challenge_solution = None
            for indx in range(len(training_samples)):
                if indx == sample_ind:
                    new_challenge['test'].append({
                        'input': sample['input']
                    })
                    new_challenge_solution = sample['output']
                else:
                    new_challenge['train'].append(training_samples[indx])

            augmented_challenge, augmented_solution = augment_horizontal_scaling(new_challenge, new_challenge_solution)
            new_challenge_id = f"{original_challenge_id}_{count}"
            augmented_dataset[new_challenge_id] = augmented_challenge
            augmented_solutions[new_challenge_id] = augmented_solution
            count += 1

            augmented_challenge, augmented_solution = augment_vertical_scaling(new_challenge, new_challenge_solution)
            new_challenge_id = f"{original_challenge_id}_{count}"
            augmented_dataset[new_challenge_id] = augmented_challenge
            augmented_solutions[new_challenge_id] = augmented_solution
            count += 1

            augmented_challenge, augmented_solution = augment_both_scaling(new_challenge, new_challenge_solution)
            new_challenge_id = f"{original_challenge_id}_{count}"
            augmented_dataset[new_challenge_id] = augmented_challenge
            augmented_solutions[new_challenge_id] = augmented_solution
            count += 1

        counter += 1
        print(f"Processed challenge {challenge_id}, processed {counter} original challenges so far...")

    challenges_save_path = os.path.join(save_base_dir, challenges_filename)
    solutions_save_path = os.path.join(save_base_dir, solutions_filename)
    with open(challenges_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_dataset), f, indent=4)

    with open(solutions_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_solutions), f, indent=4)

    print(f"Augmented dataset saved at: {challenges_save_path}")
    print(f"Augmented solutions saved at: {solutions_save_path}")
    return challenges_save_path, solutions_save_path

# --------------------------
# Command line usage
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment challenges by generating more problems')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the dataset JSON files')
    parser.add_argument('--challenges_filename', type=str, required=True, help='Filename of the challenges JSON file')
    parser.add_argument('--solutions_filename', type=str, required=False, help='Filename of the solutions JSON file')
    parser.add_argument('--save_base_dir', type=str, required=True, help='Directory to save the augmented dataset JSON files')
    args = parser.parse_args()
    augment_dataset_for_test_time_finetuning(args.base_dir, args.challenges_filename, args.solutions_filename, args.save_base_dir)
