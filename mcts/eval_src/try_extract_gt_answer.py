import sys
sys.path.append('.')
from mcts.eval_src.Evaluator import MATHEvaluator
from datasets import load_dataset

def main():
    dataset_path = '/pubshare/fwk/code/MCGEP/dataset/math/train.jsonl'
    dataset_name = 'MATH'
    evaluator = eval(f"{dataset_name}Evaluator()")
    # math_dataset = load_dataset(dataset_path)
    ext = 'json'
    math_dataset = load_dataset(ext, data_files=dataset_path)
    for data in math_dataset['train']:
        gt_solution = data['solution']
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
        print(gt_answer)
        if gt_answer is None:
            breakpoint()


if __name__ == '__main__':
    main()
