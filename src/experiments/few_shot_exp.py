"""
Few-shot learning experiments for French NLI.

This script runs few-shot experiments with different numbers of examples
and evaluates on multiple datasets.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.data.loader import NLIDataLoader
from src.evaluation.metrics import NLIMetrics
from src.models.api_models import LLMFactory
from src.prompts.templates import PromptBuilder


def run_few_shot_experiment(
    model_name: str,
    eval_dataset: str,
    few_shot_source: str,
    num_shots: int,
    split: str = 'test',
    max_examples: int = None
) -> Dict:
    """
    Run a single few-shot experiment.
    
    Args:
        model_name: Name of the LLM model
        eval_dataset: Dataset to evaluate on
        few_shot_source: Dataset to get few-shot examples from
        num_shots: Number of few-shot examples
        split: Split to evaluate on
        max_examples: Maximum examples to evaluate
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Few-Shot Experiment")
    print(f"Model: {model_name}")
    print(f"Eval Dataset: {eval_dataset}")
    print(f"Few-Shot Source: {few_shot_source}")
    print(f"Num Shots: {num_shots}")
    print(f"{'='*60}")
    
    # Load data
    loader = NLIDataLoader()
    dataset = loader.load_dataset(eval_dataset, split=split)
    
    if max_examples and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))
    
    print(f"Evaluating on {len(dataset)} examples")
    
    # Get few-shot examples
    if num_shots > 0:
        few_shot_examples = loader.get_few_shot_examples(
            few_shot_source,
            num_examples=num_shots,
            strategy='stratified'
        )
        print(f"Using {len(few_shot_examples)} few-shot examples from {few_shot_source}")
    else:
        few_shot_examples = None
        print("Zero-shot (no examples)")
    
    # Initialize model
    model = LLMFactory.create(model_name)
    prompt_builder = PromptBuilder()
    metrics = NLIMetrics()
    
    predictions = []
    true_labels = []
    
    # Evaluate
    print("Running predictions...")
    for example in tqdm(dataset):
        messages = prompt_builder.build_prompt(
            premise=example['premise'],
            hypothesis=example['hypothesis'],
            examples=few_shot_examples,
            format_type='chat'
        )
        
        try:
            response = model.predict(messages)
            pred_label = model.parse_response(response)
            pred_id = NLIMetrics.label_name_to_id(pred_label)
        except Exception as e:
            print(f"Prediction error: {e}")
            pred_id = 1  # Default neutral
        
        predictions.append(pred_id)
        true_labels.append(example['label'])
    
    # Calculate metrics
    metrics.add_batch(predictions, true_labels)
    results = metrics.compute()
    
    # Print results
    metrics.print_results()
    
    return {
        'model': model_name,
        'eval_dataset': eval_dataset,
        'few_shot_source': few_shot_source,
        'num_shots': num_shots,
        'num_examples': len(dataset),
        'split': split,
        'metrics': results
    }


def run_few_shot_grid_search(
    model_name: str,
    eval_datasets: List[str],
    num_shots_list: List[int] = [0, 1, 3, 5, 10],
    few_shot_source: str = 'xnli',
    max_examples: int = None,
    output_dir: str = 'results/few_shot'
):
    """
    Run few-shot experiments with different numbers of shots.
    
    Args:
        model_name: LLM model name
        eval_datasets: Datasets to evaluate on
        num_shots_list: List of num_shots to try
        few_shot_source: Where to get few-shot examples
        max_examples: Max examples per dataset
        output_dir: Output directory
    """
    print(f"\n🔬 Few-Shot Grid Search")
    print(f"Model: {model_name}")
    print(f"Eval Datasets: {', '.join(eval_datasets)}")
    print(f"Num Shots: {num_shots_list}")
    print(f"Few-Shot Source: {few_shot_source}")
    
    all_results = []
    
    for dataset in eval_datasets:
        for num_shots in num_shots_list:
            try:
                result = run_few_shot_experiment(
                    model_name=model_name,
                    eval_dataset=dataset,
                    few_shot_source=few_shot_source,
                    num_shots=num_shots,
                    max_examples=max_examples
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{model_name.replace('/', '_')}_few_shot.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY - Few-Shot Results")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Shots':<10} {'Accuracy':<12} {'F1-Macro':<12}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['eval_dataset']:<20} "
              f"{result['num_shots']:<10} "
              f"{result['metrics']['accuracy']:<12.4f} "
              f"{result['metrics']['f1_macro']:<12.4f}")
    
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Few-shot learning experiments for French NLI'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4, gemini-2.0-flash)'
    )
    parser.add_argument(
        '--eval-datasets',
        type=str,
        nargs='+',
        default=['xnli'],
        help='Datasets to evaluate on'
    )
    parser.add_argument(
        '--num-shots',
        type=int,
        nargs='+',
        default=[0, 1, 3, 5, 10],
        help='Numbers of few-shot examples to try'
    )
    parser.add_argument(
        '--few-shot-source',
        type=str,
        default='xnli',
        help='Dataset to get few-shot examples from'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Max examples per dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/few_shot',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    run_few_shot_grid_search(
        model_name=args.model,
        eval_datasets=args.eval_datasets,
        num_shots_list=args.num_shots,
        few_shot_source=args.few_shot_source,
        max_examples=args.max_examples,
        output_dir=args.output_dir
    )
