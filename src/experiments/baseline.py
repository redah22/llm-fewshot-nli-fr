"""
Baseline experiment - Evaluate pre-trained models on French NLI datasets.

This script evaluates baseline models (e.g., models trained on XNLI) 
on all available French NLI datasets.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.data.loader import NLIDataLoader
from src.evaluation.metrics import NLEMetrics
from src.models.api_models import LLMFactory
from src.prompts.templates import PromptBuilder


def evaluate_model_on_dataset(
    model,
    dataset_name: str,
    split: str = 'test',
    max_examples: int = None,
    use_few_shot: bool = False,
    num_shots: int = 0
) -> Dict:
    """
    Evaluate a model on a specific dataset.
    
    Args:
        model: Model instance (from LLMFactory or local model)
        dataset_name: Name of the dataset
        split: Dataset split to evaluate on
        max_examples: Maximum number of examples to evaluate (None = all)
        use_few_shot: Whether to use few-shot prompting
        num_shots: Number of few-shot examples
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name.upper()} ({split})")
    print(f"{'='*60}")
    
    # Load dataset
    loader = NLIDataLoader()
    dataset = loader.load_dataset(dataset_name, split=split)
    
    # Limit examples if specified
    if max_examples and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))
    
    print(f"Number of examples: {len(dataset)}")
    
    # Get few-shot examples if needed
    few_shot_examples = None
    if use_few_shot and num_shots > 0:
        print(f"Using {num_shots}-shot prompting")
        few_shot_examples = loader.get_few_shot_examples(
            dataset_name,
            num_examples=num_shots,
            strategy='stratified'
        )
    
    # Prepare for evaluation
    prompt_builder = PromptBuilder()
    metrics = NLIMetrics()
    
    predictions = []
    true_labels = []
    
    # Evaluate
    print("Running predictions...")
    for example in tqdm(dataset):
        premise = example['premise']
        hypothesis = example['hypothesis']
        true_label_id = example['label']
        
        # Create prompt
        messages = prompt_builder.build_prompt(
            premise=premise,
            hypothesis=hypothesis,
            examples=few_shot_examples,
            format_type='chat'
        )
        
        # Get prediction
        try:
            response = model.predict(messages)
            pred_label_name = model.parse_response(response)
            pred_label_id = NLIMetrics.label_name_to_id(pred_label_name)
        except Exception as e:
            print(f"Error during prediction: {e}")
            pred_label_id = 1  # Default to neutral
        
        predictions.append(pred_label_id)
        true_labels.append(true_label_id)
    
    # Calculate metrics
    metrics.add_batch(predictions, true_labels)
    results = metrics.compute()
    
    # Print results
    metrics.print_results()
    
    # Return results
    return {
        'dataset': dataset_name,
        'split': split,
        'num_examples': len(dataset),
        'num_shots': num_shots if use_few_shot else 0,
        'metrics': results
    }


def run_baseline_experiments(
    model_name: str,
    datasets: List[str],
    split: str = 'test',
    max_examples: int = None,
    output_dir: str = 'results/baseline'
):
    """
    Run baseline experiments on multiple datasets.
    
    Args:
        model_name: Name of the model to use
        datasets: List of dataset names to evaluate on
        split: Dataset split to use
        max_examples: Maximum examples per dataset
        output_dir: Directory to save results
    """
    print(f"\n🚀 Running Baseline Experiments")
    print(f"Model: {model_name}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Split: {split}")
    if max_examples:
        print(f"Max examples per dataset: {max_examples}")
    
    # Create model
    print(f"\nInitializing model: {model_name}...")
    model = LLMFactory.create(model_name)
    
    # Run experiments
    all_results = []
    
    for dataset_name in datasets:
        try:
            results = evaluate_model_on_dataset(
                model=model,
                dataset_name=dataset_name,
                split=split,
                max_examples=max_examples,
                use_few_shot=False,
                num_shots=0
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            continue
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{model_name.replace('/', '_')}_baseline.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Accuracy':<10} {'F1-Macro':<10}")
    print("-" * 60)
    
    for result in all_results:
        dataset = result['dataset']
        accuracy = result['metrics']['accuracy']
        f1_macro = result['metrics']['f1_macro']
        print(f"{dataset:<20} {accuracy:<10.4f} {f1_macro:<10.4f}")
    
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline evaluation of models on French NLI datasets'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4, gemini-2.0-flash, mistral-large)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['xnli'],
        help='Datasets to evaluate on (default: xnli)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples per dataset (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    run_baseline_experiments(
        model_name=args.model,
        datasets=args.datasets,
        split=args.split,
        max_examples=args.max_examples,
        output_dir=args.output_dir
    )
