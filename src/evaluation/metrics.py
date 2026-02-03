"""
Evaluation metrics for NLI tasks.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class NLIMetrics:
    """Metrics calculator for NLI evaluation."""
    
    LABEL_NAMES = ['entailment', 'neutral', 'contradiction']
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
    
    def add_batch(self, predictions: List[int], labels: List[int]):
        """
        Add a batch of predictions and labels.
        
        Args:
            predictions: List of predicted label IDs
            labels: List of true label IDs
        """
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with computed metrics
        """
        if not self.all_predictions or not self.all_labels:
            raise ValueError("No predictions to evaluate. Call add_batch first.")
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # F1 scores
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
        }
        
        # Add per-class metrics
        for i, label_name in enumerate(self.LABEL_NAMES):
            metrics[f'precision_{label_name}'] = precision[i]
            metrics[f'recall_{label_name}'] = recall[i]
            metrics[f'f1_{label_name}'] = f1[i]
            metrics[f'support_{label_name}'] = int(support[i])
        
        # Add confusion matrix
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def get_classification_report(self) -> str:
        """
        Get a formatted classification report.
        
        Returns:
            String with classification report
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        return classification_report(
            y_true, 
            y_pred,
            target_names=self.LABEL_NAMES,
            digits=4
        )
    
    def print_results(self):
        """Print formatted results."""
        metrics = self.compute()
        
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  F1-Macro:    {metrics['f1_macro']:.4f}")
        print(f"  F1-Micro:    {metrics['f1_micro']:.4f}")
        print(f"  F1-Weighted: {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for label in self.LABEL_NAMES:
            print(f"\n  {label.upper()}:")
            print(f"    Precision: {metrics[f'precision_{label}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_{label}']:.4f}")
            print(f"    F1:        {metrics[f'f1_{label}']:.4f}")
            print(f"    Support:   {metrics[f'support_{label}']}")
        
        print(f"\nConfusion Matrix:")
        print("             Predicted")
        print("           ", "  ".join([l[:3].upper() for l in self.LABEL_NAMES]))
        cm = np.array(metrics['confusion_matrix'])
        for i, label in enumerate(self.LABEL_NAMES):
            print(f"  {label[:3].upper():3s}  ", end="")
            print("  ".join([f"{cm[i][j]:4d}" for j in range(3)]))
        
        print("\n" + "=" * 60)
    
    @staticmethod
    def label_name_to_id(label_name: str) -> int:
        """Convert label name to ID."""
        label_map = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }
        return label_map.get(label_name.lower(), 1)  # Default to neutral
    
    @staticmethod
    def label_id_to_name(label_id: int) -> str:
        """Convert label ID to name."""
        return NLIMetrics.LABEL_NAMES[label_id]


def evaluate_predictions(
    predictions: List[str],
    labels: List[str]
) -> Dict[str, float]:
    """
    Convenience function to evaluate predictions.
    
    Args:
        predictions: List of predicted label names
        labels: List of true label names
        
    Returns:
        Dictionary with metrics
    """
    # Convert to IDs
    pred_ids = [NLIMetrics.label_name_to_id(p) for p in predictions]
    label_ids = [NLIMetrics.label_name_to_id(l) for l in labels]
    
    # Calculate metrics
    metrics = NLIMetrics()
    metrics.add_batch(pred_ids, label_ids)
    
    return metrics.compute()


if __name__ == '__main__':
    # Example usage
    print("Testing NLI Metrics...\n")
    
    # Simulated predictions
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # True labels
    y_pred = [0, 1, 2, 0, 2, 2, 0, 1, 1, 0]  # Predictions (some errors)
    
    metrics = NLIMetrics()
    metrics.add_batch(y_pred, y_true)
    
    # Print results
    metrics.print_results()
    
    # Get classification report
    print("\nClassification Report:")
    print(metrics.get_classification_report())
