import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Tuple, List, Dict


class RecommenderEvaluator:
    def __init__(self):
        """Initialize the recommender system evaluator."""
        self.predictions = None
        self.true_labels = None
        self.thresholds = None

    def load_data(self, predictions: np.ndarray, true_labels: np.ndarray):
        """
        Load prediction and ground truth data.

        Parameters:
        predictions (np.ndarray): Predicted scores/probabilities
        true_labels (np.ndarray): Actual binary labels (0 or 1)
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true labels must have same length")

        self.predictions = predictions
        self.true_labels = true_labels

    def compute_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve points.

        Returns:
        Tuple containing false positive rates, true positive rates, and thresholds
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No data loaded. Call load_data first.")

        fpr, tpr, thresholds = roc_curve(self.true_labels, self.predictions)
        return fpr, tpr, thresholds

    def compute_auc(self) -> float:
        """
        Compute Area Under the ROC Curve (AUC).

        Returns:
        float: AUC score
        """
        fpr, tpr, _ = self.compute_roc_curve()
        return auc(fpr, tpr)

    def compute_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve points.

        Returns:
        Tuple containing precision, recall, and thresholds
        """
        precision, recall, thresholds = precision_recall_curve(
            self.true_labels, self.predictions)
        return precision, recall, thresholds

    def compute_metrics_at_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Compute various metrics at a specific threshold.

        Parameters:
        threshold (float): Classification threshold

        Returns:
        Dict containing various metric scores
        """
        predictions_binary = (self.predictions >= threshold).astype(int)

        # True/False Positives/Negatives
        tp = np.sum((predictions_binary == 1) & (self.true_labels == 1))
        fp = np.sum((predictions_binary == 1) & (self.true_labels == 0))
        tn = np.sum((predictions_binary == 0) & (self.true_labels == 0))
        fn = np.sum((predictions_binary == 0) & (self.true_labels == 1))

        # Calculate metrics
        accuracy = (tp + tn) / len(self.true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

    def plot_roc_curve(self, ax=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot ROC curve.

        Parameters:
        ax: matplotlib axis object (optional)

        Returns:
        Tuple of arrays (fpr, tpr) used for plotting
        """
        fpr, tpr, _ = self.compute_roc_curve()
        auc_score = self.compute_auc()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        ax.grid(True)

        return fpr, tpr

    def plot_precision_recall_curve(self, ax=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot precision-recall curve.

        Parameters:
        ax: matplotlib axis object (optional)

        Returns:
        Tuple of arrays (precision, recall) used for plotting
        """
        precision, recall, _ = self.compute_precision_recall_curve()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True)

        return precision, recall

    def find_optimal_threshold(self, method='f1') -> float:
        """
        Find optimal classification threshold based on specified metric.

        Parameters:
        method (str): Metric to optimize ('f1', 'accuracy', 'balanced')

        Returns:
        float: Optimal threshold value
        """
        _, _, thresholds = self.compute_roc_curve()
        best_threshold = 0
        best_score = 0

        for threshold in thresholds:
            metrics = self.compute_metrics_at_threshold(threshold)

            if method == 'f1':
                score = metrics['f1_score']
            elif method == 'accuracy':
                score = metrics['accuracy']
            elif method == 'balanced':
                score = (metrics['recall'] + metrics['specificity']) / 2
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold


def main():
    """Example usage of the recommender evaluator."""

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # True labels (0 or 1)
    true_labels = np.random.binomial(1, 0.3, n_samples)

    # Predicted probabilities (with some noise)
    predictions = true_labels + np.random.normal(0, 0.3, n_samples)
    predictions = np.clip(predictions, 0, 1)

    # Initialize evaluator
    evaluator = RecommenderEvaluator()
    evaluator.load_data(predictions, true_labels)

    # Create subplots for ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot ROC curve
    evaluator.plot_roc_curve(ax1)

    # Plot Precision-Recall curve
    evaluator.plot_precision_recall_curve(ax2)

    # Find optimal threshold
    optimal_threshold = evaluator.find_optimal_threshold(method='f1')
    metrics = evaluator.compute_metrics_at_threshold(optimal_threshold)

    print(f"\nOptimal threshold (F1): {optimal_threshold:.3f}")
    print("\nMetrics at optimal threshold:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()