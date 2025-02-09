import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple


class PageRankRecommender:
    def __init__(self, damping_factor: float = 0.85, epsilon: float = 1e-8,
                 max_iterations: int = 100):
        """
        Initialize PageRank recommender.

        Parameters:
        damping_factor (float): Damping factor (usually 0.85)
        epsilon (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
        """
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        self.transition_matrix = None
        self.item_indices = None
        self.pagerank_scores = None

    def build_graph(self, interactions_df: pd.DataFrame,
                    weight_col: str = None):
        """
        Build item-item graph from user interactions.

        Parameters:
        interactions_df: DataFrame with columns [user_id, item_id]
        weight_col: Optional column name for interaction weights
        """
        # Create user-item matrix
        if weight_col:
            values = interactions_df[weight_col].values
        else:
            values = np.ones(len(interactions_df))

        user_item = csr_matrix(
            (values,
             (interactions_df.user_id, interactions_df.item_id))
        )

        # Create item-item co-occurrence matrix
        item_item = user_item.T @ user_item

        # Normalize transitions
        row_sums = np.array(item_item.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        self.transition_matrix = csr_matrix(
            item_item / row_sums[:, np.newaxis]
        )

        self.item_indices = {
            i: idx for idx, i in enumerate(
                np.unique(interactions_df.item_id)
            )
        }

    def compute_pagerank(self):
        """
        Compute PageRank scores for items.

        Returns:
        Dict mapping item IDs to PageRank scores
        """
        n_items = self.transition_matrix.shape[0]

        # Initialize PageRank scores
        pagerank = np.ones(n_items) / n_items

        # Power iteration method
        for iteration in range(self.max_iterations):
            prev_pagerank = pagerank.copy()

            # Update PageRank scores
            pagerank = (1 - self.damping_factor) / n_items + \
                       self.damping_factor * \
                       self.transition_matrix.T @ prev_pagerank

            # Check convergence
            if np.sum(np.abs(pagerank - prev_pagerank)) < self.epsilon:
                break

        # Store and return results
        self.pagerank_scores = {
            item_id: pagerank[idx]
            for item_id, idx in self.item_indices.items()
        }
        return self.pagerank_scores

    def get_recommendations(self, user_id: int, interactions_df: pd.DataFrame,
                            n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get personalized recommendations for a user.

        Parameters:
        user_id: User ID
        interactions_df: DataFrame with user interactions
        n_recommendations: Number of recommendations to return

        Returns:
        List of (item_id, score) tuples
        """
        if self.pagerank_scores is None:
            self.compute_pagerank()

        # Get items user hasn't interacted with
        user_items = set(
            interactions_df[interactions_df.user_id == user_id].item_id
        )
        candidate_items = set(self.pagerank_scores.keys()) - user_items

        # Sort candidates by PageRank score
        recommendations = sorted(
            [(item_id, self.pagerank_scores[item_id])
             for item_id in candidate_items],
            key=lambda x: x[1],
            reverse=True
        )

        return recommendations[:n_recommendations]

    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar items based on transition probabilities.

        Parameters:
        item_id: Item ID
        n_similar: Number of similar items to return

        Returns:
        List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_indices:
            return []

        item_idx = self.item_indices[item_id]
        transition_scores = self.transition_matrix[item_idx].toarray().flatten()

        # Get top similar items
        similar_indices = np.argsort(transition_scores)[::-1][:n_similar]

        # Map back to item IDs
        reverse_mapping = {
            idx: item_id for item_id, idx in self.item_indices.items()
        }

        return [
            (reverse_mapping[idx], transition_scores[idx])
            for idx in similar_indices
            if transition_scores[idx] > 0
        ]


def main():
    """Example usage of PageRank recommender."""
    # Generate sample interaction data
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_interactions = 1000

    interactions_data = {
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'timestamp': np.random.randint(1, 100, n_interactions)  # Example weight
    }

    interactions_df = pd.DataFrame(interactions_data)

    # Initialize and train recommender
    recommender = PageRankRecommender(damping_factor=0.85)
    recommender.build_graph(interactions_df, weight_col='timestamp')

    # Compute PageRank scores
    pagerank_scores = recommender.compute_pagerank()

    print("Top 5 items by PageRank score:")
    top_items = sorted(
        pagerank_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for item_id, score in top_items:
        print(f"Item {item_id}: {score:.6f}")

    # Get recommendations for a user
    test_user = 0
    recommendations = recommender.get_recommendations(
        test_user,
        interactions_df,
        n_recommendations=5
    )

    print("\nTop 5 recommendations for test user:")
    for item_id, score in recommendations:
        print(f"Item {item_id}: {score:.6f}")

    # Get similar items
    test_item = 0
    similar_items = recommender.get_similar_items(test_item, n_similar=5)

    print(f"\nTop 5 similar items to item {test_item}:")
    for item_id, similarity in similar_items:
        print(f"Item {item_id}: {similarity:.6f}")


if __name__ == "__main__":
    main()