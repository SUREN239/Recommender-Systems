import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, List


class BPRMatrixFactorization:
    def __init__(self, n_factors: int = 20, learning_rate: float = 0.01,
                 regularization: float = 0.01, n_iterations: int = 100):
        """
        Initialize BPR with Matrix Factorization model.

        Parameters:
        n_factors (int): Number of latent factors
        learning_rate (float): Learning rate for SGD
        regularization (float): Regularization term
        n_iterations (int): Number of training iterations
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_iterations = n_iterations

        self.user_factors = None
        self.item_factors = None
        self.n_users = None
        self.n_items = None

    def _init_matrices(self, n_users: int, n_items: int):
        """Initialize user and item factor matrices."""
        self.user_factors = np.random.normal(
            scale=1. / self.n_factors,
            size=(n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=1. / self.n_factors,
            size=(n_items, self.n_factors)
        )

    def create_training_set(self, ratings_df: pd.DataFrame) -> List[Tuple]:
        """
        Create training set of (user, positive_item, negative_item) triplets.

        Parameters:
        ratings_df: DataFrame with columns [user_id, item_id, rating]

        Returns:
        List of training triplets
        """
        # Create user-item interaction matrix
        user_item = csr_matrix(
            (np.ones_like(ratings_df.rating),
             (ratings_df.user_id, ratings_df.item_id))
        )

        self.n_users = user_item.shape[0]
        self.n_items = user_item.shape[1]

        if self.user_factors is None:
            self._init_matrices(self.n_users, self.n_items)

        # Create training triplets
        triplets = []
        for user in range(self.n_users):
            # Get positive items for user
            pos_items = user_item[user].indices

            if len(pos_items) == 0:
                continue

            # Sample negative items
            neg_items = np.setdiff1d(
                np.arange(self.n_items),
                pos_items,
                assume_unique=True
            )

            # Create triplets
            for pos_item in pos_items:
                neg_item = np.random.choice(neg_items)
                triplets.append((user, pos_item, neg_item))

        return triplets

    def _update_factors(self, u: int, i: int, j: int):
        """
        Update user and item factors using SGD.

        Parameters:
        u: user index
        i: positive item index
        j: negative item index
        """
        # Compute predicted preference difference
        x_ui = np.dot(self.user_factors[u], self.item_factors[i])
        x_uj = np.dot(self.user_factors[u], self.item_factors[j])
        x_uij = x_ui - x_uj

        # Compute sigmoid loss
        sigmoid = 1.0 / (1.0 + np.exp(x_uij))

        # Compute gradients
        grad_u = sigmoid * (self.item_factors[i] - self.item_factors[j]) - \
                 self.regularization * self.user_factors[u]
        grad_i = sigmoid * self.user_factors[u] - \
                 self.regularization * self.item_factors[i]
        grad_j = -sigmoid * self.user_factors[u] - \
                 self.regularization * self.item_factors[j]

        # Update factors
        self.user_factors[u] += self.learning_rate * grad_u
        self.item_factors[i] += self.learning_rate * grad_i
        self.item_factors[j] += self.learning_rate * grad_j

    def fit(self, ratings_df: pd.DataFrame):
        """
        Train the BPR model.

        Parameters:
        ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        # Create training triplets
        triplets = self.create_training_set(ratings_df)

        # Training loop
        for iteration in range(self.n_iterations):
            np.random.shuffle(triplets)

            # Update factors for each triplet
            for u, i, j in triplets:
                self._update_factors(u, i, j)

            if (iteration + 1) % 10 == 0:
                print(f"Completed iteration {iteration + 1}/{self.n_iterations}")

    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Predict rankings for given user-item pairs.

        Parameters:
        user_id: User ID
        item_ids: List of item IDs

        Returns:
        Array of predicted rankings
        """
        return np.dot(self.user_factors[user_id], self.item_factors[item_ids].T)


def main():
    """Example usage of BPR-MF."""
    # Generate sample data
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000

    ratings_data = {
        'user_id': np.random.randint(0, n_users, n_ratings),
        'item_id': np.random.randint(0, n_items, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings)
    }

    ratings_df = pd.DataFrame(ratings_data)

    # Initialize and train model
    bpr_mf = BPRMatrixFactorization(n_factors=20, n_iterations=50)
    bpr_mf.fit(ratings_df)

    # Make predictions for a user
    test_user = 0
    test_items = list(range(5))
    predictions = bpr_mf.predict(test_user, test_items)

    print("\nPredictions for test user:")
    for item, pred in zip(test_items, predictions):
        print(f"Item {item}: {pred:.3f}")


if __name__ == "__main__":
    main()