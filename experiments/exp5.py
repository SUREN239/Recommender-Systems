import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class CollaborativeFiltering:
    def __init__(self, method='user'):
        """
        Initialize the collaborative filtering model.

        Parameters:
        method (str): 'user' for user-based or 'item' for item-based collaborative filtering
        """
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None

    def create_user_item_matrix(self, ratings_df):
        """
        Create user-item matrix from ratings dataframe.
        Handle duplicate entries by taking the mean rating.

        Parameters:
        ratings_df (pd.DataFrame): Dataframe with columns ['user_id', 'item_id', 'rating']
        """
        # Aggregate duplicate ratings by taking the mean
        ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

        self.user_item_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

    def compute_similarity(self):
        """Compute similarity matrix based on chosen method."""
        if self.method == 'user':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        else:  # item-based
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)

    def get_top_n_similar(self, index, n=5):
        """
        Get top N similar users/items for given index.

        Parameters:
        index (int): Index of user/item
        n (int): Number of similar users/items to return

        Returns:
        list: Top N similar users/items with similarity scores
        """
        sim_scores = self.similarity_matrix[index]
        top_indices = np.argsort(sim_scores)[::-1][1:n + 1]  # Exclude self
        return list(zip(top_indices, sim_scores[top_indices]))

    def predict_rating(self, user_id, item_id, k=5):
        """
        Predict rating for a user-item pair.

        Parameters:
        user_id (int): User ID
        item_id (int): Item ID
        k (int): Number of neighbors to consider

        Returns:
        float: Predicted rating
        """
        if self.method == 'user':
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            similar_users = self.get_top_n_similar(user_idx, n=k)

            item_col = self.user_item_matrix.columns.get_loc(item_id)
            weighted_sum = 0
            similarity_sum = 0

            for sim_user_idx, similarity in similar_users:
                rating = self.user_item_matrix.iloc[sim_user_idx, item_col]
                if rating > 0:  # Only consider actual ratings
                    weighted_sum += similarity * rating
                    similarity_sum += similarity

            if similarity_sum == 0:
                return self.user_item_matrix.iloc[:, item_col].mean()

            return weighted_sum / similarity_sum

        else:  # item-based
            item_idx = self.user_item_matrix.columns.get_loc(item_id)
            similar_items = self.get_top_n_similar(item_idx, n=k)

            user_row = self.user_item_matrix.index.get_loc(user_id)
            weighted_sum = 0
            similarity_sum = 0

            for sim_item_idx, similarity in similar_items:
                rating = self.user_item_matrix.iloc[user_row, sim_item_idx]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity

            if similarity_sum == 0:
                return self.user_item_matrix.iloc[user_row, :].mean()

            return weighted_sum / similarity_sum

    def evaluate(self, test_data):
        """
        Evaluate the model using test data.

        Parameters:
        test_data (pd.DataFrame): Test data with columns ['user_id', 'item_id', 'rating']

        Returns:
        float: Root Mean Squared Error (RMSE)
        """
        # Handle duplicates in test data as well
        test_data = test_data.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

        predictions = []
        actual_ratings = []

        for _, row in test_data.iterrows():
            try:
                pred = self.predict_rating(row['user_id'], row['item_id'])
                predictions.append(pred)
                actual_ratings.append(row['rating'])
            except:
                continue

        return np.sqrt(mean_squared_error(actual_ratings, predictions))


def main():
    # Create sample data (now with potential duplicates)
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_ratings = 1000

    # Generate data with some duplicate user-item pairs
    data = {
        'user_id': np.random.randint(0, n_users, n_ratings),
        'item_id': np.random.randint(0, n_items, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings)
    }

    df = pd.DataFrame(data)

    # Split data into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # User-based collaborative filtering
    user_cf = CollaborativeFiltering(method='user')
    user_cf.create_user_item_matrix(train_data)
    user_cf.compute_similarity()

    # Item-based collaborative filtering
    item_cf = CollaborativeFiltering(method='item')
    item_cf.create_user_item_matrix(train_data)
    item_cf.compute_similarity()

    # Evaluate both methods
    user_rmse = user_cf.evaluate(test_data)
    item_rmse = item_cf.evaluate(test_data)

    print(f"User-based CF RMSE: {user_rmse:.4f}")
    print(f"Item-based CF RMSE: {item_rmse:.4f}")

    # Example predictions
    test_user = test_data['user_id'].iloc[0]
    test_item = test_data['item_id'].iloc[0]

    user_pred = user_cf.predict_rating(test_user, test_item)
    item_pred = item_cf.predict_rating(test_user, test_item)

    print(f"\nPredictions for user {test_user} and item {test_item}:")
    print(f"User-based prediction: {user_pred:.2f}")
    print(f"Item-based prediction: {item_pred:.2f}")


if __name__ == "__main__":
    main()