import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserProfile:
    """
    Represents a user's preferences and learning profile
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences = defaultdict(float)  # Feature weights
        self.rated_items = {}  # Items rated by user
        self.feature_vector = None  # Learned feature vector
        self.rating_history = []  # Keep track of all ratings

    def update_preferences(self, item_features: Dict, rating: float, learning_rate: float = 0.1) -> None:
        """
        Update user preferences based on new item rating
        """
        try:
            # Normalize rating to [-1, 1] range
            normalized_rating = (rating - 3) / 2

            # Update feature weights using gradient descent
            for feature, value in item_features.items():
                current_weight = self.preferences[feature]
                error = normalized_rating - current_weight * value
                self.preferences[feature] += learning_rate * error * value

            # Store rated item
            item_id = len(self.rated_items)
            self.rated_items[item_id] = {
                'features': item_features,
                'rating': rating,
                'timestamp': pd.Timestamp.now()
            }

            # Add to rating history
            self.rating_history.append({
                'rating': rating,
                'timestamp': pd.Timestamp.now()
            })

        except Exception as e:
            logger.error(f"Error updating preferences for user {self.user_id}: {str(e)}")
            raise

    def compute_feature_vector(self) -> Optional[np.ndarray]:
        """
        Compute user feature vector based on rated items
        """
        if not self.rated_items:
            return None

        try:
            # Get all unique features
            all_features = set()
            for item in self.rated_items.values():
                all_features.update(item['features'].keys())

            # Create feature vector
            self.feature_vector = np.zeros(len(all_features))
            feature_to_idx = {f: i for i, f in enumerate(all_features)}

            # Compute weighted average of features
            total_weight = 0
            for item in self.rated_items.values():
                rating_weight = (item['rating'] - 3) / 2
                total_weight += abs(rating_weight)

                for feature, value in item['features'].items():
                    idx = feature_to_idx[feature]
                    self.feature_vector[idx] += rating_weight * value

            if total_weight > 0:
                self.feature_vector /= total_weight

            return self.feature_vector

        except Exception as e:
            logger.error(f"Error computing feature vector for user {self.user_id}: {str(e)}")
            return None

    def get_rating_statistics(self) -> Dict:
        """
        Get statistical summary of user's rating behavior
        """
        if not self.rating_history:
            return {}

        ratings = [r['rating'] for r in self.rating_history]
        return {
            'mean_rating': np.mean(ratings),
            'std_rating': np.std(ratings),
            'num_ratings': len(ratings),
            'rating_range': (min(ratings), max(ratings))
        }


class ContentBasedRecommender:
    """
    Content-based recommender system with user profile learning
    """

    def __init__(self):
        self.items = {}
        self.users = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.metadata = {}

    def add_item(self, item_id: int, content: str, metadata: Dict = None) -> None:
        """
        Add new item to the recommender system
        """
        try:
            # Initialize vectorizer with first item
            if len(self.items) == 0:
                self.vectorizer.fit([content])

            # Extract features using TF-IDF
            features = self.vectorizer.transform([content]).toarray()[0]

            self.items[item_id] = {
                'content': content,
                'features': features,
                'metadata': metadata or {},
                'added_date': pd.Timestamp.now()
            }

            logger.info(f"Added item {item_id} to the system")

        except Exception as e:
            logger.error(f"Error adding item {item_id}: {str(e)}")
            raise

    def add_user(self, user_id: str) -> None:
        """
        Add new user to the system
        """
        try:
            self.users[user_id] = UserProfile(user_id)
            logger.info(f"Added user {user_id} to the system")
        except Exception as e:
            logger.error(f"Error adding user {user_id}: {str(e)}")
            raise

    def record_rating(self, user_id: str, item_id: int, rating: float) -> None:
        """
        Record user rating and update their profile
        """
        try:
            # Add user if not exists
            if user_id not in self.users:
                self.add_user(user_id)

            # Validate item exists
            if item_id not in self.items:
                raise ValueError(f"Item {item_id} not found in the system")

            # Convert item features to dictionary
            item_features = {
                f'feature_{i}': v
                for i, v in enumerate(self.items[item_id]['features'])
            }

            # Update user preferences
            self.users[user_id].update_preferences(item_features, rating)
            logger.info(f"Recorded rating {rating} from user {user_id} for item {item_id}")

        except Exception as e:
            logger.error(f"Error recording rating: {str(e)}")
            raise

    def get_recommendations(self, user_id: str, n: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for user
        """
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")

            user = self.users[user_id]
            user.compute_feature_vector()

            if user.feature_vector is None:
                return []

            # Compute similarity between user profile and all items
            similarities = {}
            for item_id, item in self.items.items():
                if item_id not in user.rated_items:
                    sim = cosine_similarity(
                        user.feature_vector.reshape(1, -1),
                        item['features'].reshape(1, -1)
                    )[0][0]
                    similarities[item_id] = sim

            # Sort by similarity and return top N
            recommended_items = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]

            return recommended_items

        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
            return []

    def evaluate_user(self, user_id: str) -> Dict:
        """
        Evaluate recommendations for a specific user
        """
        try:
            user = self.users.get(user_id)
            if not user:
                return {}

            stats = user.get_rating_statistics()
            recommendations = self.get_recommendations(user_id)

            return {
                'user_stats': stats,
                'num_recommendations': len(recommendations),
                'top_recommendations': recommendations[:3]
            }

        except Exception as e:
            logger.error(f"Error evaluating user {user_id}: {str(e)}")
            return {}


def run_test_scenario():
    """
    Run comprehensive test scenario
    """
    logger.info("Starting test scenario")
    recommender = ContentBasedRecommender()

    # Sample movies database
    movies = {
        1: "Fast-paced action movie with explosions and car chases",
        2: "Romantic comedy about relationships and dating in New York",
        3: "Nature documentary exploring wildlife in Africa",
        4: "Science fiction thriller with space exploration and aliens",
        5: "Drama about family relationships and personal growth",
        6: "Action-packed superhero movie with special effects",
        7: "Comedy about college life and friendship",
        8: "Historical drama set in World War II",
        9: "Animated family movie with musical numbers",
        10: "Horror movie about haunted house and supernatural events"
    }

    # Add movies
    logger.info("Adding sample movies")
    for movie_id, description in movies.items():
        recommender.add_item(
            movie_id,
            description,
            {"type": "movie", "genre": description.split()[0].lower()}
        )

    # Create test users
    test_users = {
        "action_fan": {
            1: 5, 4: 4, 6: 5, 2: 2, 5: 2  # Likes action and sci-fi
        },
        "drama_lover": {
            5: 5, 8: 5, 2: 4, 1: 2, 6: 2  # Likes drama and historical
        },
        "family_viewer": {
            9: 5, 3: 4, 5: 4, 10: 1, 1: 2  # Likes animation and family
        }
    }

    # Record ratings
    logger.info("Recording user ratings")
    for user_id, ratings in test_users.items():
        recommender.add_user(user_id)
        for movie_id, rating in ratings.items():
            recommender.record_rating(user_id, movie_id, rating)

    # Generate and display recommendations
    logger.info("Generating recommendations")
    results = {}
    for user_id in test_users.keys():
        recommendations = recommender.get_recommendations(user_id, n=3)
        results[user_id] = {
            'recommendations': [
                {
                    'movie_id': movie_id,
                    'description': movies[movie_id],
                    'similarity': similarity
                }
                for movie_id, similarity in recommendations
            ],
            'evaluation': recommender.evaluate_user(user_id)
        }

    return recommender, results


def plot_user_ratings(recommender, user_id):
    """
    Plot user rating history
    """
    user = recommender.users.get(user_id)
    if not user or not user.rating_history:
        return

    ratings = [r['rating'] for r in user.rating_history]
    timestamps = [r['timestamp'] for r in user.rating_history]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ratings, 'bo-')
    plt.title(f'Rating History for User {user_id}')
    plt.xlabel('Time')
    plt.ylabel('Rating')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run the complete test scenario
    recommender, results = run_test_scenario()

    # Display results
    print("\nTest Results:")
    print("=============")

    for user_id, user_results in results.items():
        print(f"\nUser: {user_id}")
        print("Recommendations:")
        for i, rec in enumerate(user_results['recommendations'], 1):
            print(f"{i}. Movie {rec['movie_id']}: {rec['description']}")
            print(f"   Similarity Score: {rec['similarity']:.3f}")

        print("\nUser Statistics:")
        stats = user_results['evaluation']['user_stats']
        print(f"Average Rating: {stats.get('mean_rating', 'N/A'):.2f}")
        print(f"Number of Ratings: {stats.get('num_ratings', 0)}")

        # Plot rating history
        plot_user_ratings(recommender, user_id)

    print("\nInteractive Testing Mode:")
    print("------------------------")
    print("You can now interact with the 'recommender' object:")
    print("1. Add new movie: recommender.add_item(11, 'Movie description')")
    print("2. Add new rating: recommender.record_rating('user_id', movie_id, rating)")
    print("3. Get recommendations: recommender.get_recommendations('user_id')")
    print("4. Evaluate user: recommender.evaluate_user('user_id')")