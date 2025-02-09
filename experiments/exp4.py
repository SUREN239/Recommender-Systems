import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
from nltk.stem import PorterStemmer


class ContentBasedRecommender:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.scaler = MinMaxScaler()
        self.feature_names = None

    def preprocess_text(self, text):
        """Preprocess text data"""
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Split into words and stem
        words = text.split()
        words = [self.stemmer.stem(word) for word in words]

        return ' '.join(words)

    def fit(self, items_df):
        """Fit the recommender system"""
        try:
            self.items_df = items_df.copy()

            print("Preprocessing text descriptions...")
            self.items_df['processed_description'] = self.items_df['description'].apply(
                self.preprocess_text
            )

            print("Creating TF-IDF matrix...")
            self.tfidf_matrix = self.vectorizer.fit_transform(
                self.items_df['processed_description']
            )

            # Store feature names after fitting
            self.feature_names = self.vectorizer.get_feature_names_out()

            print("Calculating similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

            # Process numerical features
            numerical_features = ['rating', 'popularity', 'year']
            num_features = [col for col in numerical_features if col in self.items_df.columns]

            if num_features:
                print("Processing numerical features...")
                numerical_matrix = self.items_df[num_features].values
                numerical_matrix = self.scaler.fit_transform(numerical_matrix)
                self.feature_matrix = np.hstack([
                    self.tfidf_matrix.toarray(),
                    numerical_matrix
                ])
            else:
                self.feature_matrix = self.tfidf_matrix.toarray()

            print("Recommender system fitted successfully!")
            return True

        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            return False

    def get_similar_items(self, item_id, n=5):
        """Get n most similar items"""
        try:
            if item_id not in self.items_df['item_id'].values:
                raise ValueError(f"Item ID {item_id} not found in the dataset")

            idx = self.items_df[self.items_df['item_id'] == item_id].index[0]
            similarity_scores = self.similarity_matrix[idx]

            # Get top N similar items (excluding self)
            similar_indices = similarity_scores.argsort()[::-1][1:n + 1]
            similar_items = self.items_df.iloc[similar_indices]
            scores = similarity_scores[similar_indices]

            return pd.DataFrame({
                'item_id': similar_items['item_id'],
                'title': similar_items['title'],
                'similarity_score': scores.round(3)
            })

        except Exception as e:
            print(f"Error getting similar items: {str(e)}")
            return pd.DataFrame()

    def get_recommendations(self, user_profile, n=5):
        """Get recommendations based on user profile"""
        try:
            scores = np.zeros(len(self.items_df))

            for feature, weight in user_profile.items():
                feature_stemmed = self.stemmer.stem(feature)
                # Check if stemmed feature exists in vocabulary
                feature_matches = [
                    i for i, name in enumerate(self.feature_names)
                    if feature_stemmed in name
                ]

                for idx in feature_matches:
                    scores += weight * self.feature_matrix[:, idx]

            # Get top N recommendations
            top_indices = scores.argsort()[::-1][:n]
            recommendations = self.items_df.iloc[top_indices]

            return pd.DataFrame({
                'item_id': recommendations['item_id'],
                'title': recommendations['title'],
                'score': scores[top_indices].round(3)
            })

        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return pd.DataFrame()


def create_sample_data():
    """Create sample movie dataset"""
    movies_data = {
        'item_id': range(1, 11),
        'title': [
            'The Matrix',
            'Inception',
            'The Dark Knight',
            'Pulp Fiction',
            'Forrest Gump',
            'The Shawshank Redemption',
            'The Godfather',
            'Fight Club',
            'Interstellar',
            'Gladiator'
        ],
        'description': [
            'A computer programmer discovers a dystopian world inside the Matrix with sci-fi action',
            'A thief who enters the dreams of others to steal secrets in this sci-fi thriller',
            'Batman fights against the criminal mastermind known as the Joker in this action movie',
            'Various interconnected stories of criminals in Los Angeles crime drama',
            'A slow-witted but kind-hearted man witnesses historic events in this drama',
            'A banker is sentenced to life in Shawshank State Penitentiary drama',
            'The aging patriarch of an organized crime dynasty transfers control crime drama',
            'An insomniac office worker and a soap maker form an underground fight club action',
            'A team of explorers travel through a wormhole in space sci-fi adventure',
            'A former Roman General seeks revenge against the corrupt emperor action drama'
        ],
        'year': [1999, 2010, 2008, 1994, 1994, 1994, 1972, 1999, 2014, 2000],
        'rating': [8.7, 8.8, 9.0, 8.9, 8.8, 9.3, 9.2, 8.8, 8.6, 8.5],
        'popularity': [85, 90, 95, 88, 92, 96, 94, 87, 89, 86]
    }
    return pd.DataFrame(movies_data)


def test_recommender():
    """Test the recommender system"""
    try:
        # Create and fit recommender
        print("Creating sample movie data...")
        movies_df = create_sample_data()

        print("\nInitializing recommender system...")
        recommender = ContentBasedRecommender()

        print("Fitting recommender system...")
        if not recommender.fit(movies_df):
            print("Failed to fit recommender system")
            return

        # Test similar items
        print("\nTesting similar items recommendation...")
        test_item_id = 1  # The Matrix
        similar_items = recommender.get_similar_items(test_item_id, n=3)
        if not similar_items.empty:
            print(f"\nMovies similar to 'The Matrix':")
            print(similar_items.to_string(index=False))

        # Test user profile recommendations
        print("\nTesting user profile recommendations...")
        user_profile = {
            'action': 0.8,
            'sci-fi': 0.9,
            'drama': 0.3,
            'crime': 0.4
        }
        recommendations = recommender.get_recommendations(user_profile, n=3)
        if not recommendations.empty:
            print("\nRecommendations based on user profile:")
            print(recommendations.to_string(index=False))

        return recommender, movies_df

    except Exception as e:
        print(f"Error in test_recommender: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Run the complete test
    recommender, movies_df = test_recommender()

    if recommender is not None:
        print("\nInteractive testing mode:")
        print("You can now interact with the 'recommender' object:")
        print("\nExample commands:")
        print("1. Get similar items:")
        print("   similar_items = recommender.get_similar_items(item_id=1, n=3)")
        print("\n2. Get recommendations for a user profile:")
        print("   user_profile = {'action': 0.8, 'sci-fi': 0.9}")
        print("   recommendations = recommender.get_recommendations(user_profile, n=3)")