import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ConstraintBasedRecommender:
    def __init__(self):
        """Initialize the constraint-based recommender system."""
        self.products = None
        self.constraints = {}
        self.weights = {}

    def load_products(self, products_df: pd.DataFrame):
        """
        Load product catalog with attributes.

        Parameters:
        products_df (pd.DataFrame): DataFrame containing product information
        """
        self.products = products_df.copy()

    def add_constraint(self, attribute: str, constraint_type: str,
                       value: Any, weight: float = 1.0):
        """
        Add a constraint for product filtering.

        Parameters:
        attribute (str): Product attribute to constrain
        constraint_type (str): Type of constraint ('exact', 'min', 'max', 'range', 'in')
        value (Any): Constraint value
        weight (float): Importance weight of the constraint
        """
        self.constraints[attribute] = {
            'type': constraint_type,
            'value': value
        }
        self.weights[attribute] = weight

    def clear_constraints(self):
        """Remove all existing constraints."""
        self.constraints.clear()
        self.weights.clear()

    def _validate_constraint(self, product_value: Any,
                             constraint: Dict[str, Any]) -> float:
        """
        Validate how well a product value satisfies a constraint.

        Returns:
        float: Score between 0 and 1 indicating constraint satisfaction
        """
        if pd.isna(product_value):
            return 0.0

        constraint_type = constraint['type']
        constraint_value = constraint['value']

        if constraint_type == 'exact':
            return float(product_value == constraint_value)

        elif constraint_type == 'min':
            try:
                return float(product_value >= constraint_value)
            except:
                return 0.0

        elif constraint_type == 'max':
            try:
                return float(product_value <= constraint_value)
            except:
                return 0.0

        elif constraint_type == 'range':
            try:
                min_val, max_val = constraint_value
                return float(min_val <= product_value <= max_val)
            except:
                return 0.0

        elif constraint_type == 'in':
            return float(product_value in constraint_value)

        return 0.0

    def _calculate_product_score(self, product: pd.Series) -> float:
        """
        Calculate overall score for a product based on all constraints.

        Returns:
        float: Weighted average score for the product
        """
        if len(self.constraints) == 0:
            return 1.0

        total_weight = sum(self.weights.values())
        weighted_score = 0

        for attribute, constraint in self.constraints.items():
            if attribute in product:
                score = self._validate_constraint(
                    product[attribute], constraint)
                weighted_score += score * self.weights[attribute]

        return weighted_score / total_weight

    def get_recommendations(self, top_n: Optional[int] = None,
                            min_score: float = 0.0) -> pd.DataFrame:
        """
        Get product recommendations based on current constraints.

        Parameters:
        top_n (int, optional): Number of top recommendations to return
        min_score (float): Minimum score threshold for recommendations

        Returns:
        pd.DataFrame: Recommended products with scores
        """
        if self.products is None:
            raise ValueError("No products loaded. Call load_products first.")

        # Calculate scores for all products
        scores = self.products.apply(self._calculate_product_score, axis=1)

        # Filter by minimum score
        recommendations = self.products.copy()
        recommendations['score'] = scores
        recommendations = recommendations[recommendations['score'] >= min_score]

        # Sort by score
        recommendations = recommendations.sort_values('score', ascending=False)

        if top_n is not None:
            recommendations = recommendations.head(top_n)

        return recommendations


def main():
    """Example usage of the constraint-based recommender."""

    # Create sample product catalog
    products_data = {
        'product_id': range(1, 11),
        'name': [f'Product {i}' for i in range(1, 11)],
        'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing',
                     'Books', 'Books', 'Electronics', 'Clothing', 'Books',
                     'Electronics'],
        'price': [999.99, 499.99, 59.99, 89.99, 19.99, 29.99, 799.99,
                  129.99, 24.99, 1499.99],
        'rating': [4.5, 4.2, 4.8, 4.0, 4.7, 4.3, 4.6, 4.1, 4.4, 4.9],
        'in_stock': [True, True, False, True, True, True, False, True,
                     True, True]
    }

    products_df = pd.DataFrame(products_data)

    # Initialize recommender
    recommender = ConstraintBasedRecommender()
    recommender.load_products(products_df)

    # Add constraints
    recommender.add_constraint('category', 'exact', 'Electronics', weight=1.0)
    recommender.add_constraint('price', 'max', 1000.0, weight=0.8)
    recommender.add_constraint('rating', 'min', 4.0, weight=0.6)
    recommender.add_constraint('in_stock', 'exact', True, weight=0.9)

    # Get recommendations
    recommendations = recommender.get_recommendations(top_n=5, min_score=0.5)

    print("Top Recommendations:")
    print(recommendations[['name', 'category', 'price', 'rating', 'score']])


if __name__ == "__main__":
    main()