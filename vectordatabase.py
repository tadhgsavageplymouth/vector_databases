import numpy as np
from scipy.spatial.distance import cosine

# Step 1: A simple vector-based travel recommender system
class SimpleTravelRecommender:
    def __init__(self):
        """
        Initialises the list to store destination vectors and their corresponding metadata.
        """
        self.destinations = []
        self.metadata = []

    def add_destination(self, vector, metadata):
        """
        Adds a destination with its associated feature vector and metadata.
        
        Parameters:
        - vector: A list of numerical features representing the destination.
        - metadata: A string containing descriptive information about the destination.
        """
        self.destinations.append(vector)
        self.metadata.append(metadata)

    def recommend(self, user_vector, top_k=3):
        """
        Recommends destinations based on their similarity to the user's preferences.

        Parameters:
        - user_vector: A vector representing the user's preferences.
        - top_k: The number of top recommendations to return (default is 3).

        Returns:
        - A list of tuples containing destination metadata and their similarity scores, sorted in descending order of similarity.
        """
        similarities = []
        for i, destination_vector in enumerate(self.destinations):
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(user_vector, destination_vector)
            similarities.append((self.metadata[i], similarity))
        
        # Sort destinations by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

# Step 2: Generate example data for travel destinations
def create_example_data(recommender):
    """
    Populates the recommender with example travel destinations.

    Each destination is represented by a vector with the following format:
    [temperature, humidity, daily_cost, beach, adventure, cultural]
    """
    recommender.add_destination([30, 70, 150, 1, 0, 0], "Sunny Beach in Thailand")
    recommender.add_destination([20, 50, 200, 0, 1, 0], "Mountain Hiking in Switzerland")
    recommender.add_destination([25, 60, 120, 0, 0, 1], "Cultural Tour in Italy")
    recommender.add_destination([15, 40, 80, 0, 1, 1], "Adventure and Culture in Nepal")
    recommender.add_destination([28, 80, 250, 1, 1, 0], "Beach and Adventure in Brazil")

# Step 3: Create a vector for the user's preferences
def get_user_preferences():
    """
    Generates a feature vector representing the user's travel preferences.

    Format:
    [preferred_temperature, preferred_humidity, max_daily_cost, beach_preference, adventure_preference, cultural_preference]

    Returns:
    - A list representing the user's preferences.
    """
    # Example user preferences: warm weather, medium humidity, low cost, prefers beach and cultural experiences
    return [27, 60, 180, 1, 0, 1]

# Step 4: Test the travel recommender system
if __name__ == "__main__":
    # Initialise the travel recommender
    recommender = SimpleTravelRecommender()

    # Populate the recommender with example destinations
    create_example_data(recommender)

    # Generate the user's preference vector
    user_preferences = get_user_preferences()

    # Obtain the top 3 recommended destinations
    recommendations = recommender.recommend(user_preferences, top_k=3)

    # Display the recommendations
    print("Top 3 Recommended Destinations:")
    for destination, similarity in recommendations:
        print(f"Destination: {destination}, Similarity: {similarity:.4f}")
