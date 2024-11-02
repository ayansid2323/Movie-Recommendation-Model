import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process


# Define a custom tokenizer function
def custom_tokenizer(text):
    """
    Tokenizes the input text into a list of words, excluding any words that contain digits.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of tokens (words) from the input text, excluding any words that contain digits.
    """
    tokens = re.findall(r'\b\w+\b', text)
    return [token for token in tokens if not any(char.isdigit() for char in token)]

def similarity_matrix(data:pd.DataFrame) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for the movie overviews.

    Returns:
        numpy.ndarray: A 2D numpy array representing the cosine similarity matrix.
    """
    # Extract features from the movie overviews using the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer)
    X_train = vectorizer.fit_transform(data["overview"])
    
    # Apply a cosine similarity function to the feature matrix
    similarity_matrix0 = cosine_similarity(X_train, X_train)
    
    return similarity_matrix0

def find_movie(data, movie_title):
    matched_title, score = process.extractOne(movie_title, data["title"].str.lower().tolist())
    # print(score)
    
    if score >= 88:  # Adjust the threshold as needed
        idx = data[data["title"].str.lower() == matched_title.lower()].index[0]
        return idx
    else:
        return "Movie not found"

def recommend_movies(movie_title:str, data:pd.DataFrame, sim_matrix:np.ndarray, n:int=5) -> pd.DataFrame:
    """
    Recommend similar movies to the input movie title based on the cosine similarity matrix.

    Args:
        movie_title (str): The title of the movie for which to find recommendations.
        data (pd.DataFrame): The input DataFrame containing the movie data.
        sim_matrix (np.ndarray): A 2D numpy array representing the cosine similarity matrix.
        n (int): The number of movie recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top n movie recommendations based on the similarity matrix.
    """
    # Get the index of the movie
    idx = find_movie(data, movie_title)
    
    if idx != "Movie not found":
        # Get the similarity row corresponding to the movie
        sim_scores = list(enumerate(sim_matrix[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top-n similar movies
        sim_scores = sim_scores[1:n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top-n most similar movies
        return list(data["title"].iloc[movie_indices])
    else:
        return "Movie not found"

def return_data():
    movies = pd.read_csv('./data/movies.csv', sep=',')
    movies = movies.dropna(subset=['overview'])
    
    return movies

def main():
    movies = return_data()
    sim_matrix = similarity_matrix(movies)
    while True:
        movie_title = input("Enter a movie title (or 'exit' to quit): ")
        if movie_title.lower() == 'exit':
            break
        recommendations = recommend_movies(movie_title, movies, sim_matrix, n=3)
        if recommendations == "Movie not found":
            print("Movie not found. Please enter another movie.")
        else:
            print(f"Top recommendations for {movie_title}:")
            for i, movie in enumerate(recommendations):
                print(f"{i+1}. {movie}")
    
if __name__ == "__main__":
    main()