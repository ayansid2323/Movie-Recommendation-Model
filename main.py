import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import ast


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

def combined_text(row):
    """
    Combines the 'overview' and 'keywords_str' columns of a DataFrame row into a single text string.
    """
    return row['overview'] + ' ' + row['keywords_str']

def similarity_matrix(data:pd.DataFrame) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for the movie overviews and keywords_str.

    Returns:
        numpy.ndarray: A 2D numpy array representing the cosine similarity matrix.
    """
    # Combine "overview" and "keywords_str" into a single text column
    data['combined_text'] = data.apply(combined_text, axis=1)
    
    # Extract features from the combined text using the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer)
    X_train = vectorizer.fit_transform(data["combined_text"])
    
    # Apply a cosine similarity function to the feature matrix
    similarity_matrix0 = cosine_similarity(X_train, X_train)
    
    return similarity_matrix0

def find_movie(data: pd.DataFrame, movie_title:str) -> int|str:
    """
    Find the index of the movie in the DataFrame based on the movie title.
    """
    # Find the closest matching movie title
    matched_title, score = process.extractOne(movie_title, data["title"].str.lower().tolist())
    
    # If the similarity score is above a certain threshold, return the index of the movie
    if score >= 88: # You can adjust the threshold here
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
    """
    Reads movie data from a CSV file, processes it to extract and format keywords, and returns the processed DataFrame.
    The function performs the following steps:
    1. Reads the 'movies.csv' file from the './data/' directory using pandas.
    2. Drops rows where the 'overview' column has missing values.
    3. Parses the 'keywords' column, which contains string representations of lists of dictionaries, into actual lists of dictionaries.
    4. Extracts the 'name' field from each dictionary in the 'keywords' lists and creates a new column 'keywords_extracted' with these names.
    5. Joins the extracted keywords into a single string for each movie and creates a new column 'keywords_str'.
    Returns:
        pandas.DataFrame: A DataFrame containing the processed movie data with additional columns 'keywords_extracted' and 'keywords_str'.
    """
    movies = pd.read_csv('./data/movies.csv', sep=',')
    
    # Drop rows with missing values in the 'overview' and 'keywords' columns
    movies = movies.dropna(subset=['overview'])
    movies = movies.dropna(subset=['keywords'])
    
    # Convert the string representation of lists of dictionaries to actual lists of dictionaries
    movies["keywords_extracted"] = movies["keywords"].apply(ast.literal_eval)
    
    # Extract the 'name' field from each dictionary in the 'keywords' lists
    movies["keywords_extracted"] = movies["keywords_extracted"].apply(lambda x: [d['name'] for d in x])
    
    # Join the extracted keywords into a single string for each movie
    movies["keywords_str"] = movies["keywords_extracted"].apply(lambda x: ', '.join(x))
    
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