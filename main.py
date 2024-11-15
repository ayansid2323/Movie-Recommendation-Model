import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import convert_to_list, get_keywords, join_series, find_movie


def get_movies():
    # Load the data
    movies = pd.read_csv('data/movies.csv', sep=',')
    
    movies['keywords'] = movies['keywords'].apply(convert_to_list)
    movies['keyword'] = movies['keywords'].apply(get_keywords)
    
    movies['combined_features'] = join_series(movies['keyword'], movies['overview'])
    
    return movies


def get_similarity_matrix(movies: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix

def get_recommendation(movie_title: str, movies: pd.DataFrame, similarity_matrix: np.ndarray) -> list[str]:
    index, proper_title = find_movie(movie_title, movies)
    if index != "Movie not found":
        similar_movies = list(enumerate(similarity_matrix[index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommendations: list[str] = []
        
        for i in range(1, 6):
            movie_index = similar_movies[i][0]
            recommendations.append(movies.loc[movie_index, 'title'])
        
        return recommendations, proper_title
    else:
        print("Movie not found"), None


def main():
    movies = get_movies()
    similarity_matrix = get_similarity_matrix(movies)
    while True:
        movie_title = input("Enter a movie title (or 'exit' to quit): ")
        if movie_title.lower() == 'exit':
            break
        else:
            recommendations, proper_title = get_recommendation(movie_title, movies, similarity_matrix)
            if proper_title:
                print(f"Movies similar to {proper_title}:")
                for movie in recommendations:
                    print(movie)
            else:
                print("Movie not found")
            

if __name__ == "__main__":
    main()