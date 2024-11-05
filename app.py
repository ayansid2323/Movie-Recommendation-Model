from fastapi import FastAPI, Request
from new_main import get_movies, get_similarity_matrix, get_recommendation
import uvicorn

app = FastAPI()

@app.get("/{sample_var}")
async def read_root(sample_var, request: Request):
    return {"message": sample_var}

@app.get("/recommend/{movie_title}") # http://127.0.0.1:8000/recommend/batman+begins
async def recommendMovies(movie_title: str, request: Request):
    if "+" in movie_title:
        movie_title = movie_title.replace("+", " ")
    movies = get_movies()
    sim_matrix = get_similarity_matrix(movies)
    recommendations, proper_title = get_recommendation(movie_title, movies, sim_matrix)
    if proper_title:
        return {"Movie Submitted": proper_title, "Recommendations": recommendations}
    else:
        return {"message": "Movie not found"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")