from fastapi import FastAPI, Request
from new_main import get_movies, get_similarity_matrix, get_recommendation
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return {"message": "Hello World!"}

@app.get("/recommend")
async def recommendMovies(request: Request):
    movies = get_movies()
    sim_matrix = get_similarity_matrix(movies)
    movie_title = "dark knight"
    recommendations, proper_title = get_recommendation(movie_title, movies, sim_matrix)
    if proper_title:
        return {"Movie Submitted": proper_title, "Recommendations": recommendations}
    else:
        return {"message": "Movie not found"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")