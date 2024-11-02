from fastapi import FastAPI, Request
from main import similarity_matrix, recommend_movies, return_data
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return {"message": "Hello World!"}

@app.get("/recommend")
async def recommendMovies(request: Request):
    movies = return_data()
    sim_matrix = similarity_matrix(movies)
    movie_title = "dark knight"
    recommendations = recommend_movies(movie_title, movies, sim_matrix, n=3)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")