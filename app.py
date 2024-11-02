from fastapi import FastAPI, Request
from main import similarity_matrix, recommend_movies, return_data
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the Movie Recommendation System!"}

@app.get("/recommend")
async def recommendMovies(request: Request):
    movies = return_data()
    sim_matrix = similarity_matrix(movies)
    movie_title = "The Dark Knight"
    recommendations = recommend_movies(movie_title, movies, sim_matrix, n=3)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)