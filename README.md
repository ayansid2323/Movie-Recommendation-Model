# Movie Recommendation System - API

This project is a movie recommendation system that utilizes content-based filtering. It is built using FastAPI for the REST API, Sci-Kit Learn for machine learning, and Pandas for data manipulation.

## Features
- **REST API**: Get movie recommendations through HTTP endpoints using FastAPI
- **Content-Based Filtering**: Recommends movies based on the similarity of their overviews and keywords
- **Machine Learning**: Utilizes Sci-Kit Learn for feature extraction and similarity computation
- **Data Manipulation**: Efficiently handles and preprocesses data using Pandas

## Getting Started
To get started with the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ayansid2323/Movie-Recommendation-Model.git
    ```
2. **Navigate to the project directory**:
    ```sh
    cd Movie-Recommendation-Model
    ```
3. **Create a virtual environment**:
    ```sh
    python -m venv .venv
    ```
4. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh 
        source venv/bin/activate
        ```
5. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
6. **Run the API server**:
    ```sh
    python app.py
    ```

## Usage
### CLI Interface
- Run `python main.py` for command line interface
- Enter movie titles when prompted to get recommendations

### REST API
The API is available at `http://localhost:8000` with the following endpoints:

- GET `/recommend/{movie_title}`: Get movie recommendations
  ```sh
  curl http://localhost:8000/recommend/batman+begins
  ```
  
Example Response:
```json
{
    "Movie Submitted": "Batman Begins",
    "Recommendations": [
        "The Dark Knight",
        "Batman Returns",
        "Batman Forever",
        "Batman & Robin",
        "The Dark Knight Rises"
    ]
}
```

## Data
The dataset used in this project is located in the `data/` directory:
- `movies.csv`: Movie metadata (titles, overviews, genres)
- `ratings.csv`: User ratings
- `credits.csv`: Cast and crew information

## Docker Support
Build and run using Docker:
```sh
docker compose up --build
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
Feel free to reach out for any questions or feedback!