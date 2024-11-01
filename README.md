# Movie Recommendation System

This project is a movie recommendation system that utilizes content-based filtering. It is built using the Sci-Kit Learn framework for machine learning and Pandas for data manipulation.

## Features
- **Content-Based Filtering**: Recommends movies based on the similarity of their overviews.
- **Machine Learning**: Utilizes Sci-Kit Learn for feature extraction and similarity computation.
- **Data Manipulation**: Efficiently handles and preprocesses data using Pandas.

## Getting Started
To get started with the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/movieRecommendation.git
    ```
2. **Navigate to the project directory**:
    ```sh
    cd movieRecommendation
    ```
3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
4. **Run the main script**:
    ```sh
    python main.py
    ```

## Usage
- The `main.py` script contains the main logic for the movie recommendation system.
- When prompted, enter a movie title to get recommendations for similar movies.
- Modify the script to customize the recommendation algorithm or input data.

## Example
Here's an example of how to use the recommendation system:

1. Run the script:
    ```sh
    python main.py
    ```
2. Enter a movie title when prompted:
    ```
    Enter a movie title (or 'exit' to quit): Avatar
    ```
3. Get the top 3 movie recommendations:
    ```
    Top 3 movies similar to 'Avatar':
    1. Pirates of the Caribbean: At World's End
    2. Spectre
    3. The Dark Knight Rises
    ```

## Data
The dataset used in this project is located in the `data/` directory and includes the following files:
- `movies.csv`: Contains movie metadata including titles, overviews, genres, etc.
- `ratings.csv`: Contains user ratings for movies.
- `credits.csv`: Contains cast and crew information for each movie.

## Acknowledgements
- This project was inspired by the concept of content-based filtering in recommendation systems.
- Special thanks to the Sci-Kit Learn and Pandas libraries for their powerful features.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
Feel free to reach out for any questions or feedback!