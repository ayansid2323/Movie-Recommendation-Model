import ast
import pandas as pd
from thefuzz import process


def convert_to_list(obj:str):
    """
    Convert a string to a dictionary or list object.
    """
    L = ast.literal_eval(obj)
    return L


def get_keywords(obj:list[dict]):
    """
    Extract the 'name' field from each dictionary in the list.
    """
    L: str = ''
    limit = len(obj) - 1
    for i in obj:
        if i != limit:
            L += i['name'] + ", "
        else:
            L += i['name']
    return L


def join_series(series1:pd.Series, series2:pd.Series) -> list[str]:
    """
    Join two series into a single series.
    """
    try:
        L: list[str] = []
        i=0
        while i < len(series1):
            L.append("Keywords: " + (series1[i]) + "| Overview: " + str(series2[i]))
            i = i + 1
        return L
    except Exception as e:
        print(e)


def find_movie(input_title: str, data: pd.DataFrame) -> int|str:
    """
    Find the index of a movie in the DataFrame.
    The input movie title is refined using fuzzywuzzy.
    """
    
    matched_data = process.extract(input_title, data['title'], limit=2)
    
    if matched_data[0][1] >= 90:
        print(matched_data[0][1])
        index = data['title'].tolist().index(matched_data[0][0])
        return index, matched_data[0][0]
    else:
        print(matched_data[0][1])
        return ["Movie not found", "Movie not found"]