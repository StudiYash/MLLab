"""
Movie Recommendation System project utilities.

This module provides:
- functions to load ratings and movie titles,
- a function to build a recommender bundle (pivot table, stats),
- functions to recommend similar movies based on item-item correlation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class MovieRecommenderBundle:
    """
    Container for the Movie Recommender data and metadata.
    """
    ratings_df: pd.DataFrame       # Merged dataframe
    pivot_table: pd.DataFrame      # userId x title matrix
    movie_stats: pd.DataFrame      # index=title, columns=[rating_count, rating_mean]
    min_ratings_default: int       # default threshold

def get_ratings_tsv_path() -> Path:
    """
    Returns the path to file.tsv inside data/raw/movie_recommendation_system.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    return root / "data" / "raw" / "movie_recommendation_system" / "file.tsv"

def get_movie_titles_csv_path() -> Path:
    """
    Returns the path to Movie_Id_Titles.csv inside data/raw/movie_recommendation_system.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    return root / "data" / "raw" / "movie_recommendation_system" / "Movie_Id_Titles.csv"

def load_movie_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the ratings and titles datasets.
    """
    ratings_path = get_ratings_tsv_path()
    titles_path = get_movie_titles_csv_path()
    
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}")
    if not titles_path.exists():
        raise FileNotFoundError(f"Titles file not found at {titles_path}")
        
    # Load ratings (tab-separated, no header)
    # Based on inspection, it seems to be userId, movieId, rating, timestamp
    ratings_df = pd.read_csv(ratings_path, sep="\t", header=None, names=["userId", "movieId", "rating", "timestamp"])
    
    # Load titles (comma-separated)
    titles_df = pd.read_csv(titles_path)
    
    # Validate columns
    required_ratings = {"user_id", "item_id", "rating"} # Checking standard names first
    # The user prompt said "userId, movieId, rating" but let's check what's actually in standard MovieLens or similar
    # Usually it's user_id, item_id, rating, timestamp OR userId, movieId, rating.
    # Let's inspect columns if we could, but here we must implement robustly.
    # The prompt said: "file.tsv is a MovieLens-style ratings file with at least: userId, movieId, rating"
    # But often these files have headers like "user_id" "item_id". 
    # Let's standardize to "userId", "movieId", "rating".
    
    # Map common variations to standard names
    ratings_df.rename(columns={
        "user_id": "userId", "item_id": "movieId", 
        "User_ID": "userId", "Item_ID": "movieId", "Rating": "rating"
    }, inplace=True)
    
    titles_df.rename(columns={
        "item_id": "movieId", "Item_ID": "movieId", 
        "Title": "title", "movie_title": "title"
    }, inplace=True)
    
    if not {"userId", "movieId", "rating"}.issubset(ratings_df.columns):
        raise ValueError(f"Ratings dataset missing required columns. Found: {ratings_df.columns.tolist()}")
        
    if not {"movieId", "title"}.issubset(titles_df.columns):
        raise ValueError(f"Titles dataset missing required columns. Found: {titles_df.columns.tolist()}")
        
    return ratings_df, titles_df

def build_movie_recommender(
    min_ratings_default: int = 50
) -> MovieRecommenderBundle:
    """
    Loads data, merges, and builds the pivot table and stats for recommendation.
    """
    ratings_df, titles_df = load_movie_datasets()
    
    # Merge
    df = pd.merge(ratings_df, titles_df, on="movieId")
    
    # Drop rows with missing title or rating
    df = df.dropna(subset=["title", "rating"])
    
    # Build pivot table
    # index=userId, columns=title, values=rating
    pivot_table = df.pivot_table(index="userId", columns="title", values="rating")
    
    # Compute stats
    movie_stats = df.groupby("title")["rating"].agg(
        rating_count="count",
        rating_mean="mean"
    )
    
    return MovieRecommenderBundle(
        ratings_df=df,
        pivot_table=pivot_table,
        movie_stats=movie_stats,
        min_ratings_default=min_ratings_default
    )

def recommend_similar_movies(
    bundle: MovieRecommenderBundle,
    movie_title: str,
    top_n: int = 10,
    min_ratings: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Recommends movies similar to the given movie_title based on rating correlation.
    """
    if min_ratings is None:
        min_ratings = bundle.min_ratings_default
        
    if movie_title not in bundle.pivot_table.columns:
        raise ValueError(f"Movie '{movie_title}' not found in the rating matrix.")
        
    # Get ratings for target movie
    target_ratings = bundle.pivot_table[movie_title]
    
    # Compute correlation with all other movies
    # corrwith returns a Series indexed by title
    corr_series = bundle.pivot_table.corrwith(target_ratings)
    
    # Create DataFrame
    corr_df = pd.DataFrame({"correlation": corr_series})
    corr_df = corr_df.dropna()
    
    # Join with stats to get rating_count
    # movie_stats index is title
    corr_df = corr_df.join(bundle.movie_stats)
    
    # Filter
    # 1. Remove self
    if movie_title in corr_df.index:
        corr_df = corr_df.drop(movie_title)
        
    # 2. Min ratings
    corr_df = corr_df[corr_df["rating_count"] >= min_ratings]
    
    # Sort by correlation desc
    corr_df = corr_df.sort_values(by="correlation", ascending=False)
    
    # Take top N
    top_df = corr_df.head(top_n)
    
    results = []
    for title, row in top_df.iterrows():
        results.append({
            "title": str(title),
            "correlation": float(row["correlation"]),
            "rating_count": int(row["rating_count"]),
            "rating_mean": float(row["rating_mean"])
        })
        
    return results

def safe_recommend(
    bundle: MovieRecommenderBundle,
    movie_title: str,
    top_n: int = 10,
    min_ratings: Optional[int] = None
) -> Dict[str, Any]:
    """
    Safe wrapper for recommendation that handles errors and returns a structured dict.
    """
    used_min_ratings = min_ratings if min_ratings is not None else bundle.min_ratings_default
    
    try:
        recs = recommend_similar_movies(bundle, movie_title, top_n, min_ratings)
        return {
            "query": movie_title,
            "min_ratings_used": used_min_ratings,
            "results": recs
        }
    except ValueError as e:
        return {
            "query": movie_title,
            "min_ratings_used": used_min_ratings,
            "results": [],
            "error": str(e)
        }
    except Exception as e:
        return {
            "query": movie_title,
            "min_ratings_used": used_min_ratings,
            "results": [],
            "error": f"An unexpected error occurred: {str(e)}"
        }

if __name__ == "__main__":
    print("Building movie recommender from file.tsv and Movie_Id_Titles.csv ...")
    try:
        bundle = build_movie_recommender(min_ratings_default=50)
        print("Number of movies:", len(bundle.movie_stats))

        # For a smoke test, pick one popular movie (highest rating_count).
        popular_titles = bundle.movie_stats.sort_values(
            by="rating_count",
            ascending=False
        ).head(1).index.tolist()

        if popular_titles:
            test_title = popular_titles[0]
            print(f"Requesting recommendations for: {test_title}")
            result = safe_recommend(bundle, test_title, top_n=5)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("Recommendations:")
                for r in result.get("results", []):
                    print(f"  - {r['title']} (corr={r['correlation']:.3f}, "
                          f"count={r['rating_count']}, mean={r['rating_mean']:.2f})")
        else:
            print("No movies found in movie_stats; check the input files.")
            
    except FileNotFoundError as e:
        print(f"Failed to load data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
