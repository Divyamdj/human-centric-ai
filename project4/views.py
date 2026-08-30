from django.http import FileResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import json
from django.http import JsonResponse
import os
from django.conf import settings


# Serve the matrix factorization PDF
def download_pdf(request):
    pdf_path = os.path.join(
        settings.BASE_DIR,
        "project4",
        "static",
        "pdfs",
        "about_matrix_factorization.pdf",
    )
    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename="about_matrix_factorization.pdf",
    )


def load_movielens_data():
    try:
        ratings_path = os.path.join(settings.BASE_DIR, "media", "ratings.csv")
        movies_path = os.path.join(settings.BASE_DIR, "media", "movies.csv")

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

        return ratings, movies
    except Exception as e:
        raise e


class MovieRecommender:
    def __init__(
        self, n_factors=10, lambda_reg=0.1, learning_rate=0.01, n_iterations=100
    ):
        try:
            self.n_factors = n_factors  # K in the formula
            self.lambda_reg = lambda_reg  # λ in the formula
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations

            self.ratings, self.movies = load_movielens_data()
            self.user_movie_matrix = None
            self.U = None  # User latent factors
            self.V = None  # Item latent factors
            self.user_id_map = {}  # Map user IDs to matrix indices
            self.movie_id_map = {}  # Map movie IDs to matrix indices
            self.setup_model()
        except Exception as e:
            raise e

    def setup_model(self):
        """Create user-movie matrix and perform matrix factorization"""
        try:
            # Create user-movie matrix
            self.user_movie_matrix = self.ratings.pivot_table(
                index="userId", columns="movieId", values="rating"
            ).fillna(0)

            # Create mappings for user and movie IDs
            self.user_id_map = {
                user_id: idx for idx, user_id in enumerate(self.user_movie_matrix.index)
            }
            self.movie_id_map = {
                movie_id: idx
                for idx, movie_id in enumerate(self.user_movie_matrix.columns)
            }

            # Convert to numpy array for matrix factorization
            self.R = self.user_movie_matrix.values
            self.n_users, self.n_movies = self.R.shape

            # Initialize U and V matrices randomly
            np.random.seed(42)
            self.U = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
            self.V = np.random.normal(0, 0.1, (self.n_movies, self.n_factors))

            # Perform matrix factorization
            self._matrix_factorization()

        except Exception as e:
            raise e

    def _matrix_factorization(self):
        """Perform matrix factorization using gradient descent"""

        # Get indices of non-zero ratings (observed ratings)
        self.observed_ratings = np.where(self.R != 0)

        for _ in range(self.n_iterations):
            # Calculate current error and cost
            error = 0

            # Update U and V using gradient descent
            for i, j in zip(self.observed_ratings[0], self.observed_ratings[1]):
                r_ij = self.R[i, j]
                prediction = np.dot(self.U[i], self.V[j])
                e_ij = r_ij - prediction
                error += e_ij**2

                # Store old values for update
                U_i = self.U[i].copy()
                V_j = self.V[j].copy()

                # Gradient descent updates (from the formula)
                self.U[i] += self.learning_rate * (e_ij * V_j - self.lambda_reg * U_i)
                self.V[j] += self.learning_rate * (e_ij * U_i - self.lambda_reg * V_j)

            # Calculate total cost with regularization
            cost = error + self.lambda_reg * (np.sum(self.U**2) + np.sum(self.V**2))

    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair using learned factors"""
        if user_id in self.user_id_map and movie_id in self.movie_id_map:
            user_idx = self.user_id_map[user_id]
            movie_idx = self.movie_id_map[movie_id]
            prediction = np.dot(self.U[user_idx], self.V[movie_idx])
            return max(0.5, min(5.0, prediction))
        return 3.0  # Default neutral rating

    def get_new_user_factors(self, user_ratings):
        """Learn factors for a new user (cold start) - formula's second equation"""

        # Initialize new user factors
        u_new = np.random.normal(0, 0.1, self.n_factors)

        # Convert user ratings to indices
        rated_movie_indices = []
        ratings_vector = []

        for movie_id, rating in user_ratings.items():
            if movie_id in self.movie_id_map:
                movie_idx = self.movie_id_map[movie_id]
                rated_movie_indices.append(movie_idx)
                ratings_vector.append(rating)

        if not rated_movie_indices:
            return u_new

        rated_movie_indices = np.array(rated_movie_indices)
        ratings_vector = np.array(ratings_vector)

        # Optimize new user factors using gradient descent
        for iteration in range(50):  # Fewer iterations for new user
            predictions = np.dot(u_new, self.V[rated_movie_indices].T)
            errors = ratings_vector - predictions

            # Gradient update for new user (from the formula's second equation)
            gradient = np.sum(
                [
                    errors[i] * self.V[rated_movie_indices[i]]
                    for i in range(len(errors))
                ],
                axis=0,
            )
            gradient -= self.lambda_reg * u_new

            u_new += self.learning_rate * gradient

            if iteration % 10 == 0:
                cost = np.sum(errors**2) + self.lambda_reg * np.sum(u_new**2)

        return u_new

    def get_movie_candidates(self, n_candidates=5):
        """Get candidate movies for rating (popular but diverse)"""
        try:

            # Get movies with most ratings (popular)
            movie_counts = self.ratings["movieId"].value_counts()
            popular_movies = movie_counts.head(20).index.tolist()

            # Sample diverse movies
            n_candidates = min(n_candidates, len(popular_movies))
            candidates = np.random.choice(popular_movies, n_candidates, replace=False)

            candidate_info = []
            for movie_id in candidates:
                movie_info = self.movies[self.movies["movieId"] == movie_id]

                if movie_info.empty:
                    continue

                movie_info = movie_info.iloc[0]
                movie_ratings = self.ratings[self.ratings["movieId"] == movie_id][
                    "rating"
                ]
                avg_rating = movie_ratings.mean()
                n_ratings = len(movie_ratings)

                candidate_info.append(
                    {
                        "movieId": int(movie_id),
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "avg_rating": round(float(avg_rating), 2),
                        "n_ratings": int(n_ratings),
                    }
                )

            return candidate_info
        except Exception:
            return []

    def predict_impact(self, user_ratings, movie_id, hypothetical_rating):
        """Predict how rating a movie will affect recommendations using matrix factorization"""
        try:

            # Create temporary user profile with hypothetical rating
            temp_ratings = user_ratings.copy()
            temp_ratings[movie_id] = hypothetical_rating

            # Learn factors for this user with the new rating
            user_factors = self.get_new_user_factors(temp_ratings)

            # Generate recommendations using matrix factorization
            recommendations = self._get_mf_recommendations(
                user_factors, temp_ratings, hypothetical_rating
            )

            return recommendations

        except Exception as e:
            return []

    def _get_mf_recommendations(self, user_factors, user_ratings, hypothetical_rating):
        """Get recommendations using matrix factorization predictions"""
        try:

            # Calculate predictions for all movies
            all_predictions = np.dot(user_factors, self.V.T)

            # Create list of movie recommendations
            recommendations = []
            rated_movies = set(user_ratings.keys())

            # Get top unrated movies with highest predicted ratings
            movie_scores = []
            for movie_idx, movie_id in enumerate(self.user_movie_matrix.columns):
                if movie_id not in rated_movies:
                    predicted_rating = max(0.5, min(5.0, all_predictions[movie_idx]))
                    movie_scores.append((movie_id, predicted_rating))

            # Sort by predicted rating and take top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_movies = movie_scores[:10]  # Get top 10 candidates

            # Convert to recommendation format
            for movie_id, predicted_rating in top_movies[:5]:  # Return top 5
                movie_info = self.movies[self.movies["movieId"] == movie_id]
                if not movie_info.empty:
                    movie_data = movie_info.iloc[0]

                    # Calculate confidence based on prediction strength and user rating pattern
                    base_confidence = min(
                        1.0, (predicted_rating - 2.5) / 2.5
                    )  # Higher for better predictions
                    rating_confidence = self._get_rating_confidence(hypothetical_rating)
                    confidence = (base_confidence + rating_confidence) / 2

                    recommendations.append(
                        {
                            "title": movie_data["title"],
                            "predicted_rating": round(float(predicted_rating), 2),
                            "genres": movie_data["genres"],
                            "confidence": round(max(0.1, confidence), 2),
                            "num_similar_users": len(
                                user_ratings
                            ),  # Based on training data size
                            "ranking_score": round(float(predicted_rating), 2),
                        }
                    )

            if not recommendations:
                return self._get_fallback_recommendations(hypothetical_rating)

            return recommendations

        except Exception:
            return self._get_fallback_recommendations(hypothetical_rating)

    def _get_rating_confidence(self, rating):
        """Calculate confidence based on rating value"""
        if rating >= 4.0:
            return 0.9  # High confidence for likes
        elif rating >= 3.0:
            return 0.7  # Medium confidence for neutral
        elif rating >= 2.0:
            return 0.6  # Lower confidence for dislikes
        else:
            return 0.4  # Lowest confidence for strong dislikes

    def _get_fallback_recommendations(self, hypothetical_rating):
        """Provide fallback recommendations when no similar users found"""
        try:
            fallback_movies = []

            if hypothetical_rating <= 2.5:
                # User dislikes the movie, recommend diverse popular movies from different genres
                # Get movies from different genres to ensure diversity
                diverse_movies = self.ratings["movieId"].value_counts().head(50)
                # Sample from these to ensure diversity
                selected_movies = diverse_movies.index[:10]
            elif hypothetical_rating >= 4.0:
                # User likes the movie, recommend popular movies
                popular_movies = self.ratings["movieId"].value_counts().head(20)
                selected_movies = popular_movies.index[:5]
            else:
                # Medium rating, use moderately popular movies
                popular_movies = self.ratings["movieId"].value_counts().head(30)
                selected_movies = popular_movies.index[5:10]  # Skip the most popular

            for movie_id in selected_movies[:5]:
                movie_info = self.movies[self.movies["movieId"] == movie_id]
                if not movie_info.empty:
                    movie_data = movie_info.iloc[0]
                    movie_ratings = self.ratings[self.ratings["movieId"] == movie_id][
                        "rating"
                    ]
                    avg_rating = movie_ratings.mean()

                    # Ensure rating is in valid range
                    display_rating = max(0.5, min(5.0, avg_rating))

                    fallback_movies.append(
                        {
                            "title": movie_data["title"],
                            "predicted_rating": round(float(display_rating), 2),
                            "genres": movie_data["genres"],
                            "confidence": 0.3,  # Lower confidence for fallback
                            "num_similar_users": 0,  # No similar users for fallback
                        }
                    )

            return fallback_movies
        except Exception:
            return []


# Global recommender instance
try:
    print("Creating global recommender instance...")
    recommender = MovieRecommender()
    print("Global recommender created successfully!")
except Exception as e:
    print(f"Failed to create global recommender: {e}")
    recommender = None


def index(request):
    return render(
        request,
        "recommendation.html",
        {
            "title": "Movie Recommender - Cold Start Learning",
            "description": "Help us learn your movie preferences!",
        },
    )


def get_movie_candidates_view(request):
    """AJAX endpoint to get movie candidates"""
    try:
        print("get_movie_candidates called")

        if recommender is None:
            print("Recommender is None!")
            return JsonResponse({"error": "Recommender not initialized"}, status=500)

        candidates = recommender.get_movie_candidates(n_candidates=5)
        print(f"Returning {len(candidates)} candidates to frontend")

        return JsonResponse({"candidates": candidates})
    except Exception as e:
        print(f"Error in get_movie_candidates: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def predict_rating_impact(request):
    """AJAX endpoint to predict impact of rating"""
    try:
        if request.method == "POST":
            data = json.loads(request.body)
            user_ratings = data.get("user_ratings", {})
            movie_id = data.get("movie_id")
            rating = data.get("rating")

            print(f"Predicting impact for movie {movie_id} with rating {rating}")
            print(f"Current user ratings: {user_ratings}")

            # Convert string keys to int
            user_ratings = {int(k): v for k, v in user_ratings.items()}

            if recommender is None:
                return JsonResponse(
                    {"error": "Recommender not initialized"}, status=500
                )

            impact = recommender.predict_impact(
                user_ratings, int(movie_id), float(rating)
            )

            print(f"Generated {len(impact)} recommendations for rating {rating}")

            # Create more informative message based on rating
            if float(rating) <= 2.5:
                message = f"If you rate this movie {rating} stars (dislike), here are movies that others who also disliked it actually enjoyed:"
            else:
                message = f"If you rate this movie {rating} stars, here are movies that similar users also liked:"

            return JsonResponse({"impact": impact, "message": message})
    except Exception as e:
        print(f"Error in predict_rating_impact: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def get_final_recommendations(request):
    """Get final recommendations based on all user ratings"""
    try:
        if request.method == "POST":
            data = json.loads(request.body)
            user_ratings = data.get("user_ratings", {})

            # Convert string keys to int
            user_ratings = {int(k): v for k, v in user_ratings.items()}

            if recommender is None:
                return JsonResponse(
                    {"error": "Recommender not initialized"}, status=500
                )

            # Get final recommendations using a comprehensive approach
            final_recs = []
            if user_ratings:
                # Use each rated movie to generate recommendations and combine them
                all_recommendations = {}

                for movie_id, rating in user_ratings.items():
                    # Get recommendations based on this rating
                    movie_recs = recommender.predict_impact(
                        user_ratings, movie_id, rating
                    )

                    # Weight recommendations based on user's rating
                    weight = rating / 5.0  # Higher weight for movies user liked more

                    for rec in movie_recs:
                        title = rec["title"]
                        if title not in all_recommendations:
                            all_recommendations[title] = {
                                "title": rec["title"],
                                "genres": rec["genres"],
                                "total_score": 0,
                                "count": 0,
                            }

                        # Add weighted score
                        all_recommendations[title]["total_score"] += (
                            rec["predicted_rating"] * weight
                        )
                        all_recommendations[title]["count"] += 1

                # Calculate final scores and sort
                final_scores = []
                for title, data in all_recommendations.items():
                    if data["count"] > 0:
                        avg_score = data["total_score"] / data["count"]
                        final_scores.append(
                            {
                                "title": data["title"],
                                "predicted_rating": round(avg_score, 2),
                                "genres": data["genres"],
                                "recommendation_strength": data[
                                    "count"
                                ],  # How many models recommended this
                            }
                        )

                # Sort by score and take top recommendations
                final_scores.sort(key=lambda x: x["predicted_rating"], reverse=True)
                final_recs = final_scores[:8]  # Return top 8 recommendations

            if not final_recs:
                # Fallback to popular movies
                final_recs = recommender._get_fallback_recommendations(3.5)[:5]

            return JsonResponse(
                {
                    "recommendations": final_recs,
                    "message": f"Based on your {len(user_ratings)} ratings, here are our top recommendations:",
                }
            )
    except Exception as e:
        print(f"Error in get_final_recommendations: {e}")
        return JsonResponse({"error": str(e)}, status=500)
