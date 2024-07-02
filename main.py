

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import joblib
from prediction_model.config.config import COUNT_VECTORIZER_PATH, SIMILARITY_MATRIX_PATH, MOVIE_LIST_PATH

app = FastAPI()


# Load models and data
count_vectorizer = joblib.load(COUNT_VECTORIZER_PATH)
similarity_matrix = joblib.load(SIMILARITY_MATRIX_PATH)
movies = joblib.load(MOVIE_LIST_PATH)

def get_recommendations(title):
    if title not in movies['title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    index = movies[movies['title'] == title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = similarity_scores[1:11]
    
    recommended_movies = [movies.iloc[i[0]]['title'] for i in top_similar_movies]
    
    return recommended_movies

@app.get("/recommend_json/{title}", response_model=list)
async def recommend_json(title: str):
    recommendations = get_recommendations(title)
    return recommendations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
