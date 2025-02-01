from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import ast
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import nltk
from nltk.stem.porter import PorterStemmer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure Flask with proper paths
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)
CORS(app)

def get_data_path(filename):
    """Get absolute path to data files"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', filename)

def preprocess_data():
    try:
        # Use absolute paths for file system
        parquet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessed.parquet')
        
        if not os.path.exists(parquet_path):
            print("🚀 Processing raw data...")
            
            # Load datasets from data directory
            movies = pd.read_csv(get_data_path('tmdb_5000_movies.csv'))
            credits = pd.read_csv(get_data_path('tmdb_5000_credits.csv'))
            movies = movies.merge(credits, on='title')
            
            # Clean and process data
            movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
            movies.dropna(inplace=True)

            def safe_convert(obj):
                try:
                    return [item['name'] for item in ast.literal_eval(obj)]
                except:
                    return []

            movies = movies.assign(
                genres=movies['genres'].apply(safe_convert),
                keywords=movies['keywords'].apply(safe_convert),
                cast=movies['cast'].apply(lambda x: safe_convert(x)[:3]),
                crew=movies['crew'].apply(
                    lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
            )

            # Text processing
            movies['overview'] = movies['overview'].apply(lambda x: x.split())
            for col in ['genres', 'keywords', 'cast', 'crew']:
                movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
            
            # Create tags column
            movies['tags'] = movies.apply(
                lambda row: ' '.join(
                    [' '.join(row['overview'])] +
                    [' '.join(row['genres'])] +
                    [' '.join(row['keywords'])] +
                    [' '.join(row['cast'])] +
                    [' '.join(row['crew'])]
                ), axis=1
            )

            new_df = movies[['movie_id', 'title', 'tags']].copy()
            
            # Stemming
            ps = PorterStemmer()
            new_df['tags'] = new_df['tags'].apply(
                lambda x: ' '.join([ps.stem(word) for word in x.split()]))
            
            # Save processed data
            new_df.to_parquet(parquet_path, compression='gzip')
            print("✅ Saved preprocessed data")
        
        # Load processed data
        processed = pd.read_parquet(parquet_path)
        return processed, processed
    
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        sys.exit(1)

# Initialize application
print("\n🎬 Initializing Movie Recommender System")
try:
    new_df, movies = preprocess_data()
    
    similarity_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'similarity.npz')
    if not os.path.exists(similarity_path):
        print("⚙️ Creating similarity matrix...")
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags'])
        similarity = cosine_similarity(vectors)
        
        # Precision conversion before saving
        similarity = similarity.astype(np.float16)
        np.savez_compressed(similarity_path, similarity)
        print("✅ Saved similarity matrix")
    else:
        print("🔍 Loading precomputed similarity matrix...")
        similarity = np.load(similarity_path)['arr_0']
        
        # Restore original precision
        similarity = similarity.astype(np.float32)

except Exception as e:
    print(f"🔥 Critical initialization error: {str(e)}")
    sys.exit(1)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        data = request.get_json()
        movie_title = data['movie'].strip().lower()
        
        matches = movies[movies['title'].str.lower() == movie_title]
        if matches.empty:
            return jsonify({"error": "Movie not found. Try another title!"}), 404
        
        index = matches.index[0]
        sim_scores = sorted(enumerate(similarity[index]), 
                         key=lambda x: x[1], reverse=True)[1:6]
        recommendations = [movies.iloc[i[0]]['title'] for i in sim_scores]
        
        return jsonify({"recommendations": recommendations})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/movies')
def get_movie_titles():
    try:
        movie_titles = movies['title'].str.strip().unique().tolist()
        return jsonify(sorted(movie_titles))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    nltk.download('punkt', quiet=True)
    print("\n🚀 Server ready at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
