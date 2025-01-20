from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

def get_sentiment_category(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'

def load_data():
    """Load and prepare data for visualization"""
    try:
        df = pd.read_csv('enriched_jobs.csv')
        
        # Convert string representations of lists to actual lists
        for col in ['programming_keywords', 'tools_keywords', 'skills_keywords']:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else [])
        
        # Calculate total skills
        df['total_skills'] = df['programming_keywords'].apply(len) + \
                            df['tools_keywords'].apply(len) + \
                            df['skills_keywords'].apply(len)
        
        # Convert applications to numeric
        df['applications_numeric'] = df['num_applications'].apply(
            lambda x: float(x.replace('<', '').replace(' applicants', '')) if pd.notnull(x) else 25
        )

        # Perform sentiment analysis
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['description'].fillna('').apply(
            lambda x: sia.polarity_scores(x)['compound']
        )
        df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

@app.route('/')
def dashboard():
    df = load_data()
    if df is None:
        return "Error loading data"

    # Create visualizations using Plotly
    # 1. Location Distribution
    location_fig = px.bar(
        df['location'].value_counts().head(10),
        title='Top 10 Job Locations'
    )

    # 2. Work Type Distribution
    work_type_fig = px.pie(
        df, 
        names='work_type',
        title='Distribution of Work Types'
    )

    # 3. Skills vs Applications Scatter
    skills_fig = px.scatter(
        df,
        x='total_skills',
        y='applications_numeric',
        title='Job Applications vs Required Skills',
        trendline="ols"
    )

    # 4. Sentiment Distribution
    sentiment_fig = px.bar(
        df['sentiment_category'].value_counts(),
        title='Job Description Sentiment Distribution'
    )

    # Convert plots to JSON for template
    graphs = {
        'location': json.dumps(location_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'work_type': json.dumps(work_type_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'skills': json.dumps(skills_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'sentiment': json.dumps(sentiment_fig, cls=plotly.utils.PlotlyJSONEncoder)
    }

    # Calculate summary statistics
    stats = {
        'total_jobs': len(df),
        'avg_applications': df['applications_numeric'].mean(),
        'avg_skills': df['total_skills'].mean(),
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return render_template('dashboard.html', graphs=graphs, stats=stats)

if __name__ == '__main__':
    app.run(debug=True) 