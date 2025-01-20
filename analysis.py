import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec

# Read the enriched dataset
df = pd.read_csv('enriched_jobs.csv')

# Convert application numbers to numeric values (moved earlier)
df['applications_numeric'] = df['num_applications'].apply(
    lambda x: float(x.replace('<', '').replace(' applicants', '')) if pd.notnull(x) else 25
)

# Calculate total skills before using it
df['total_skills'] = df['programming_keywords'].apply(len) + \
                    df['tools_keywords'].apply(len) + \
                    df['skills_keywords'].apply(len)

# Create a new figure with a more complex layout using GridSpec
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)

# 1. Location Trends
print("=== Top 5 Locations with Most Job Postings ===")
location_counts = df['location'].value_counts().head()
ax1 = fig.add_subplot(gs[0, 0])
location_counts.plot(kind='bar')
ax1.set_title('Top 5 Locations with Most Job Postings')
ax1.tick_params(axis='x', rotation=45)
print(location_counts)
print("\n")

# 2. Word Cloud of Job Description Keywords
print("=== Word Cloud of Job Keywords ===")
# Combine all text from descriptions
text = ' '.join(df['clean_description'].fillna(''))
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word Cloud of Job Descriptions')

# 3. Job Type Distribution (Remote vs. In-office vs. Hybrid)
print("=== Job Type Distribution ===")
work_type_counts = df['work_type'].value_counts()
ax3 = fig.add_subplot(gs[1, 0])
work_type_counts.plot(kind='pie', autopct='%1.1f%%')
ax3.set_title('Distribution of Job Types')
print(work_type_counts)
print("\n")

# 4. Scatter Plot: Job Popularity vs. Skills Count
print("=== Job Popularity vs. Skills Analysis ===")
print(f"Average number of skills required: {df['total_skills'].mean():.1f}")
print(f"Correlation between skills and applications: {df['total_skills'].corr(df['applications_numeric']):.3f}")
print(f"Maximum skills required: {df['total_skills'].max()}")
print(f"Minimum skills required: {df['total_skills'].min()}")
print("\n")

ax4 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=df, x='total_skills', y='applications_numeric', alpha=0.5, ax=ax4)
ax4.set_title('Job Applications vs. Required Skills')
ax4.set_xlabel('Number of Required Skills')
ax4.set_ylabel('Number of Applications')

# Perform sentiment analysis before creating visualizations
print("\n=== Sentiment Analysis of Job Descriptions ===")
sia = SentimentIntensityAnalyzer()

def get_sentiment_category(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'

# Analyze sentiment for each job description
df['sentiment_score'] = df['description'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)

# Now create visualizations
sentiment_counts = df['sentiment_category'].value_counts()
print("Distribution of sentiments:")
print(sentiment_counts)
print(f"\nPercentage of positive descriptions: {(sentiment_counts.get('Positive', 0) / len(df) * 100):.1f}%")
print(f"Percentage of neutral descriptions: {(sentiment_counts.get('Neutral', 0) / len(df) * 100):.1f}%")
print(f"Percentage of negative descriptions: {(sentiment_counts.get('Negative', 0) / len(df) * 100):.1f}%")
print(f"Average sentiment score: {df['sentiment_score'].mean():.3f}")
print("\n")

ax5 = fig.add_subplot(gs[2, 0])
sentiment_counts.plot(kind='bar')
ax5.set_title('Distribution of Job Description Sentiments')
ax5.set_xlabel('Sentiment Category')
ax5.set_ylabel('Count')

# Save the visualizations before model training
plt.tight_layout()
plt.savefig('job_market_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Job Popularity Prediction Model
print("\n=== Job Popularity Prediction Model ===")

# Prepare features for the model
# Convert categorical variables to numeric
le = LabelEncoder()
X = pd.DataFrame({
    'description_length': df['description'].fillna('').str.len(),
    'num_programming_skills': df['programming_keywords'].apply(len),
    'num_tools': df['tools_keywords'].apply(len),
    'num_skills': df['skills_keywords'].apply(len),
    'location_encoded': le.fit_transform(df['location'].fillna('unknown')),
    'days_ago': df['days_ago'].fillna(df['days_ago'].median()),
    'sentiment_score': df['sentiment_score']
})

y = df['applications_numeric']

# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Create a new figure for feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.plot(kind='barh', x='Feature', y='Importance')
plt.title('Feature Importance in Popularity Prediction')
plt.xlabel('Absolute Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
plt.close()

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Model Performance:")
print(f"Training R² Score: {train_score:.3f}")
print(f"Testing R² Score: {test_score:.3f}")
print("\nFeature Importance:")
print(feature_importance)

# Additional Statistics
print("=== Additional Statistics ===")
print(f"Total number of job postings: {len(df)}")
print(f"Average job age: {df['days_ago'].mean():.1f} days")
print(f"Number of companies posting jobs: {df['company_name'].nunique()}")
print(f"Number of unique locations: {df['location'].nunique()}")

# Export summary to text file
with open('analysis_summary.txt', 'w') as f:
    f.write("Job Market Analysis Summary\n")
    f.write("==========================\n\n")
    
    f.write("1. Top 5 Cities with Most Job Postings\n")
    f.write(str(location_counts) + "\n\n")
    
    f.write("2. Word Cloud of Job Keywords\n")
    f.write("\n")
    
    f.write("3. Job Type Distribution\n")
    f.write(str(work_type_counts) + "\n\n")
    
    f.write("4. Job Popularity vs. Skills Analysis\n")
    f.write("\n")
    
    f.write("5. Sentiment Analysis\n")
    f.write("\nSentiment Distribution:\n")
    f.write(str(sentiment_counts) + "\n\n")
    
    f.write("6. Feature Importance Plot\n")
    f.write("\nFeature Importance:\n")
    f.write(str(feature_importance))

# Update the analysis summary file with new visualizations
with open('analysis_summary.txt', 'a') as f:
    f.write("\n7. Additional Visualization Insights\n")
    f.write("\nJob Type Distribution:\n")
    f.write(str(work_type_counts) + "\n\n")
    
    f.write("Skills Analysis:\n")
    f.write(f"Average number of required skills per job: {df['total_skills'].mean():.1f}\n")
    f.write(f"Maximum number of required skills: {df['total_skills'].max()}\n")
    f.write(f"Minimum number of required skills: {df['total_skills'].min()}\n")

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# Add results to the analysis summary file
with open('analysis_summary.txt', 'a') as f:
    f.write("\n6. Job Popularity Prediction Model\n")
    f.write(f"Training R² Score: {train_score:.3f}\n")
    f.write(f"Testing R² Score: {test_score:.3f}\n")
    f.write("\nFeature Importance:\n")
    f.write(str(feature_importance)) 