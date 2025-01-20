#Import dependencies
import requests
from bs4 import BeautifulSoup
import random
import pandas as pd

title = ""  # Job title
location = ""  # Job location
start = 0  # Starting point for pagination

# Initialize an empty list to store all job IDs
id_list = []

# Loop through multiple pages (e.g., first 10 pages = 250 jobs)
for page in range(3):  # Adjust this number to get more or fewer pages
    start = page * 25  # Each page has 25 jobs
    
    # Construct the URL for LinkedIn job search with updated start parameter
    list_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={title}&location={location}&start={start}"
    
    # Send a GET request to the URL and store the response
    response = requests.get(list_url)
    
    # Only process pages with successful responses
    if response.status_code != 200:
        print(f"Failed to fetch page {page + 1} - Status code: {response.status_code}")
        continue
    
    # Get the HTML, parse the response and find all list items(jobs postings)
    list_data = response.text
    list_soup = BeautifulSoup(list_data, "html.parser")
    page_jobs = list_soup.find_all("li")
    
    # Break the loop if no more jobs are found
    if not page_jobs:
        print(f"No more jobs found after page {page}")
        break
        
    # Iterate through job postings to find job ids
    for job in page_jobs:
        try:
            base_card_div = job.find("div", {"class": "base-card"})
            job_id = base_card_div.get("data-entity-urn").split(":")[3]
            print(f"Found job ID: {job_id} on page {page + 1}")
            
            # Get individual job details and verify response code
            job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
            job_response = requests.get(job_url)
            
            if job_response.status_code == 200:
                id_list.append(job_id)
            else:
                print(f"Skipping job {job_id} - Status code: {job_response.status_code}")
                
        except Exception as e:
            print(f"Error processing job on page {page + 1}: {str(e)}")
            continue

print(f"Total valid jobs found: {len(id_list)}")

# Initialize an empty list to store job information
job_list = []

# Loop through the list of job IDs and get each URL
for job_id in id_list:
    # Construct the URL for each job using the job ID
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    
    # Send a GET request to the job URL and parse the reponse
    job_response = requests.get(job_url)
    print(job_response.status_code)
    job_soup = BeautifulSoup(job_response.text, "html.parser")
    
     # Create a dictionary to store job details
    job_post = {}
    
    # Try to extract and store the job title
    try:
        job_post["job_title"] = job_soup.find("h2", {"class":"top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title"}).text.strip()
    except:
        job_post["job_title"] = None
        
    # Try to extract and store the company name
    try:
        job_post["company_name"] = job_soup.find("a", {"class": "topcard__org-name-link topcard__flavor--black-link"}).text.strip()
    except:
        job_post["company_name"] = None
        
    # Try to extract and store the time posted
    try:
        job_post["time_posted"] = job_soup.find("span", {"class": "posted-time-ago__text posted-time-ago__text--new topcard__flavor--metadata"}).text.strip()
    except:
        job_post["time_posted"] = None
        
    # Try to extract and store the number of applications
    try:
        job_post["num_applications"] = job_soup.find("span", {"class": "num-applicants__caption"}).text.strip()
    except:
        job_post["num_applications"] = "<25"
    
    # Try to extract and store the location
    try:
        job_post["location"] = job_soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"}).text.strip()
    except:
        job_post["location"] = None
    
    # Try to extract and store the job description
    try:
        job_post["description"] = job_soup.find("div", {"class": "show-more-less-html__markup"}).text.strip()
    except:
        job_post["description"] = None
        
    # Try to extract and store the skills/requirements
    try:
        skills_section = job_soup.find("div", {"class": "description__text description__text--rich"})
        # Look for common headers that might contain skills/requirements
        headers = skills_section.find_all(["h3", "h4"])
        requirements_text = ""
        for header in headers:
            if any(keyword in header.text.lower() for keyword in ["requirement", "qualification", "skill"]):
                # Get the text content following this header until the next header
                current = header.next_sibling
                while current and not current.name in ["h3", "h4"]:
                    if hasattr(current, "text"):
                        requirements_text += current.text.strip() + "\n"
                    current = current.next_sibling
        job_post["requirements"] = requirements_text.strip() if requirements_text else None
    except:
        job_post["requirements"] = None
        
    # Append the job details to the job_list
    job_list.append(job_post)
    
#Check if the list contains all the desired data
job_list

# Create a pandas DataFrame using the list of job dictionaries 'job_list'
jobs_df = pd.DataFrame(job_list)
jobs_df

# 1. Clean the data
# Remove duplicates
jobs_df = jobs_df.drop_duplicates()

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    # Keep alphanumeric and spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

# Apply text cleaning to relevant columns
text_columns = ['job_title', 'description', 'requirements']
for col in text_columns:
    jobs_df[f'clean_{col}'] = jobs_df[col].apply(clean_text)

# 2. Extract key information from job descriptions
# Define keywords to look for
tech_keywords = {
    'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'sql', 'r'],
    'tools': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp'],
    'skills': ['machine learning', 'ai', 'data science', 'deep learning', 'nlp']
}

# Function to check for keywords in text
def find_keywords(text, keyword_list):
    if pd.isna(text):
        return []
    text = str(text).lower()
    return [keyword for keyword in keyword_list if keyword in text]

# Create new columns for each keyword category
for category, keywords in tech_keywords.items():
    jobs_df[f'{category}_keywords'] = jobs_df['clean_description'].apply(
        lambda x: find_keywords(x, keywords)
    )

# 3. Add derived columns
# Convert time_posted to days ago
def extract_days(time_text):
    if pd.isna(time_text):
        return None
    time_text = str(time_text).lower()
    if 'hour' in time_text or 'hr' in time_text:
        return 0
    elif 'day' in time_text:
        return int(''.join(filter(str.isdigit, time_text)))
    elif 'week' in time_text:
        return int(''.join(filter(str.isdigit, time_text))) * 7
    elif 'month' in time_text:
        return int(''.join(filter(str.isdigit, time_text))) * 30
    return None

jobs_df['days_ago'] = jobs_df['time_posted'].apply(extract_days)

# Identify remote/hybrid/in-office
def get_work_type(text):
    if pd.isna(text):
        return 'unknown'
    text = str(text).lower()
    if 'remote' in text:
        if 'hybrid' in text:
            return 'hybrid'
        return 'remote'
    elif 'hybrid' in text:
        return 'hybrid'
    elif 'in office' in text or 'on-site' in text or 'onsite' in text:
        return 'in-office'
    return 'unknown'

jobs_df['work_type'] = jobs_df['clean_description'].apply(get_work_type)

# Display some statistics
print("\nDataset Statistics:")
print(f"Total number of jobs: {len(jobs_df)}")
print(f"Number of remote jobs: {len(jobs_df[jobs_df['work_type'] == 'remote'])}")
print(f"Number of hybrid jobs: {len(jobs_df[jobs_df['work_type'] == 'hybrid'])}")
print(f"Average job posting age: {jobs_df['days_ago'].mean():.1f} days")

# Most common programming languages
programming_counts = jobs_df['programming_keywords'].explode().value_counts()
print("\nMost common programming languages:")
print(programming_counts)

# Remove rows that are completely empty or have only '<25' in 'num_applications'
jobs_df = jobs_df.dropna(how='all')  # Drop rows where all elements are NaN
jobs_df = jobs_df[~((jobs_df.isna() | (jobs_df == '<25')).all(axis=1))]

# Save the enriched dataset
jobs_df.to_csv('enriched_jobs.csv', index=False)