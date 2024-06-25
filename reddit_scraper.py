from time import sleep

import praw
import pandas as pd


def search_reddit_for_j1_experiences(subreddits, keywords, post_limit=100):
    """
    Search specified subreddits for given keywords related to J-1 visa experiences.

    Parameters:
    - subreddits: list of str. Subreddits to search.
    - keywords: list of str. Keyword queries to use.
    - post_limit: int. Maximum number of posts to fetch per keyword per subreddit.
    """
    reddit = praw.Reddit(
        client_id='WYsQeCzYURn1oNP63gdhFQ',
        client_secret='YkjB6KKcpyzE8NZY1SuT4l9d-zC6Uw',
        user_agent='script:sentiment-j1:1.0 (by /u/Immediate-Bet4542)'
    )

    results = []
    for subreddit in subreddits:
        sub = reddit.subreddit(subreddit)
        for keyword in keywords:
            for submission in sub.search(keyword, limit=post_limit):
                print(f"found post in r/{subreddit} for keyword '{keyword}'")
                sleep(1)
                results.append({
                    'search_string': keyword,
                    'subreddit': subreddit,
                    'post_title': submission.title,
                    'post_body': submission.selftext,
                    'post_url': submission.url,
                    'top_comment': submission.comments[0].body if submission.comments else ""
                })
    return results

def create_dataframe_from_results(results):
    """
    Create a pandas DataFrame from the list of results.

    Parameters:
    - results: list of dicts with search result data.
    """
    return pd.DataFrame(results)

# Define your subreddits and keywords
subreddits = ['immigration', 'AskAnAmerican',
              'workandtravel', 'studyAbroad',
              'hospitality', 'internships', 'jobs']
keywords = [
    '"J-1 visa" "hotel experience"',
    '"J-1 visa" "hospitality"',
    '"exchange visitor" "hospitality experience"',
    '"summer work travel" "hotel"',
    '"J-1 visa" "seasonal work" "hospitality"',
    '"J-1 visa" "hotel internship" "US"',
    '"J-1 visa" "resort job" "experience"',
    '"international students" "hotel jobs"',
    '"J-1 visa" "work and travel" "hotel"',
    '"cultural exchange" "hotel work"',
    '"J-1 visa stories" "hotel"',
    '"J-1 visa experiences" "hospitality"',
    '"J-1 visa" "working in the US" "hotels"',
    '"J-1 visa" "customer service" "experience"',
    '"J-1 visa" "culinary internship"',
    '"J-1 visa" "culinary experiences"',
    '"J-1 visa" "guest services"',
    '"J-1 visa" "management trainee"',
    '"J-1 visa" "business internship"',
    '"J-1 visa traineeship" "hospitality"',
    '"J-1 visa" "international workers"',
    '"J-1 visa" "work exchange" "hotel"',
    '"J-1 visa" "cultural exchange" "hospitality"',
    '"J-1 visa" "food and beverage" "internship"',
    '"J-1 visa" "housekeeping" "internship"',
    '"J-1 visa" "hotel management"',
    '"J-1 visa" "hotel operations"',
    '"J-1 visa" "event planning"',
    '"J-1 visa" "resort management"',
    '"J-1 visa" "tourism industry" "experiences"'
]

# Fetch results and create a DataFrame
results = search_reddit_for_j1_experiences(subreddits, keywords)
df = create_dataframe_from_results(results)

# Save the DataFrame to a CSV file
file_path = '/Users/albliu/Downloads/j1_visa_experiences.csv'
df.to_csv(file_path, index=False)
print("DataFrame saved successfully to CSV.")
print(df)
