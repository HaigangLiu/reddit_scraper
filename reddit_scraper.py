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
                results.append({
                    'search_string': keyword,
                    'subreddit': subreddit,
                    'post_title': submission.title,
                    'post_body': submission.selftext,
                    'post_url': f"https://www.reddit.com{submission.permalink}",
                    'top_comment': submission.comments[0].body if submission.comments else ""
                })
    return results


# Define your subreddits and keywords
subreddits = ["all"]
keywords = [
    "J1 hotel",
    "J1 visa hotel",
    "J1 visa experience",
    "J1 visa hotel intern",
    "J1 visa hotel summer",
    "J1 visa hotel student",
    "J1 visa hotel work"
]
# Fetch results and create a DataFrame
results = search_reddit_for_j1_experiences(subreddits, keywords)
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
file_path = '/Users/albliu/Downloads/j1_visa_experiences_v2.csv'
df.to_csv(file_path, index=False)
print(df)
