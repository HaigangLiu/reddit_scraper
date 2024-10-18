import numpy as np
from reddit_scraper import reddit_scraper_for_j1
import pandas as pd
from cluster import get_dbscan_clustering, get_embeddings, get_kmeans_clustering, get_hierarchical_clustering
import pickle

RUN_SCRAPER = False  # Set to True to get fresh posts from Reddit
RUN_EMBEDDINGS = False  # Set to True to get embeddings from the text
RUN_CLUSTERING = True  # Set to True to run clustering on the embeddings

FILE_PATH = '/Users/albliu/Downloads/j1_visa_20241007_LLM_label.csv'
EMBEDDING_PATH = '/Users/albliu/Downloads/embeddings_LLM_only_1018.pkl'
FINAL_PATH = '/Users/albliu/Downloads/clustered_reddit_post_1018.csv'

def _make_text(row):
    """
    Concatenate the post title, post body, and top comment into a single string.
    """
    return f"{row['post_title']}\n{row['post_body']}\n{row['top_comment']}"


def run_scraper(subreddits=None,
                keywords=None,
                comment_limit=5,
                file_path=None):

    if subreddits is None:
        subreddits = ["all"]

    if keywords is None:
        keywords = [
            "J1 hotel",
            "J1 visa hotel",
            "J1 visa experience",
            "J1 visa hotel intern",
            "J1 visa hotel summer",
            "J1 visa hotel student",
            "J1 visa hotel work"
        ]

    if file_path is None:
        file_path = FILE_PATH

    df = pd.DataFrame(reddit_scraper_for_j1(subreddits=subreddits,
                                            keywords=keywords,
                                            comment_limit=comment_limit))
    # Concatenate text for clustering
    df['text'] = df.apply(_make_text, axis=1)

    df.to_csv(file_path, index=False)
    print('File saved to', file_path)
    return df


def run_embeddings(df, col, file_path, model_name=None):
    if model_name is None:
        model_name = 'bert-base-uncased'

    if file_path is None:
        file_path = EMBEDDING_PATH

    # Reset index to ensure consistent ordering
    df = df.reset_index(drop=True)

    list_of_passages = df[col].tolist()
    embedding_from_text = get_embeddings(list_of_passages,
                                         model_name=model_name)

    with open(file_path, 'wb') as f:
        pickle.dump(embedding_from_text, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Embeddings saved to', file_path)
    return embedding_from_text


def run_clustering(embedding_from_text,
                   df,
                   file_path=None):

    # Ensure DataFrame order is consistent
    df = df.reset_index(drop=True)

    labels_1 = get_dbscan_clustering(embedding_from_text,
                                     eps=2.5,
                                     min_samples=5)
    labels_2 = get_dbscan_clustering(embedding_from_text,
                                     eps=3,
                                     min_samples=5)

    labels_3, _ = get_kmeans_clustering(embedding_from_text, num_clusters=3)
    labels_4, _ = get_kmeans_clustering(embedding_from_text, num_clusters=5)
    labels_5, _ = get_kmeans_clustering(embedding_from_text, num_clusters=7)

    labels_6 = get_hierarchical_clustering(embedding_from_text, num_clusters=3)
    labels_7 = get_hierarchical_clustering(embedding_from_text, num_clusters=5)
    labels_8 = get_hierarchical_clustering(embedding_from_text, num_clusters=7)

    # Assign labels to DataFrame
    df['label_dbscan_v1'] = labels_1
    df['label_dbscan_v2'] = labels_2
    df['label_kmeans_3_clusters'] = labels_3
    df['label_kmeans_5_clusters'] = labels_4
    df['label_kmeans_7_clusters'] = labels_5
    df['label_hierarchical_3_clusters'] = labels_6
    df['label_hierarchical_5_clusters'] = labels_7
    df['label_hierarchical_7_clusters'] = labels_8

    # Display value counts for each clustering label
    for k in ['label_dbscan_v1',
              'label_dbscan_v2',
              'label_kmeans_3_clusters',
              'label_kmeans_5_clusters',
              'label_kmeans_7_clusters',
              'label_hierarchical_3_clusters',
              'label_hierarchical_5_clusters',
              'label_hierarchical_7_clusters']:

        print(f"{k} value counts:")
        print(df[k].value_counts())

    if file_path is None:
        file_path = FINAL_PATH
    df.to_csv(file_path, index=False)
    print('File saved to', file_path)


if __name__ == '__main__':
    if RUN_SCRAPER:
        df = run_scraper()
    else:
        df = pd.read_csv(FILE_PATH)
        # only process these verified by LLM
        df = df[df['LLM_label'] == 1]

    if RUN_EMBEDDINGS:
        embedding_from_text = run_embeddings(df,
                                             col='text',
                                             file_path=EMBEDDING_PATH)
    else:
        with open(EMBEDDING_PATH, 'rb') as f:
            embedding_from_text = pickle.load(f)

    if RUN_CLUSTERING:
        run_clustering(embedding_from_text, df, file_path=FINAL_PATH)
