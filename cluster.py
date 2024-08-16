from sklearn.manifold import TSNE
import random
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import mode
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import torch
import matplotlib.pyplot as plt

# available models
available_models = [
    'bert-base-uncased',
    'bert-large-uncased',
    'roberta-large',
    'gpt2-large',
    't5-large']


def create_sample_phrases():
    """
    Create sample phrases with labels for testing the clustering algorithm.
    :return: A list of tuples where each tuple contains a phrase and its label
    """
    categories = {
        'Ancient Rome': [
            "As dawn breaks, the ancient Roman forum comes alive with the bustling sounds of merchants and "
            "philosophers.",
            "Under the reign of Caesar, the architectural marvel of the aqueducts brought water to the farthest "
            "corners of Rome.",
            "The Senate chamber, with its heavy air of intrigue and power, decided the fates of men and nations alike.",
            "At the heart of Rome, the mighty Colosseum echoed with the roars of lions and the cheers of the eager "
            "crowd.",
            "The dusty roads leading to the Roman Empire's outer provinces were trodden by legionnaires and traders."
        ],
        'Music': [
            "A lone violinist fills the night air with melodies that resonate through the cobblestone streets of "
            "Vienna.",
            "The rise of rock 'n' roll in the 1950s America changed the cultural landscape with its rebellious energy.",
            "In a small New Orleans club, a jazz band weaves complex rhythms that capture the essence of the city’s "
            "vibrant history.",
            "A grand piano sits center stage at Carnegie Hall, awaiting the virtuoso whose fingers bring life to "
            "Chopin’s nocturnes.",
            "The electronic beats of a Berlin DJ set pulse through the crowd, defining modern musical movements."
        ],
        'Chess': [
            "In a quiet library, two old friends silently battle over a chessboard, each move heavy with years of "
            "rivalry.",
            "The world chess championship unfolds in Moscow, with grandmasters from across the globe vying for "
            "supremacy.",
            "A young prodigy studies the Queen’s Gambit, dreaming of defeating seasoned players with her strategic "
            "prowess.",
            "Chess pieces move with precision across the board in a high-stakes game that draws a crowd at the park.",
            "Historic chess tournaments are revisited through aged books that recount the victories and defeats of "
            "chess legends."
        ]
    }

    labeled_phrases = [(phrase, label) for label, phrases in categories.items() for phrase in phrases]
    random.shuffle(labeled_phrases)
    return labeled_phrases


def get_embeddings(sentences, model_name='bert-base-uncased'):
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "t5" in model_name:
        model = T5EncoderModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    # Set the pad_token for tokenizers that need it (e.g., GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize sentences and prepare input tensors
    inputs = tokenizer(sentences,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        output_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    return output_embeddings


# Example usage

def get_kmeans_clustering(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_

def get_dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(embeddings)
    return dbscan.labels_  # DBSCAN labels

def evaluate_clustering(labels_pred, labels_true,
                        print_report=True,
                        print_confusion=True):
    """
    Evaluate the clustering output against true labels.
    :param print_confusion:
    :param print_report:
    :param labels_pred: Predicted labels from clustering
    :param labels_true: True labels
    :return: None
    """
    if print_confusion:
        print("Confusion Matrix:")
        print(confusion_matrix(labels_true, labels_pred))
    if print_report:
        print("\nClassification Report:")
        print(classification_report(labels_true, labels_pred))


def align_cluster_labels(cluster_labels, true_labels):
    """
    Aligns predicted cluster labels with true labels
    assuming that the predicted labels are not terribly wrong and
    the mode should match across predicted and true labels.

    :param cluster_labels: Cluster labels from the clustering algorithm.
    :param true_labels: True labels for each data point.
    :return: Aligned labels as numpy array.
    """
    unique_clusters = np.unique(cluster_labels)
    label_map = {}
    for cluster in unique_clusters:
        # Find indices where cluster_labels matches the current cluster
        indices = np.where(cluster_labels == cluster)[0]
        # Use these indices to find the corresponding true labels
        cluster_true_labels = np.array(true_labels)[indices]
        # Find the mode of the true labels
        mode_label = mode(cluster_true_labels)[0][0]
        label_map[cluster] = mode_label

    # Apply the mapping to align labels
    aligned_labels = np.array([label_map[label] for label in cluster_labels])
    return aligned_labels


def apply_tsne(embeddings,
               perplexity=10.0,
               verbose=1,
               n_iter=300,
               n_components=2):
    # Initialize t-SNE
    tsne = TSNE(n_components=n_components,
                verbose=verbose,
                perplexity=perplexity,
                n_iter=n_iter)
    # Reduce dimensions
    tsne_results = tsne.fit_transform(embeddings)
    return tsne_results


def plot_tsne(tsne_results, labels=None):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', cmap='viridis')

    # Annotate points if labels are provided
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]))

    # Set labels and title
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Text Embeddings')
    ax.grid(True)
    plt.show()
    return ax


# generate sample data and label
labeled_sentences = create_sample_phrases()
sentences, true_labels = zip(*labeled_sentences)
label_mapping = {'Ancient Rome': 0, 'Music': 1, 'Chess': 2}
numerical_true_labels = [label_mapping[label] for label in true_labels]

PRINT_EMBEDDINGS = True
if PRINT_EMBEDDINGS:
    for model in ['bert-base-uncased']:
        print(f"Model: {model}")
        embeddings = get_embeddings(sentences, model_name=model)
        print(embeddings)
        print(embeddings.shape)

RUN_KMEANS = False
RUN_tSNE = True
RUN_DBSCAN = False
# Evaluate clustering
if RUN_KMEANS:
    for model in ['bert-base-uncased']:
        print(f"\nEvaluating clustering for model: {model}")
        labels, _ = get_kmeans_clustering(get_embeddings(sentences,
                                                         model_name=model),
                                          num_clusters=3)
        labels = align_cluster_labels(labels, numerical_true_labels)

        evaluate_clustering(labels, numerical_true_labels,
                            print_confusion=True,
                            print_report=False)

if RUN_DBSCAN:
    for model in ['bert-base-uncased']:
        print(f"\nEvaluating clustering for model: {model}")
        labels = get_dbscan_clustering(get_embeddings(sentences, model_name=model), eps=0.5, min_samples=5)
        labels = align_cluster_labels(labels, numerical_true_labels)

        # Filtering noise points, if needed
        filtered_labels = labels[labels != -1]
        filtered_true_labels = np.array(numerical_true_labels)[labels != -1]

        evaluate_clustering(filtered_labels, filtered_true_labels, print_confusion=True, print_report=False)

if RUN_tSNE:
    # Visualize the embeddings using t-SNE
    embeddings = get_embeddings(sentences,
                                model_name='bert-base-uncased')
    tsne_results = apply_tsne(embeddings,
                              perplexity=1.5,
                              n_components=2,
                              verbose=1,
                              n_iter=1000,
                              )
    ax, scatter = plot_tsne(tsne_results, labels=sentences)

    # Customize the scatter points further, such as changing their sizes
    scatter.set_sizes([50 for _ in range(len(tsne_results))])  # Increase the size of all points
    scatter.set_cmap('plasma')

