import os
import sys
import argparse
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
from collections import Counter
from typing import List, Dict, Tuple, Callable, Optional


########################################
# Utility Functions
########################################


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def check_and_download_glove(
    emb_dir: str,
    emb_filename: str,
    url: str = "https://nlp.stanford.edu/data/glove.6B.zip",
) -> None:
    """
    Check if GloVe embedding file exists; if not, download and extract it.

    Args:
        emb_dir (str): Directory to store embeddings.
        emb_filename (str): GloVe file name to check for (e.g. 'glove.6B.300d.txt').
        url (str): URL to download the GloVe zip from if missing.

    This function ensures that the specified GloVe embedding file is present locally.
    If not, it downloads a zip file from the provided URL, extracts the target file,
    and cleans up the zip file afterward. If any error occurs, the program exits.
    """
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir, exist_ok=True)

    emb_path = os.path.join(emb_dir, emb_filename)
    if os.path.exists(emb_path):
        print("[Info] GloVe file already exists.")
        return

    print(f"[Info] GloVe file not found at {emb_path}. Downloading from: {url}")
    zip_path = os.path.join(emb_dir, "glove.6B.zip")

    # Download the zip file
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"[Error] Failed to download GloVe embeddings: {e}")
        sys.exit(1)

    # Extract the requested file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = [m for m in zip_ref.namelist() if m.endswith(emb_filename)]
            if not members:
                print("[Error] The requested GloVe file was not found in the zip.")
                sys.exit(1)
            zip_ref.extractall(emb_dir, members)
        print("[Info] Extraction complete.")
    except Exception as e:
        print(f"[Error] Failed to extract GloVe embeddings: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def load_glove_embeddings(emb_path: str, emb_dim: int = 300) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a .txt file into a dictionary.

    Args:
        emb_path (str): Path to GloVe embedding file (e.g. 'glove.6B.300d.txt').
        emb_dim (int): Dimension of embeddings.

    Returns:
        Dict[str, np.ndarray]: Mapping from word to its vector representation.

    Raises:
        SystemExit: If the GloVe file is not found.
    """
    if not os.path.exists(emb_path):
        print(f"[Error] GloVe file not found at {emb_path}.")
        sys.exit(1)

    word2vec = {}
    with open(emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            if vec.shape[0] == emb_dim:
                word2vec[word] = vec
    return word2vec


def sentence_embedding(
    sentence: str,
    w2v: Dict[str, np.ndarray],
    emb_dim: int = 300,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute a weighted sentence embedding by averaging word vectors.

    Args:
        sentence (str): Input sentence.
        w2v (Dict[str, np.ndarray]): Word-to-vector mapping.
        emb_dim (int): Embedding dimension.
        weights (Optional[Dict[str, float]]): Optional word-level weights.

    Returns:
        np.ndarray: A single sentence embedding of shape (emb_dim,).
    """
    tokens = sentence.lower().split()
    vecs = []
    w_sum = 0.0
    for t in tokens:
        if t in w2v:
            w = weights[t] if (weights and t in weights) else 1.0
            vecs.append(w2v[t] * w)
            w_sum += w
    if len(vecs) == 0:
        return np.zeros(emb_dim)
    return np.sum(vecs, axis=0) / (w_sum + 1e-9)


def compute_word_frequencies(sentences1: List[str], sentences2: List[str]) -> Counter:
    """
    Compute word frequency from two lists of sentences.

    Args:
        sentences1 (List[str]): List of first set of sentences.
        sentences2 (List[str]): List of second set of sentences.

    Returns:
        Counter: A counter mapping word -> frequency.
    """
    all_tokens = []
    for s1, s2 in zip(sentences1, sentences2):
        all_tokens.extend(s1.lower().split())
        all_tokens.extend(s2.lower().split())
    counter = Counter(all_tokens)
    return counter


########################################
# Transformations
########################################


def baseline_transform(x: np.ndarray) -> np.ndarray:
    """No transformation: returns the input vector as is."""
    return x


def pca_whitening(emb_matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a PCA whitening transform function fitted on the given embedding matrix.

    Args:
        emb_matrix (np.ndarray): Embedding matrix (vocab_size, emb_dim).

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that applies PCA whitening.
    """
    pca = PCA(n_components=emb_matrix.shape[1], whiten=True)
    pca.fit(emb_matrix)

    def transform(x: np.ndarray) -> np.ndarray:
        return pca.transform(x)

    return transform


def zipfian_whitening(
    emb_matrix: np.ndarray, freq: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a Zipfian whitening transform (weighted PCA) based on the given embeddings and frequencies.

    Args:
        emb_matrix (np.ndarray): Embedding matrix (vocab_size, emb_dim).
        freq (np.ndarray): Word frequency array (vocab_size,).

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that applies Zipfian whitening.
    """
    w = freq / freq.sum()
    mean_vec = np.average(emb_matrix, axis=0, weights=w)
    centered = emb_matrix - mean_vec

    cov = np.zeros((emb_matrix.shape[1], emb_matrix.shape[1]))
    for i in range(emb_matrix.shape[0]):
        cov += w[i] * np.outer(centered[i], centered[i])

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    def transform(x: np.ndarray) -> np.ndarray:
        centered_x = x - mean_vec
        x_rot = np.dot(centered_x, eigvecs)
        x_white = x_rot / np.sqrt(eigvals + 1e-9)
        return x_white

    return transform


def sif_weights(word_freq: Dict[str, int], alpha: float = 1e-3) -> Dict[str, float]:
    """
    Compute SIF weights for words.

    Based on Arora et al. (2017), weights are defined as:
    w_i = alpha / (alpha + p(word)), where p(word) = frequency(word)/total_count.

    Args:
        word_freq (Dict[str, int]): A mapping of word -> frequency.
        alpha (float): SIF parameter.

    Returns:
        Dict[str, float]: Mapping of word -> SIF weight.
    """
    total_count = sum(word_freq.values())
    weights = {}
    for w, f in word_freq.items():
        p = f / total_count
        weights[w] = alpha / (alpha + p)
    return weights


########################################
# Evaluation
########################################


def evaluate(
    transform_func: Callable[[np.ndarray], np.ndarray],
    sents1: List[str],
    sents2: List[str],
    scores: List[float],
    w2v: Dict[str, np.ndarray],
    emb_dim: int = 300,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Evaluate correlation with the STS-B gold scores using Pearson and Spearman correlation.

    Args:
        transform_func (Callable): A transformation function for embeddings.
        sents1 (List[str]): First set of sentences.
        sents2 (List[str]): Second set of sentences.
        scores (List[float]): Gold STS scores.
        w2v (Dict[str, np.ndarray]): Word-to-vector mapping.
        emb_dim (int): Embedding dimension.
        weights (Optional[Dict[str, float]]): Word-level weights for sentence embedding.

    Returns:
        (float, float): Pearson correlation, Spearman correlation
    """
    pred_sims = []
    gold_sims = scores
    for sent1, sent2, gs in zip(sents1, sents2, gold_sims):
        v1 = sentence_embedding(sent1, w2v, emb_dim, weights=weights)
        v2 = sentence_embedding(sent2, w2v, emb_dim, weights=weights)
        v1_trans = transform_func(v1.reshape(1, -1))[0]
        v2_trans = transform_func(v2.reshape(1, -1))[0]

        denom = np.linalg.norm(v1_trans) * np.linalg.norm(v2_trans) + 1e-9
        cos_sim = np.dot(v1_trans, v2_trans) / denom
        # Scale similarity to roughly 0-5
        pred_sims.append(cos_sim * 5.0)

    pear = pearsonr(pred_sims, gold_sims)[0]
    spear = spearmanr(pred_sims, gold_sims)[0]
    return pear, spear


def plot_results(
    methods: List[str],
    pearson_scores: List[float],
    spearman_scores: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot bar charts of Pearson and Spearman correlations for each method.

    Args:
        methods (List[str]): Names of the methods.
        pearson_scores (List[float]): Pearson correlations.
        spearman_scores (List[float]): Spearman correlations.
        save_path (Optional[str]): If provided, save the plot to this file.
    """
    x = np.arange(len(methods))
    colors = ["blue", "green", "red", "orange", "purple"]

    plt.figure(figsize=(12, 5))

    # Pearson Plot
    plt.subplot(1, 2, 1)
    plt.bar(x, pearson_scores, color=colors[: len(methods)])
    plt.xticks(x, methods, rotation=45, ha="right")
    plt.title("Pearson Correlation (STS-B Dev)")
    plt.ylim(0, 1.0)
    for i, v in enumerate(pearson_scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    # Spearman Plot
    plt.subplot(1, 2, 2)
    plt.bar(x, spearman_scores, color=colors[: len(methods)])
    plt.xticks(x, methods, rotation=45, ha="right")
    plt.title("Spearman Correlation (STS-B Dev)")
    plt.ylim(0, 1.0)
    for i, v in enumerate(spearman_scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Info] Plot saved to {save_path}")
    else:
        plt.show()


########################################
# Main Execution
########################################


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate various embeddings transformations on STS-B."
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        default="./embeddings/glove.6B.300d.txt",
        help="Path to GloVe embeddings (will download if not found)",
    )
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="If provided, the plot will be saved to this file (e.g. results.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    emb_dir = os.path.dirname(args.emb_path)
    emb_filename = os.path.basename(args.emb_path)

    # Check and download embeddings if needed
    check_and_download_glove(emb_dir, emb_filename)

    # Load embeddings
    print("[Info] Loading embeddings...")
    word2vec = load_glove_embeddings(args.emb_path, args.emb_dim)

    # Load STS-B dataset
    # The 'glue' dataset loader handles downloading if needed.
    print("[Info] Loading STS-B dataset...")
    sts = load_dataset("glue", "stsb")
    train_data = sts["train"]
    dev_data = sts["validation"]

    train_sentences1 = train_data["sentence1"]
    train_sentences2 = train_data["sentence2"]
    train_scores = train_data["label"]

    dev_sentences1 = dev_data["sentence1"]
    dev_sentences2 = dev_data["sentence2"]
    dev_scores = dev_data["label"]

    # Compute word frequencies and prepare embedding matrix
    print("[Info] Computing word frequencies...")
    counter = compute_word_frequencies(train_sentences1, train_sentences2)
    vocab = list(word2vec.keys())
    # Create a frequency array aligned with vocab
    word_freq = {w: counter[w] for w in vocab}
    freq_arr = np.array([word_freq[w] for w in vocab], dtype=float)
    emb_matrix = np.array([word2vec[w] for w in vocab])

    # Prepare transformations
    print("[Info] Preparing transformations...")
    pca_transform = pca_whitening(emb_matrix)
    zipf_transform = zipfian_whitening(emb_matrix, freq_arr)

    # SIF weighting
    print("[Info] Computing SIF weights...")
    sif_w = sif_weights(word_freq)
    sif_weighted_emb = emb_matrix * np.array([sif_w[w] for w in vocab])[:, None]

    # SIF + PCA
    sif_pca_transform = pca_whitening(sif_weighted_emb)

    # Zipfian + SIF + PCA (Weighted PCA on SIF-weighted embeddings)
    zipf_sif_pca_transform = zipfian_whitening(sif_weighted_emb, freq_arr)

    # Evaluate all methods
    print("[Info] Evaluating transformations...")
    # Baseline
    baseline_pear, baseline_spear = evaluate(
        baseline_transform, dev_sentences1, dev_sentences2, dev_scores, word2vec
    )
    # PCA Whitening
    pca_pear, pca_spear = evaluate(
        pca_transform, dev_sentences1, dev_sentences2, dev_scores, word2vec
    )
    # Zipfian Whitening
    zipf_pear, zipf_spear = evaluate(
        zipf_transform, dev_sentences1, dev_sentences2, dev_scores, word2vec
    )
    # SIF + PCA Whitening
    sif_pca_pear, sif_pca_spear = evaluate(
        sif_pca_transform,
        dev_sentences1,
        dev_sentences2,
        dev_scores,
        word2vec,
        weights=sif_w,
    )
    # Zipfian + SIF + PCA Whitening
    zipf_sif_pca_pear, zipf_sif_pca_spear = evaluate(
        zipf_sif_pca_transform,
        dev_sentences1,
        dev_sentences2,
        dev_scores,
        word2vec,
        weights=sif_w,
    )

    # Print results in a neat format
    print("\n[Results]")
    print(
        "Baseline:                Pearson: {:.4f}, Spearman: {:.4f}".format(
            baseline_pear, baseline_spear
        )
    )
    print(
        "PCA Whitening:           Pearson: {:.4f}, Spearman: {:.4f}".format(
            pca_pear, pca_spear
        )
    )
    print(
        "Zipfian Whitening:       Pearson: {:.4f}, Spearman: {:.4f}".format(
            zipf_pear, zipf_spear
        )
    )
    print(
        "SIF + PCA Whitening:     Pearson: {:.4f}, Spearman: {:.4f}".format(
            sif_pca_pear, sif_pca_spear
        )
    )
    print(
        "Zipfian + SIF + PCA Whtn:Pearson: {:.4f}, Spearman: {:.4f}".format(
            zipf_sif_pca_pear, zipf_sif_pca_spear
        )
    )

    # Visualization
    methods = [
        "Baseline",
        "PCA Whitening",
        "Zipfian Whitening",
        "SIF + PCA",
        "Zipf + SIF + PCA",
    ]
    pearson_scores = [
        baseline_pear,
        pca_pear,
        zipf_pear,
        sif_pca_pear,
        zipf_sif_pca_pear,
    ]
    spearman_scores = [
        baseline_spear,
        pca_spear,
        zipf_spear,
        sif_pca_spear,
        zipf_sif_pca_spear,
    ]

    plot_results(methods, pearson_scores, spearman_scores, save_path=args.save_plot)


if __name__ == "__main__":
    main()
