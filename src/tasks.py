import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_style('whitegrid')


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and merging red and white wine datasets.

    Note: Don't forget to add the 'type' attribute.
    """
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    red['type'] = 'red'
    white['type'] = 'white'
    data = pd.concat([red, white], ignore_index=True)
    if data.empty:
        raise ValueError("The loaded wine dataset is empty")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(
        data[numeric_columns].median()
    )
    data['type'] = data['type'].fillna('unknown')
    return data


def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from a dataset using the IQR method."""
    df_clean = df.copy()
    if isinstance(features, pd.DataFrame):
        feature_names = list(features.columns)
    else:
        feature_names = list(features)

    for feature in feature_names:
        if feature not in df_clean.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[feature]):
            continue
        q1 = df_clean[feature].quantile(0.25)
        q3 = df_clean[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[
            (df_clean[feature] >= lower_bound)
            & (df_clean[feature] <= upper_bound)
        ]
    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encoding of categorical variables."""
    data_encoded = data.copy()
    if 'type' in data_encoded.columns:
        data_encoded = pd.get_dummies(
            data_encoded,
            columns=['type'],
            prefix='type',
            dtype=int
        )
    return data_encoded


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Standartization (scaling) of features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train_scaled


def apply_pca(X: pd.DataFrame) -> np.ndarray:
    """
    Application of the principal component analysis (PCA) method.
    """
    X_scaled = scale_features(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def perform_clustering(X_pca: np.ndarray) -> dict[str, float | np.ndarray]:
    """Performing clustering based on PCA results."""
    X_pca = np.asarray(X_pca)
    if X_pca.shape[0] < 2:
        raise ValueError("At least 2 samples are required for clustering")

    best_score = -1.0
    best_labels = np.array([], dtype=int)
    for n_clusters in range(2, 11):
        if n_clusters >= X_pca.shape[0]:
            break
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    results = {
        'labels': best_labels,
        'silhouette_score': float(best_score)
    }
    return results


def main():
    # Loading data
    data = load_data('winequality-red.csv', 'winequality-white.csv')

    # Removing outliers
    features = data.columns[:-2]  # Without 'quality' and 'type'
    data = remove_outliers(data, features)

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Unsupervised learning task
    X_features = data.drop(['quality'], axis=1)
    X_pca = apply_pca(X_features)
    clustering_results = perform_clustering(X_pca)
    print("\nClustering Results:")
    print(f"Silhouette coefficient: {clustering_results['silhouette_score']:.4f}")

    # Visualization of clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    hue=clustering_results['labels'],
                    palette='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
