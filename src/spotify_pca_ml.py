from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "spotify_tracks_selected.csv"
IMAGES_DIR = ROOT_DIR / "images"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SELECTED_GENRES = ["Pop", "Rock", "Hip-Hop", "Electronic", "Classical"]
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "popularity",
]
GENRE_PALETTE = {
    "Pop": "#FF6B6B",
    "Rock": "#4D96FF",
    "Hip-Hop": "#FFD93D",
    "Electronic": "#6BCB77",
    "Classical": "#845EC2",
}


@dataclass
class PCAArtifacts:
    standardized_features: pd.DataFrame
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_percent: np.ndarray
    cumulative_variance_percent: np.ndarray
    scores: pd.DataFrame
    loadings: pd.DataFrame
    contributions: pd.DataFrame
    variable_quality: pd.DataFrame
    centroids: pd.DataFrame


def ensure_directories() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe["genre"] = pd.Categorical(dataframe["genre"], categories=SELECTED_GENRES, ordered=True)
    return dataframe.sort_values("genre").reset_index(drop=True)


def standardize_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataframe[FEATURE_COLUMNS])
    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=dataframe.index)


def compute_pca_artifacts(dataframe: pd.DataFrame) -> PCAArtifacts:
    standardized = standardize_features(dataframe)
    correlation_matrix = pd.DataFrame(
        (standardized.T @ standardized) / len(standardized),
        index=FEATURE_COLUMNS,
        columns=FEATURE_COLUMNS,
    ).round(4)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained_variance_percent = eigenvalues / eigenvalues.sum() * 100
    cumulative_variance_percent = np.cumsum(explained_variance_percent)

    scores_array = standardized.values @ eigenvectors
    scores = pd.DataFrame(
        scores_array,
        columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))],
        index=dataframe.index,
    )
    scores["genre"] = dataframe["genre"].astype(str)

    loadings_array = eigenvectors * np.sqrt(eigenvalues)
    loadings = pd.DataFrame(
        loadings_array,
        index=FEATURE_COLUMNS,
        columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))],
    )
    contributions_array = (loadings_array**2 / np.sum(loadings_array**2, axis=0)) * 100
    variable_quality_array = loadings_array**2 / np.sum(loadings_array**2, axis=1, keepdims=True)
    contributions = pd.DataFrame(
        contributions_array,
        index=FEATURE_COLUMNS,
        columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))],
    )
    variable_quality = pd.DataFrame(
        variable_quality_array,
        index=FEATURE_COLUMNS,
        columns=[f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))],
    )

    centroids = scores.groupby("genre")[["PC1", "PC2"]].mean().reindex(SELECTED_GENRES)

    return PCAArtifacts(
        standardized_features=standardized,
        correlation_matrix=correlation_matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_percent=explained_variance_percent,
        cumulative_variance_percent=cumulative_variance_percent,
        scores=scores,
        loadings=loadings,
        contributions=contributions,
        variable_quality=variable_quality,
        centroids=centroids,
    )


def get_experiment_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression - Original Features": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=4000))]
        ),
        "Logistic Regression - PCA (2 components)": Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=2, random_state=42)), ("clf", LogisticRegression(max_iter=4000))]
        ),
        "Logistic Regression - PCA (95% variance)": Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", LogisticRegression(max_iter=4000))]
        ),
        "KNN - Original Features": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=11))]),
        "KNN - PCA (2 components)": Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=2, random_state=42)), ("clf", KNeighborsClassifier(n_neighbors=11))]
        ),
        "KNN - PCA (95% variance)": Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", KNeighborsClassifier(n_neighbors=11))]
        ),
        "Random Forest - Original Features": Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]
        ),
        "Random Forest - PCA (2 components)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=2, random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        ),
        "Random Forest - PCA (95% variance)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        ),
    }


def run_model_experiments(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, Pipeline], tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X = dataframe[FEATURE_COLUMNS]
    y = dataframe["genre"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: list[dict[str, object]] = []
    reports: dict[str, str] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for experiment_name, pipeline in get_experiment_pipelines().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        pca_step = pipeline.named_steps.get("pca")
        n_components = getattr(pca_step, "n_components_", "original")
        results.append(
            {
                "experiment": experiment_name,
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "n_components_used": n_components,
            }
        )
        reports[experiment_name] = classification_report(y_test, predictions, digits=4)
        fitted_pipelines[experiment_name] = pipeline

    results_frame = pd.DataFrame(results).sort_values(["accuracy", "macro_f1"], ascending=False).reset_index(drop=True)
    return results_frame, reports, fitted_pipelines, (X_train, X_test, y_train, y_test)


def save_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Standardized Spotify Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_explained_variance_plot(artifacts: PCAArtifacts, output_path: Path) -> None:
    labels = [f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))]
    fig, ax1 = plt.subplots(figsize=(11, 6))

    bars = ax1.bar(labels, artifacts.explained_variance_percent, color="#4D96FF", alpha=0.85)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Principal Components", fontsize=12, fontweight="bold")
    ax1.set_title("Explained Variance by Principal Component", fontsize=15, fontweight="bold")

    for i, bar in enumerate(bars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{artifacts.explained_variance_percent[i]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = ax1.twinx()
    ax2.plot(labels, artifacts.cumulative_variance_percent, color="#FF6B6B", marker="o", linewidth=2.5)
    ax2.axhline(95, color="#6BCB77", linestyle="--", linewidth=1.5, label="95% threshold")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pca_scatter_plot(artifacts: PCAArtifacts, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    plotting_frame = artifacts.scores.copy()

    for genre in SELECTED_GENRES:
        genre_rows = plotting_frame[plotting_frame["genre"] == genre]
        ax.scatter(
            genre_rows["PC1"],
            genre_rows["PC2"],
            s=22,
            alpha=0.40,
            color=GENRE_PALETTE[genre],
            label=genre,
            edgecolors="none",
        )

    for genre, row in artifacts.centroids.iterrows():
        ax.scatter(row["PC1"], row["PC2"], color=GENRE_PALETTE[genre], marker="X", s=230, edgecolors="black", linewidth=0.8)
        ax.annotate(genre, (row["PC1"], row["PC2"]), xytext=(8, 8), textcoords="offset points", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"PC1 ({artifacts.explained_variance_percent[0]:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({artifacts.explained_variance_percent[1]:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_title("PCA Projection of Songs by Genre", fontsize=15, fontweight="bold")
    ax.legend(title="Genre")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_correlation_circle(artifacts: PCAArtifacts, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    circle = plt.Circle((0, 0), 1, fill=False, color="#4D96FF", linewidth=1.8)
    ax.add_artist(circle)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    pc1 = artifacts.loadings["PC1"]
    pc2 = artifacts.loadings["PC2"]
    label_offsets = {
        "danceability": (-0.10, -0.03),
        "valence": (-0.05, -0.03),
        "popularity": (-0.02, -0.03),
        "liveness": (-0.03, 0.03),
        "tempo": (-0.02, 0.02),
    }
    for feature in FEATURE_COLUMNS:
        ax.arrow(
            0,
            0,
            pc1.loc[feature],
            pc2.loc[feature],
            head_width=0.03,
            head_length=0.04,
            linewidth=1.5,
            color="#FF6B6B",
            length_includes_head=True,
        )
        offset_x, offset_y = label_offsets.get(feature, (0.0, 0.0))
        ax.text(pc1.loc[feature] * 1.08 + offset_x, pc2.loc[feature] * 1.08 + offset_y, feature, fontsize=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"PC1 ({artifacts.explained_variance_percent[0]:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({artifacts.explained_variance_percent[1]:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_title("Correlation Circle of the Audio Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(results_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = results_frame.copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_frame["experiment_label"] = plot_frame["experiment"]
    sns.barplot(
        data=plot_frame,
        x="accuracy",
        y="experiment",
        hue="experiment_label",
        dodge=False,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold")

    for i, value in enumerate(plot_frame["accuracy"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_logistic_confusion_matrices(
    fitted_pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    selected_experiments = [
        "Logistic Regression - Original Features",
        "Logistic Regression - PCA (95% variance)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, experiment_name in zip(axes, selected_experiments):
        predictions = fitted_pipelines[experiment_name].predict(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=SELECTED_GENRES)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=SELECTED_GENRES,
            yticklabels=SELECTED_GENRES,
            ax=ax,
        )
        ax.set_title(experiment_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Genre")
        ax.set_ylabel("True Genre")

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_pca_tables(artifacts: PCAArtifacts) -> None:
    pca_summary = pd.DataFrame(
        {
            "eigenvalue": artifacts.eigenvalues.round(4),
            "explained_variance_percent": artifacts.explained_variance_percent.round(2),
            "cumulative_variance_percent": artifacts.cumulative_variance_percent.round(2),
        },
        index=[f"PC{i + 1}" for i in range(len(FEATURE_COLUMNS))],
    )
    pca_summary.to_csv(OUTPUTS_DIR / "pca_summary.csv")
    artifacts.correlation_matrix.to_csv(OUTPUTS_DIR / "correlation_matrix.csv")
    artifacts.standardized_features.to_csv(OUTPUTS_DIR / "standardized_features.csv", index=False)
    artifacts.loadings.round(4).to_csv(OUTPUTS_DIR / "pca_loadings.csv")
    artifacts.contributions.round(4).to_csv(OUTPUTS_DIR / "pca_feature_contributions.csv")
    artifacts.variable_quality.round(4).to_csv(OUTPUTS_DIR / "pca_feature_quality_cos2.csv")
    artifacts.centroids.round(4).to_csv(OUTPUTS_DIR / "genre_centroids_pc1_pc2.csv")

    scores_export = artifacts.scores[["PC1", "PC2", "genre"]].round(4)
    scores_export.to_csv(OUTPUTS_DIR / "pca_scores_pc1_pc2.csv", index=False)


def export_model_outputs(results_frame: pd.DataFrame, reports: dict[str, str]) -> None:
    results_frame.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    report_lines: list[str] = []
    for experiment_name, report in reports.items():
        report_lines.append(experiment_name)
        report_lines.append(report)
        report_lines.append("-" * 80)
    (OUTPUTS_DIR / "classification_reports.txt").write_text("\n".join(report_lines))


def print_summary(artifacts: PCAArtifacts, results_frame: pd.DataFrame) -> None:
    top_pc1 = artifacts.contributions["PC1"].sort_values(ascending=False).head(5).round(2)
    top_pc2 = artifacts.contributions["PC2"].sort_values(ascending=False).head(5).round(2)

    print("Spotify PCA + ML project completed.")
    print(f"Dataset size: {len(load_dataset())} tracks")
    print(f"Selected genres: {', '.join(SELECTED_GENRES)}")
    print(f"PC1 explained variance: {artifacts.explained_variance_percent[0]:.2f}%")
    print(f"PC2 explained variance: {artifacts.explained_variance_percent[1]:.2f}%")
    print(f"Cumulative variance of PC1 + PC2: {artifacts.cumulative_variance_percent[1]:.2f}%")
    print(f"Components needed to retain at least 95% variance: {(artifacts.cumulative_variance_percent < 95).sum() + 1}")
    print()
    print("Top contributors to PC1:")
    print(top_pc1.to_string())
    print()
    print("Top contributors to PC2:")
    print(top_pc2.to_string())
    print()
    print("Best model results:")
    print(results_frame.head(5).to_string(index=False))


def main() -> None:
    ensure_directories()
    dataframe = load_dataset()
    pca_artifacts = compute_pca_artifacts(dataframe)
    results_frame, reports, fitted_pipelines, split_data = run_model_experiments(dataframe)
    _, X_test, _, y_test = split_data

    export_pca_tables(pca_artifacts)
    export_model_outputs(results_frame, reports)

    save_correlation_heatmap(pca_artifacts.correlation_matrix, IMAGES_DIR / "correlation_heatmap.png")
    save_explained_variance_plot(pca_artifacts, IMAGES_DIR / "explained_variance.png")
    save_pca_scatter_plot(pca_artifacts, IMAGES_DIR / "pca_scatter.png")
    save_correlation_circle(pca_artifacts, IMAGES_DIR / "correlation_circle.png")
    save_model_comparison_plot(results_frame, IMAGES_DIR / "model_accuracy_comparison.png")
    save_logistic_confusion_matrices(fitted_pipelines, X_test, y_test, IMAGES_DIR / "confusion_matrices_logistic.png")

    print_summary(pca_artifacts, results_frame)


if __name__ == "__main__":
    main()
