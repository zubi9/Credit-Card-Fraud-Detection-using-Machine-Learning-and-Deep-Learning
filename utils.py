import numpy as np
import time

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def find_correlated_features(
    df,
    threshold: float = 0.98,
    return_corr: bool = False,
):
    """
    Identify highly correlated features.

    Parameters
    ----------
    df : pandas.DataFrame or cudf.DataFrame
        Input feature dataframe.

    threshold : float, default=0.98
        Correlation threshold above which features are considered redundant.

    return_corr : bool, default=False
        If True, also return the absolute correlation matrix.

    Returns
    -------
    list
        List of correlated feature names.

    Optional
    --------
    corr : pandas.DataFrame or cudf.DataFrame
        Absolute correlation matrix.
    """

    # Detect GPU support
    try:
        import cudf

        gpu_available = True
    except ImportError:
        gpu_available = False

    if gpu_available:
        # Convert pandas -> cuDF if necessary
        if not isinstance(df, cudf.DataFrame):
            df = cudf.from_pandas(df)

        corr = df.corr().abs()

        mask = cudf.DataFrame(
            np.triu(
                np.ones(corr.shape),
                k=1,
                dtype=bool,
            ),
            index=corr.index,
            columns=corr.columns,
        )

        upper = corr.where(mask)

        correlated_features = [
            col
            for col in upper.columns
            if (upper[col] > threshold).any()
        ]

    else:
        corr = df.corr().abs()

        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1,
                dtype=bool,
            )
        )

        correlated_features = [
            col
            for col in upper.columns
            if (upper[col] > threshold).any()
        ]

    if return_corr:
        return correlated_features, corr

    return correlated_features


    
# Find Best Classification Threshold
def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_scores = (
        2 *
        precision[:-1] *
        recall[:-1]
    ) / (
        precision[:-1] +
        recall[:-1] +
        1e-12
    )

    best_idx = np.argmax(f1_scores)

    return {
        "threshold": thresholds[best_idx],
        "precision": precision[best_idx],
        "recall": recall[best_idx],
        "f1": f1_scores[best_idx],
        "curve_precision": precision,
        "curve_recall": recall,
        "curve_thresholds": thresholds,
        "curve_f1": f1_scores
    }

def plot_threshold_curve(result):

    plt.figure(figsize=(10,6))

    plt.plot(
        result["curve_thresholds"],
        result["curve_f1"],
        linewidth=2
    )

    plt.axvline(
        result["threshold"],
        color="red",
        linestyle="--",
        label=f"Best = {result['threshold']:.3f}"
    )

    plt.xlabel("Threshold")

    plt.ylabel("F1 Score")

    plt.title("Threshold Optimization")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()

    
def plot_feature_importance(model,
                            feature_names,
                            top_n=20):

    if not hasattr(model, "feature_importances_"):

        return

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=importance.head(top_n),
        x="Importance",
        y="Feature"
    )

    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.show()
    return importance


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    model_name,
    feature_names=None
):

    print("="*80)
    print(model_name)
    print("="*80)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = time.time() - start

    probabilities = model.predict_proba(
        X_valid
    )[:,1]

    threshold_result = find_best_threshold(
        y_valid,
        probabilities
    )

    threshold = threshold_result["threshold"]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Model": model_name,
        "Training Time": train_time,
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_valid,
            predictions
        ),

        "Balanced Accuracy":
        balanced_accuracy_score(
            y_valid,
            predictions
        ),

        "Precision":
        precision_score(
            y_valid,
            predictions,
            zero_division=0
        ),

        "Recall":
        recall_score(
            y_valid,
            predictions
        ),

        "F1":
        f1_score(
            y_valid,
            predictions
        ),

        "ROC_AUC":
        roc_auc_score(
            y_valid,
            probabilities
        ),

        "PR_AUC":
        average_precision_score(
            y_valid,
            probabilities
        )
    }

    display(
        pd.DataFrame(metrics,index=[0])
    )

    print(classification_report(
        y_valid,
        predictions
    ))

    cm = confusion_matrix(
        y_valid,
        predictions
    )

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    plot_threshold_curve(
        threshold_result
    )

    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(
        y_valid,
        probabilities
    )

    plt.show()

    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(
        y_valid,
        probabilities
    )
    plt.show()

    if feature_names is not None:

        plot_feature_importance(
            model,
            feature_names
        )

    return metrics


    