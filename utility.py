import os
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# ---------- 1) scores extractor (proba/decision) ----------
def get_model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), "proba"
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision"
    raise ValueError(
        "Model has neither predict_proba nor decision_function. Can't plot ROC/PR."
    )


# ---------- 2) save + log (NO show here) ----------
def save_and_log_fig(fig, filename, artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    mlflow.log_artifact(path)
    return path


# ---------- 3) draw ROC into a provided axis ----------
def plot_roc_auc_ax(ax, model, X, y, class_names, title="ROC", log_prefix="test"):
    scores, kind = get_model_scores(model, X)
    y = np.asarray(y)
    class_names = list(class_names)

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    # binary
    if len(class_names) == 2:
        s = (
            scores[:, 1]
            if (
                isinstance(scores, np.ndarray)
                and scores.ndim == 2
                and scores.shape[1] == 2
            )
            else np.ravel(scores)
        )
        y_bin = (y == class_names[1]).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, s)
        auc = roc_auc_score(y_bin, s)

        ax.plot(fpr, tpr)
        ax.set_title(f"{title}\nAUC={auc:.3f} ({kind})")

        mlflow.log_metric(f"{log_prefix}_roc_auc", float(auc))
        mlflow.log_param("score_kind", kind)
        return

    # multiclass
    y_bin = label_binarize(y, classes=class_names)

    if not (
        isinstance(scores, np.ndarray)
        and scores.ndim == 2
        and scores.shape[1] == len(class_names)
    ):
        ax.set_title(f"{title}\n(skipped: wrong score shape {np.shape(scores)})")
        return

    auc_macro = roc_auc_score(y_bin, scores, average="macro", multi_class="ovr")
    auc_micro = roc_auc_score(y_bin, scores, average="micro", multi_class="ovr")

    grid = np.linspace(0, 1, 300)
    tprs = []
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)

    ax.plot(grid, mean_tpr)
    ax.set_title(
        f"{title}\nmacro AUC={auc_macro:.3f}, micro AUC={auc_micro:.3f} ({kind})"
    )

    mlflow.log_metrics(
        {
            f"{log_prefix}_roc_auc_macro": float(auc_macro),
            f"{log_prefix}_roc_auc_micro": float(auc_micro),
        }
    )
    mlflow.log_param("score_kind", kind)


# ---------- 4) draw PR into a provided axis ----------
def plot_pr_ax(ax, model, X, y, class_names, title="PR", log_prefix="test"):
    scores, kind = get_model_scores(model, X)
    y = np.asarray(y)
    class_names = list(class_names)

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # binary
    if len(class_names) == 2:
        s = (
            scores[:, 1]
            if (
                isinstance(scores, np.ndarray)
                and scores.ndim == 2
                and scores.shape[1] == 2
            )
            else np.ravel(scores)
        )
        y_bin = (y == class_names[1]).astype(int)

        prec, rec, _ = precision_recall_curve(y_bin, s)
        ap = average_precision_score(y_bin, s)

        ax.plot(rec, prec)
        ax.set_title(f"{title}\nAP={ap:.3f} ({kind})")

        mlflow.log_metric(f"{log_prefix}_avg_precision", float(ap))
        mlflow.log_param("score_kind", kind)
        return

    # multiclass
    y_bin = label_binarize(y, classes=class_names)

    if not (
        isinstance(scores, np.ndarray)
        and scores.ndim == 2
        and scores.shape[1] == len(class_names)
    ):
        ax.set_title(f"{title}\n(skipped: wrong score shape {np.shape(scores)})")
        return

    ap_macro = average_precision_score(y_bin, scores, average="macro")
    ap_micro = average_precision_score(y_bin, scores, average="micro")

    grid = np.linspace(0, 1, 300)
    precs = []
    for i in range(y_bin.shape[1]):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        rec_inc = rec[::-1]
        prec_inc = prec[::-1]
        precs.append(np.interp(grid, rec_inc, prec_inc))
    mean_prec = np.mean(precs, axis=0)

    ax.plot(grid, mean_prec)
    ax.set_title(f"{title}\nmacro AP={ap_macro:.3f}, micro AP={ap_micro:.3f} ({kind})")

    mlflow.log_metrics(
        {
            f"{log_prefix}_avg_precision_macro": float(ap_macro),
            f"{log_prefix}_avg_precision_micro": float(ap_micro),
        }
    )
    mlflow.log_param("score_kind", kind)


# ---------- 5) draw F1 per class into a provided axis ----------
def plot_f1_per_class_ax(
    ax, y_true, y_pred, class_names, title="F1 per class", log_prefix="test"
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = list(class_names)

    f1s = []
    for c in class_names:
        t = y_true == c
        p = y_pred == c
        tp = np.sum(t & p)
        fp = np.sum(~t & p)
        fn = np.sum(t & ~p)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    f1s = np.array(f1s)
    order = np.argsort(f1s)

    ax.bar(np.arange(len(class_names)), f1s[order])
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(np.array(class_names)[order], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(title)

    for c, v in zip(np.array(class_names)[order], f1s[order]):
        mlflow.log_metric(f"{log_prefix}_f1_{c}", float(v))


# ---------- 6) one figure with 3 horizontal subplots + save/log once ----------
def log_three_plots_row(
    *,
    model,
    X_eval,
    y_eval,
    y_pred_eval,
    class_names,
    artifacts_dir="artifacts",
    filename="eval_row.png",
    log_prefix="test",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    plot_roc_auc_ax(
        axes[0], model, X_eval, y_eval, class_names, title="ROC", log_prefix=log_prefix
    )
    plot_pr_ax(
        axes[1],
        model,
        X_eval,
        y_eval,
        class_names,
        title="Precision-Recall",
        log_prefix=log_prefix,
    )
    plot_f1_per_class_ax(
        axes[2],
        y_eval,
        y_pred_eval,
        class_names,
        title="F1 per class",
        log_prefix=log_prefix,
    )

    path = save_and_log_fig(fig, filename, artifacts_dir)
    plt.show()
    plt.close(fig)
    return path


def run_mlflow_experiment(
    *,
    model,
    run_name_prefix: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    artifacts_dir: str = "artifacts",
    log_model: bool = True,
):
    """
    Uses ONLY test for evaluation plots + main evaluation metrics.
    Train metrics are still logged (overfit check).

    Logs:
      - params
      - metrics for train + test: accuracy, balanced_accuracy, f1_weighted
      - plots row (ROC/PR/F1 per class) on TEST split
      - split_metrics.csv as artifact

    Displays:
      - the 1x3 plots row (TEST)
      - a dataframe with train + test metrics
    """
    run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(artifacts_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        # train
        model.fit(X_train, y_train)

        # predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # metrics (requested)
        train_metrics = {
            "accuracy": float(accuracy_score(y_train, y_pred_train)),
            "balanced_accuracy": float(balanced_accuracy_score(y_train, y_pred_train)),
            "f1_weighted": float(f1_score(y_train, y_pred_train, average="weighted")),
        }
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_test)),
            "f1_weighted": float(f1_score(y_test, y_pred_test, average="weighted")),
        }

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # class order
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        elif hasattr(model, "named_steps") and hasattr(
            list(model.named_steps.values())[-1], "classes_"
        ):
            classes = list(list(model.named_steps.values())[-1].classes_)
        else:
            classes = sorted(pd.unique(pd.Series(y_train)))

        # plots (TEST)
        log_three_plots_row(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            y_pred_eval=y_pred_test,
            class_names=classes,
            artifacts_dir=artifacts_dir,
            filename="test_eval_row.png",
            log_prefix="test",
        )

        # display df (train/test only)
        metrics_df = pd.DataFrame(
            [{"split": "train", **train_metrics}, {"split": "test", **test_metrics}]
        ).set_index("split")
        display(metrics_df)

        # log df as artifact
        metrics_path = os.path.join(artifacts_dir, "split_metrics.csv")
        metrics_df.to_csv(metrics_path)
        mlflow.log_artifact(metrics_path)

        if log_model:
            mlflow.sklearn.log_model(model, name="model")

        print("Run ID:", run.info.run_id)
        return run.info.run_id


def run_mlflow_validation_only(
    *,
    model,
    run_name_prefix: str,
    params: dict,
    X_val,
    y_val,
    artifacts_dir: str = "artifacts",
):
    """
    Uses an ALREADY FITTED model.
    Evaluates ONLY on validation set.

    Logs:
      - params
      - validation metrics: accuracy, balanced_accuracy, f1_weighted
      - plots row (ROC / PR / F1 per class) on VALIDATION split
      - val_split_metrics.csv as artifact

    Displays:
      - the 1x3 plots row (VALIDATION)
      - a dataframe with validation metrics
    """
    run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(artifacts_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        # --- predict (NO training here)
        y_pred_val = model.predict(X_val)

        # --- metrics
        val_metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred_val)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred_val)),
            "f1_weighted": float(f1_score(y_val, y_pred_val, average="weighted")),
        }

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # --- class order (robust)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        elif hasattr(model, "named_steps") and hasattr(
            list(model.named_steps.values())[-1], "classes_"
        ):
            classes = list(list(model.named_steps.values())[-1].classes_)
        else:
            classes = sorted(pd.unique(pd.Series(y_val)))

        # --- plots (VALIDATION)
        log_three_plots_row(
            model=model,
            X_eval=X_val,
            y_eval=y_val,
            y_pred_eval=y_pred_val,
            class_names=classes,
            artifacts_dir=artifacts_dir,
            filename="val_eval_row.png",
            log_prefix="val",
        )

        # --- display df
        metrics_df = pd.DataFrame([{"split": "val", **val_metrics}]).set_index("split")
        display(metrics_df)

        # --- log df
        metrics_path = os.path.join(artifacts_dir, "val_split_metrics.csv")
        metrics_df.to_csv(metrics_path, index=True)
        mlflow.log_artifact(metrics_path)

        print("Validation run ID:", run.info.run_id)
        return run.info.run_id
