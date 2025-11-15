"""
Command-line interface examples for PSOD.

This module demonstrates how to use PSOD from the command line for various tasks:
- Batch outlier detection on CSV files
- Model training and saving
- Scoring new data with saved models
- Generating reports
- Configurable via command-line arguments

Usage examples:
    # Basic detection
    python cli_examples.py detect --input data.csv --output results.csv

    # Train and save model
    python cli_examples.py train --input train_data.csv --model model.pkl

    # Score with saved model
    python cli_examples.py score --input new_data.csv --model model.pkl --output scores.csv

    # Generate report
    python cli_examples.py report --input data.csv --output report.html
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# For development, add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psod import PSOD, evaluate_outlier_detection, load_model, save_model


def detect_outliers(args):
    """Detect outliers in a CSV file."""
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")

    # Parse categorical columns
    cat_cols = args.cat_columns.split(",") if args.cat_columns else None
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    # Initialize detector
    print("\nInitializing PSOD detector...")
    detector = PSOD(
        cat_columns=cat_cols,
        min_cols_chosen=args.min_cols,
        max_cols_chosen=args.max_cols,
        stdevs_to_outlier=args.stdevs,
        transform_algorithm=args.transform,
        contamination=args.contamination,
        random_seed=args.seed,
    )

    # Detect outliers
    print("Detecting outliers...")
    scores = detector.fit_predict(df, return_class=False)
    labels = detector.fit_predict(df, return_class=True)

    # Create output DataFrame
    df_output = df.copy()
    df_output["outlier_score"] = scores
    df_output["is_outlier"] = labels

    # Save results
    df_output.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Outliers detected: {sum(labels)} ({100 * sum(labels) / len(df):.2f}%)")
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    print("=" * 60)

    # Save model if requested
    if args.save_model:
        save_model(detector, args.save_model)
        print(f"\nModel saved to: {args.save_model}")


def train_model(args):
    """Train a PSOD model and save it."""
    print(f"Loading training data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")

    # Parse categorical columns
    cat_cols = args.cat_columns.split(",") if args.cat_columns else None
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    # Initialize detector
    print("\nInitializing and training PSOD detector...")
    detector = PSOD(
        cat_columns=cat_cols,
        min_cols_chosen=args.min_cols,
        max_cols_chosen=args.max_cols,
        stdevs_to_outlier=args.stdevs,
        transform_algorithm=args.transform,
        contamination=args.contamination,
        random_seed=args.seed,
    )

    # Train
    detector.fit(df)
    print("Training complete!")

    # Save model
    save_model(detector, args.model)
    print(f"Model saved to: {args.model}")

    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(df.columns),
        "features": df.columns.tolist(),
        "cat_columns": cat_cols,
        "parameters": {
            "min_cols_chosen": args.min_cols,
            "max_cols_chosen": args.max_cols,
            "stdevs_to_outlier": args.stdevs,
            "transform_algorithm": args.transform,
            "contamination": args.contamination,
        },
    }

    metadata_path = args.model.replace(".pkl", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def score_data(args):
    """Score new data using a saved model."""
    print(f"Loading model from: {args.model}")
    detector = load_model(args.model)

    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")

    # Score data
    print("\nScoring data...")
    scores = detector.predict(df, return_class=False)
    labels = detector.predict(df, return_class=True)

    # Create output
    df_output = df.copy()
    df_output["outlier_score"] = scores
    df_output["is_outlier"] = labels

    # Save results
    df_output.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SCORING SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Outliers detected: {sum(labels)} ({100 * sum(labels) / len(df):.2f}%)")
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    print("=" * 60)


def generate_report(args):
    """Generate an HTML report with visualizations."""
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")

    # Parse categorical columns
    cat_cols = args.cat_columns.split(",") if args.cat_columns else None

    # Detect outliers
    print("Detecting outliers...")
    detector = PSOD(
        cat_columns=cat_cols,
        min_cols_chosen=args.min_cols,
        max_cols_chosen=args.max_cols,
        stdevs_to_outlier=args.stdevs,
        random_seed=args.seed,
    )

    scores = detector.fit_predict(df, return_class=False)
    labels = detector.fit_predict(df, return_class=True)

    # Generate visualizations
    print("Generating report...")

    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    from psod import compute_feature_importance
    from psod.visualization import plot_feature_contributions, plot_outlier_scores

    # Create figures
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_outlier_scores(scores, labels, ax=ax1)
    fig1.savefig("temp_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Feature importance
    feature_imp = compute_feature_importance(detector, df)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_features = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:15])
    ax2.barh(list(top_features.keys()), list(top_features.values()), color="steelblue")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Top 15 Features for Outlier Detection")
    ax2.grid(True, alpha=0.3, axis="x")
    fig2.savefig("temp_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PSOD Outlier Detection Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #4CAF50;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            img {{
                max-width: 100%;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PSOD Outlier Detection Report</h1>

            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Input File:</strong> {args.input}</p>

            <h2>Summary Statistics</h2>

            <div class="metric">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{len(df)}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Outliers Detected</div>
                <div class="metric-value">{sum(labels)}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Outlier Rate</div>
                <div class="metric-value">{100 * sum(labels) / len(df):.2f}%</div>
            </div>

            <h2>Detection Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>min_cols_chosen</td>
                    <td>{args.min_cols}</td>
                </tr>
                <tr>
                    <td>max_cols_chosen</td>
                    <td>{args.max_cols}</td>
                </tr>
                <tr>
                    <td>stdevs_to_outlier</td>
                    <td>{args.stdevs}</td>
                </tr>
                <tr>
                    <td>transform_algorithm</td>
                    <td>{args.transform}</td>
                </tr>
            </table>

            <h2>Score Statistics</h2>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{scores.mean():.4f}</td>
                </tr>
                <tr>
                    <td>Std Dev</td>
                    <td>{scores.std():.4f}</td>
                </tr>
                <tr>
                    <td>Min</td>
                    <td>{scores.min():.4f}</td>
                </tr>
                <tr>
                    <td>Max</td>
                    <td>{scores.max():.4f}</td>
                </tr>
                <tr>
                    <td>Median</td>
                    <td>{np.median(scores):.4f}</td>
                </tr>
            </table>

            <h2>Outlier Score Distribution</h2>
            <img src="temp_scores.png" alt="Outlier Score Distribution">

            <h2>Feature Importance</h2>
            <img src="temp_features.png" alt="Feature Importance">

            <h2>Top Outliers</h2>
            <table>
                <tr>
                    <th>Index</th>
                    <th>Score</th>
                </tr>
    """

    # Add top 20 outliers
    top_indices = np.argsort(scores)[-20:][::-1]
    for idx in top_indices:
        html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{scores[idx]:.4f}</td>
                </tr>
        """

    html_content += """
            </table>

            <div class="footer">
                <p>Generated by PSOD (Pseudo-Supervised Outlier Detection)</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(args.output, "w") as f:
        f.write(html_content)

    print(f"Report saved to: {args.output}")
    print("Note: Temporary image files (temp_*.png) created for report")


def evaluate_model(args):
    """Evaluate model performance with ground truth labels."""
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)

    if args.labels not in df.columns:
        print(f"Error: Label column '{args.labels}' not found in data")
        print(f"Available columns: {df.columns.tolist()}")
        return

    y_true = df[args.labels].values
    df_features = df.drop(columns=[args.labels])

    print(f"Dataset shape: {df_features.shape}")
    print(f"True outliers: {sum(y_true)} ({100 * sum(y_true) / len(y_true):.2f}%)")

    # Parse categorical columns
    cat_cols = args.cat_columns.split(",") if args.cat_columns else None

    # Detect outliers
    print("\nDetecting outliers...")
    detector = PSOD(
        cat_columns=cat_cols,
        min_cols_chosen=args.min_cols,
        max_cols_chosen=args.max_cols,
        stdevs_to_outlier=args.stdevs,
        random_seed=args.seed,
    )

    scores = detector.fit_predict(df_features, return_class=False)
    labels = detector.fit_predict(df_features, return_class=True)

    # Evaluate
    metrics = evaluate_outlier_detection(y_true, labels, scores)

    # Print results
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Precision:  {metrics['precision']:.3f}")
    print(f"Recall:     {metrics['recall']:.3f}")
    print(f"F1-Score:   {metrics['f1']:.3f}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.3f}")
    print("=" * 60)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, labels)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Normal  Outlier")
    print(f"Actual Normal   {cm[0, 0]:6d}  {cm[0, 1]:7d}")
    print(f"       Outlier  {cm[1, 0]:6d}  {cm[1, 1]:7d}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PSOD Command-Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Detect outliers:
    python cli_examples.py detect --input data.csv --output results.csv

  Train model:
    python cli_examples.py train --input train.csv --model model.pkl

  Score new data:
    python cli_examples.py score --input new.csv --model model.pkl --output scores.csv

  Generate report:
    python cli_examples.py report --input data.csv --output report.html

  Evaluate with ground truth:
    python cli_examples.py evaluate --input data.csv --labels is_outlier
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", "-i", required=True, help="Input CSV file")
    common.add_argument("--cat-columns", help="Comma-separated categorical column names")
    common.add_argument(
        "--min-cols", type=float, default=0.5, help="Min fraction of columns to use"
    )
    common.add_argument(
        "--max-cols", type=float, default=1.0, help="Max fraction of columns to use"
    )
    common.add_argument("--stdevs", type=float, default=2.0, help="Std devs for outlier threshold")
    common.add_argument("--transform", default="logarithmic", help="Transformation algorithm")
    common.add_argument(
        "--contamination", type=float, default=0.1, help="Expected outlier fraction"
    )
    common.add_argument("--seed", type=int, default=42, help="Random seed")

    # Detect command
    detect_parser = subparsers.add_parser("detect", parents=[common], help="Detect outliers")
    detect_parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    detect_parser.add_argument("--save-model", help="Save model to file")
    detect_parser.set_defaults(func=detect_outliers)

    # Train command
    train_parser = subparsers.add_parser("train", parents=[common], help="Train and save model")
    train_parser.add_argument("--model", "-m", required=True, help="Output model file")
    train_parser.set_defaults(func=train_model)

    # Score command
    score_parser = subparsers.add_parser("score", help="Score data with saved model")
    score_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    score_parser.add_argument("--model", "-m", required=True, help="Saved model file")
    score_parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    score_parser.set_defaults(func=score_data)

    # Report command
    report_parser = subparsers.add_parser("report", parents=[common], help="Generate HTML report")
    report_parser.add_argument("--output", "-o", required=True, help="Output HTML file")
    report_parser.set_defaults(func=generate_report)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="Evaluate with ground truth"
    )
    eval_parser.add_argument("--labels", "-l", required=True, help="Ground truth label column name")
    eval_parser.set_defaults(func=evaluate_model)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
