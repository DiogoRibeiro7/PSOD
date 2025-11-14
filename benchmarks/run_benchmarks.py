"""
Comprehensive Benchmarking Suite for PSOD Outlier Detection.

Compare PSOD performance against other popular outlier detection methods
across various datasets, metrics, and scenarios.
"""

import time
import warnings
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

# Import benchmark modules
from datasets import get_benchmark_datasets, generate_dataset, generate_scalability_datasets
from methods import get_all_methods, get_method_subset
from metrics import (
    compute_metrics,
    compute_precision_recall_at_k,
    compute_roc_curve_data,
    compute_pr_curve_data,
    compare_methods,
    aggregate_metrics,
)
from visualization import (
    plot_method_comparison,
    plot_performance_vs_time,
    plot_multi_metric_comparison,
    plot_scalability,
    plot_dataset_comparison,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_memory_usage,
    plot_radar_chart,
    create_benchmark_dashboard,
)

warnings.filterwarnings("ignore")


class BenchmarkRunner:
    """Main benchmark runner class."""

    def __init__(self, random_state: int = 42, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark runner.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        output_dir : str
            Directory to save results
        """
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results = []
        self.scalability_results = {}
        self.robustness_results = {}
        self.memory_results = {}

    def benchmark_method(
        self,
        method_wrapper,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        contamination: float,
        track_memory: bool = True,
    ) -> Dict:
        """
        Benchmark a single method on a dataset.

        Parameters
        ----------
        method_wrapper : OutlierDetectorWrapper
            Wrapped outlier detection method
        X_train : np.ndarray
            Training data
        X_test : np.ndarray
            Test data
        y_test : np.ndarray
            Test labels
        contamination : float
            Expected contamination level
        track_memory : bool
            Whether to track memory usage

        Returns
        -------
        results : Dict
            Benchmark results
        """
        method_name = method_wrapper.name
        print(f"\n  Benchmarking {method_name}...")

        results = {"method": method_name}

        try:
            # Start memory tracking
            if track_memory:
                tracemalloc.start()
                mem_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

            # Training time
            start_time = time.time()
            method_wrapper.fit(X_train)
            train_time = time.time() - start_time

            # Memory after training
            if track_memory:
                mem_after_train = tracemalloc.get_traced_memory()[0] / 1024 / 1024
                memory_train = mem_after_train - mem_before

            # Prediction time
            start_time = time.time()
            y_scores = method_wrapper.score_samples(X_test)
            pred_time = time.time() - start_time

            # Get binary predictions
            try:
                y_pred = method_wrapper.predict(X_test)
            except:
                # Fallback: use top-k as predictions
                k = int(len(y_test) * contamination)
                threshold = np.percentile(y_scores, 100 * (1 - contamination))
                y_pred = (y_scores > threshold).astype(int)

            # Memory after prediction
            if track_memory:
                mem_after_pred = tracemalloc.get_traced_memory()[0] / 1024 / 1024
                memory_pred = mem_after_pred - mem_after_train
                tracemalloc.stop()

            # Compute metrics
            metrics = compute_metrics(y_test, y_scores, y_pred, contamination)

            # Compile results
            results.update(
                {
                    "train_time": train_time,
                    "pred_time": pred_time,
                    "total_time": train_time + pred_time,
                    **metrics,
                }
            )

            if track_memory:
                results["memory_train_mb"] = memory_train
                results["memory_pred_mb"] = memory_pred
                results["memory_total_mb"] = memory_train + memory_pred

            # Print summary
            print(f"    Training time: {train_time:.3f}s")
            print(f"    Prediction time: {pred_time:.3f}s")
            print(f"    ROC-AUC: {results.get('roc_auc', 0):.3f}")
            print(f"    Average Precision: {results.get('avg_precision', 0):.3f}")
            if track_memory:
                print(f"    Memory usage: {results['memory_total_mb']:.2f} MB")

        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results.update(
                {
                    "train_time": np.nan,
                    "pred_time": np.nan,
                    "total_time": np.nan,
                    "roc_auc": np.nan,
                    "avg_precision": np.nan,
                    "error": str(e),
                }
            )

        return results

    def benchmark_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        method_subset: str = "basic",
        test_size: float = 0.3,
    ) -> pd.DataFrame:
        """
        Benchmark methods on multiple datasets.

        Parameters
        ----------
        dataset_names : List[str], optional
            Datasets to benchmark (default: all)
        method_subset : str
            Method subset: 'basic', 'fast', 'accurate', 'all'
        test_size : float
            Test set proportion

        Returns
        -------
        results_df : pd.DataFrame
            Benchmark results
        """
        print("=" * 70)
        print("BENCHMARKING ON MULTIPLE DATASETS")
        print("=" * 70)

        datasets = get_benchmark_datasets()

        if dataset_names is not None:
            datasets = {k: v for k, v in datasets.items() if k in dataset_names}

        all_results = []

        for dataset_name, dataset_config in datasets.items():
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name}")
            print(f"Description: {dataset_config.get('description', 'N/A')}")
            print(f"{'='*70}")

            # Generate dataset
            X, y = generate_dataset(dataset_config, random_state=self.random_state)

            print(
                f"Samples: {len(X)}, Features: {X.shape[1]}, Outliers: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)"
            )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            contamination = dataset_config.get("contamination", 0.1)

            # Get methods
            methods = get_method_subset(method_subset, contamination, self.random_state)

            if not methods:
                print("  WARNING: No methods available!")
                continue

            # Benchmark each method
            for method_name, method_wrapper in methods.items():
                result = self.benchmark_method(
                    method_wrapper, X_train_scaled, X_test_scaled, y_test, contamination
                )
                result["dataset"] = dataset_name
                result["n_samples"] = len(X)
                result["n_features"] = X.shape[1]
                result["contamination"] = contamination
                all_results.append(result)

        self.results.extend(all_results)
        return pd.DataFrame(all_results)

    def benchmark_scalability(self, method_subset: str = "fast", n_trials: int = 3) -> Dict:
        """
        Test scalability with increasing data size.

        Parameters
        ----------
        method_subset : str
            Method subset to test
        n_trials : int
            Number of trials for averaging

        Returns
        -------
        scalability_results : Dict
            Results indexed by method
        """
        print("\n" + "=" * 70)
        print("SCALABILITY BENCHMARKING")
        print("=" * 70)

        scalability_datasets = generate_scalability_datasets()
        methods = get_method_subset(
            method_subset, contamination=0.05, random_state=self.random_state
        )

        results = {
            method_name: {"n_samples": [], "n_features": [], "train_time": [], "pred_time": []}
            for method_name in methods.keys()
        }

        for dataset_name, (X, y) in scalability_datasets.items():
            n_samples, n_features = X.shape
            print(f"\nDataset: {dataset_name} (n={n_samples}, d={n_features})")

            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            for method_name, method_wrapper in methods.items():
                print(f"  {method_name}...", end=" ")

                train_times = []
                pred_times = []

                for trial in range(n_trials):
                    try:
                        # Training
                        start = time.time()
                        method_wrapper.fit(X_scaled)
                        train_time = time.time() - start

                        # Prediction
                        start = time.time()
                        method_wrapper.score_samples(X_scaled)
                        pred_time = time.time() - start

                        train_times.append(train_time)
                        pred_times.append(pred_time)

                    except Exception as e:
                        print(f"ERROR: {e}")
                        break

                if train_times:
                    results[method_name]["n_samples"].append(n_samples)
                    results[method_name]["n_features"].append(n_features)
                    results[method_name]["train_time"].append(np.mean(train_times))
                    results[method_name]["pred_time"].append(np.mean(pred_times))
                    print(f"Train: {np.mean(train_times):.3f}s, Pred: {np.mean(pred_times):.3f}s")

        self.scalability_results = results
        return results

    def benchmark_robustness(self, method_subset: str = "basic") -> Dict:
        """
        Test robustness to various conditions.

        Parameters
        ----------
        method_subset : str
            Method subset to test

        Returns
        -------
        robustness_results : Dict
            Robustness test results
        """
        print("\n" + "=" * 70)
        print("ROBUSTNESS BENCHMARKING")
        print("=" * 70)

        results = {"contamination_levels": [], "noise_levels": [], "dimensionality": []}

        # Test 1: Different contamination levels
        print("\n1. Testing different contamination levels...")
        contamination_levels = [0.01, 0.05, 0.10, 0.15, 0.20]

        for contamination in contamination_levels:
            print(f"\nContamination: {contamination*100:.0f}%")

            from datasets import generate_global_outliers

            X, y = generate_global_outliers(
                n_samples=2000,
                n_features=20,
                contamination=contamination,
                random_state=self.random_state,
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            methods = get_method_subset(method_subset, contamination, self.random_state)

            for method_name, method_wrapper in methods.items():
                try:
                    method_wrapper.fit(X_scaled)
                    y_scores = method_wrapper.score_samples(X_scaled)
                    metrics = compute_metrics(y, y_scores, contamination=contamination)

                    results["contamination_levels"].append(
                        {
                            "method": method_name,
                            "contamination": contamination,
                            "roc_auc": metrics["roc_auc"],
                            "avg_precision": metrics["avg_precision"],
                        }
                    )

                except Exception as e:
                    print(f"  {method_name}: ERROR - {e}")

        # Test 2: Different noise levels (dimensionality curse)
        print("\n2. Testing high-dimensional sparse data...")
        dimensions = [10, 50, 100, 200]

        for n_features in dimensions:
            print(f"\nDimensions: {n_features}")

            from datasets import generate_high_dimensional_outliers

            X, y = generate_high_dimensional_outliers(
                n_samples=1000,
                n_features=n_features,
                contamination=0.1,
                random_state=self.random_state,
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            methods = get_method_subset(method_subset, 0.1, self.random_state)

            for method_name, method_wrapper in methods.items():
                try:
                    method_wrapper.fit(X_scaled)
                    y_scores = method_wrapper.score_samples(X_scaled)
                    metrics = compute_metrics(y, y_scores, contamination=0.1)

                    results["dimensionality"].append(
                        {
                            "method": method_name,
                            "n_features": n_features,
                            "roc_auc": metrics["roc_auc"],
                            "avg_precision": metrics["avg_precision"],
                        }
                    )

                except Exception as e:
                    print(f"  {method_name}: ERROR - {e}")

        self.robustness_results = results
        return results

    def save_results(self, filename: str = "benchmark_results.csv"):
        """Save benchmark results to CSV."""
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_path = self.output_dir / filename
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
            return results_df
        else:
            print("No results to save!")
            return None

    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("\n" + "=" * 70)
        print("GENERATING BENCHMARK REPORT")
        print("=" * 70)

        if not self.results:
            print("No results available!")
            return

        results_df = pd.DataFrame(self.results)

        # Summary statistics
        print("\n1. Overall Performance Summary")
        print("-" * 70)
        summary = results_df.groupby("method")[
            ["roc_auc", "avg_precision", "train_time", "pred_time"]
        ].mean()
        summary = summary.sort_values("roc_auc", ascending=False)
        print(summary.to_string())

        # Best method per dataset
        if "dataset" in results_df.columns:
            print("\n2. Best Method per Dataset")
            print("-" * 70)
            for dataset in results_df["dataset"].unique():
                dataset_results = results_df[results_df["dataset"] == dataset]
                best_method = dataset_results.loc[dataset_results["roc_auc"].idxmax(), "method"]
                best_score = dataset_results["roc_auc"].max()
                print(f"{dataset:25s}: {best_method:20s} (ROC-AUC: {best_score:.3f})")

        # Create visualizations
        print("\n3. Creating visualizations...")

        # Main comparison plot
        plot_method_comparison(
            results_df, metric="roc_auc", save_path=self.output_dir / "method_comparison_roc.png"
        )
        print("  - Method comparison (ROC-AUC)")

        # Performance vs time
        if "total_time" in results_df.columns:
            plot_performance_vs_time(
                results_df, save_path=self.output_dir / "performance_vs_time.png"
            )
            print("  - Performance vs time scatter")

        # Multi-metric comparison
        metrics = ["roc_auc", "avg_precision", "precision", "recall", "f1_score"]
        available_metrics = [m for m in metrics if m in results_df.columns]
        if len(available_metrics) >= 3:
            plot_multi_metric_comparison(
                results_df,
                metrics=available_metrics,
                save_path=self.output_dir / "multi_metric_heatmap.png",
            )
            print("  - Multi-metric heatmap")

        # Dataset comparison
        if "dataset" in results_df.columns and len(results_df["dataset"].unique()) > 1:
            plot_dataset_comparison(
                results_df, save_path=self.output_dir / "dataset_comparison.png"
            )
            print("  - Dataset comparison")

        # Scalability plots
        if self.scalability_results:
            plot_scalability(
                self.scalability_results, save_path=self.output_dir / "scalability.png"
            )
            print("  - Scalability curves")

        # Memory usage
        if "memory_total_mb" in results_df.columns:
            memory_data = results_df.groupby("method")["memory_total_mb"].apply(list).to_dict()
            plot_memory_usage(memory_data, save_path=self.output_dir / "memory_usage.png")
            print("  - Memory usage comparison")

        # Comprehensive dashboard
        create_benchmark_dashboard(
            results_df, save_path=self.output_dir / "benchmark_dashboard.png"
        )
        print("  - Comprehensive dashboard")

        # Save markdown report
        self._save_markdown_report(results_df)

        print(f"\nAll results saved to: {self.output_dir}")

    def _save_markdown_report(self, results_df: pd.DataFrame):
        """Save results as markdown report."""
        report_path = self.output_dir / "BENCHMARK_REPORT.md"

        with open(report_path, "w") as f:
            f.write("# PSOD Benchmark Report\n\n")
            f.write("## 1. Overall Performance Summary\n\n")

            summary = results_df.groupby("method")[
                ["roc_auc", "avg_precision", "train_time", "pred_time"]
            ].agg(["mean", "std"])
            summary = summary.round(4)

            f.write(summary.to_markdown())

            f.write("\n\n## 2. Method Rankings\n\n")
            rankings = results_df.groupby("method")["roc_auc"].mean().sort_values(ascending=False)
            f.write("| Rank | Method | ROC-AUC |\n")
            f.write("|------|--------|----------|\n")
            for rank, (method, score) in enumerate(rankings.items(), 1):
                f.write(f"| {rank} | {method} | {score:.4f} |\n")

            f.write("\n\n## 3. Visualizations\n\n")
            f.write("- [Method Comparison](method_comparison_roc.png)\n")
            f.write("- [Performance vs Time](performance_vs_time.png)\n")
            f.write("- [Multi-Metric Heatmap](multi_metric_heatmap.png)\n")
            f.write("- [Benchmark Dashboard](benchmark_dashboard.png)\n")

        print(f"  - Markdown report: {report_path}")


def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("PSOD COMPREHENSIVE BENCHMARKING SUITE")
    print("=" * 70)

    # Initialize benchmark runner
    runner = BenchmarkRunner(random_state=42, output_dir="benchmark_results")

    # 1. Benchmark on standard datasets
    print("\nStep 1: Benchmarking on standard datasets...")
    dataset_names = ["small_global", "medium_global", "medium_mixed", "low_contamination"]
    results_df = runner.benchmark_datasets(
        dataset_names=dataset_names,
        method_subset="basic",  # Use 'basic' for quick testing, 'all' for comprehensive
    )

    # 2. Scalability testing
    print("\nStep 2: Running scalability tests...")
    scalability_results = runner.benchmark_scalability(method_subset="fast")

    # 3. Robustness testing
    print("\nStep 3: Running robustness tests...")
    robustness_results = runner.benchmark_robustness(method_subset="basic")

    # 4. Save results
    print("\nStep 4: Saving results...")
    runner.save_results()

    # 5. Generate comprehensive report
    print("\nStep 5: Generating report...")
    runner.generate_report()

    print("\n" + "=" * 70)
    print("BENCHMARKING COMPLETED!")
    print("=" * 70)
    print(f"\nResults available in: {runner.output_dir}")


if __name__ == "__main__":
    main()
