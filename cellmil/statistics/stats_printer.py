import wandb
import pandas as pd
from tqdm import tqdm
import bambi as bmb  # type: ignore
import arviz as az
from markdown_it import MarkdownIt
from weasyprint import HTML  # type: ignore
from typing import Any, cast
from statsmodels.formula.api import mixedlm  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
from cellmil.interfaces.StatsPrinterConfig import StatsPrinterConfig
from cellmil.utils.wandb import WandbClient
from cellmil.utils import logger


class StatsPrinter:
    # Constants for configuration keys
    FEATURES_KEY = "FEATURES"
    GNNS_KEY = "GNNs"
    MILS_KEY = "MILs"
    GNN_KEY = "GNN"
    MIL_KEY = "MIL"

    # Constants for DataFrame columns
    COLUMN_EXPERIMENT_ID = "EXPERIMENT_ID"
    COLUMN_TASK = "TASK"
    METRICS = ["f1", "recall", "precision", "auroc"]

    def __init__(self, config: StatsPrinterConfig):
        self.config = config
        wandb.login()  # Ensure wandb is logged in

        self.tasks: list[str] = ["ADENOvsSQUA", "PDL1", "DCR", "OS6", "OS24"]

        # TODO: Make this configurable
        self.base_run_config: dict[str, Any] = {
            self.FEATURES_KEY: "RESNET",
            self.GNN_KEY: None,
            self.MIL_KEY: "ATTENTION",
        }
        self.run_configs: dict[str, Any] = {
            self.FEATURES_KEY: [
                "RESNET",
                "GIGAPATH",
                "UNI",
                "MORPHO",
                "PYRAD",
                "GRAPH",
                "ALL",
            ],
            self.GNNS_KEY: [None, "GAT", "SMALLWORLD"],
            self.MILS_KEY: ["ATTENTION", "CLAM", "HEAD4TYPE"],
        }
        # TODO: ----

        self.df = pd.DataFrame()
        self.wandb_client = WandbClient(
            team=self.config.team, projects=self.config.projects, tasks=self.tasks
        )
        self.runs = self.wandb_client.get_runs(preprocess=True)
        logger.info(f"Total accessible runs after preprocessing: {len(self.runs)}")

        self._load_runs_into_df()

        if self.df.empty:
            raise RuntimeError(
                "DataFrame is empty after loading runs. Check logs for errors during run processing."
            )

        print(self.df.head())

    def _load_runs_into_df(self):
        def process_run(run: Any) -> dict[str, str | int | None | float]:
            """Process a single run and return its data."""
            experiment_id = self.wandb_client.get_experiment_id(run)

            run_data: dict[str, str | int | None | float] = {
                self.COLUMN_EXPERIMENT_ID: experiment_id,
                self.COLUMN_TASK: self.wandb_client.get_task(experiment_id),
                **self._get_run_config(experiment_id),
                **{
                    metric: self.wandb_client.get_metric(run, metric)
                    for metric in self.METRICS
                },
            }
            return run_data

        data: list[dict[str, Any]] = []

        # Use ThreadPoolExecutor for parallel processing (better for I/O-bound operations)
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_run = {
                executor.submit(process_run, run): run for run in self.runs
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(self.runs), desc="Loading runs into DataFrame") as pbar:
                for future in as_completed(future_to_run):
                    try:
                        run_data = future.result()
                        data.append(run_data)
                    except Exception as e:
                        run = future_to_run[future]
                        logger.error(f"Error processing run {run.name}: {e}")
                    finally:
                        pbar.update(1)

        self.df = pd.DataFrame(data)
        logger.info(f"Loaded {len(self.df)} runs into DataFrame.")

    def _get_run_config(self, experiment_id: str) -> dict[str, int]:
        """
        Get the run configuration associated with a given experiment ID as one-hot encoded features.
        Excludes base configuration options.

        Args:
            experiment_id: The ID of the experiment

        Returns:
            A dictionary with one-hot encoded columns for each configuration option (excluding base config)
        """
        run_config: dict[str, int] = {}

        # Check for each feature type (excluding base)
        for feature in self.run_configs[self.FEATURES_KEY]:
            if feature != self.base_run_config[self.FEATURES_KEY]:
                run_config[feature] = 1 if feature in experiment_id else 0

        # Check for each GNN type (excluding base, which is None)
        for gnn in self.run_configs[self.GNNS_KEY]:
            if gnn is not None and gnn != self.base_run_config[self.GNN_KEY]:
                run_config[gnn] = 1 if gnn in experiment_id else 0

        # Check for each MIL type (excluding base)
        for mil in self.run_configs[self.MILS_KEY]:
            if mil != self.base_run_config[self.MIL_KEY]:
                run_config[mil] = 1 if mil in experiment_id else 0

        return run_config

    def _fit_frequentist_models(
        self, metric: str, config_columns: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Fit frequentist Linear Mixed Effects models for each task.

        Args:
            config_columns: List of configuration column names

        Returns:
            Dictionary mapping task names to their model results
        """
        logger.info("Fitting frequentist LME models...")
        results: dict[str, dict[str, Any]] = {}

        for task in self.tasks:
            task_df = self.df[self.df[self.COLUMN_TASK] == task].copy()

            if task_df.empty:
                logger.warning(f"No data found for task: {task}")
                results[task] = {"error": "No data available"}
                continue

            # Only include configs that have variance
            available_configs = [
                config for config in config_columns if task_df[config].nunique() >= 2
            ]

            if not available_configs:
                logger.warning(f"No configurations with variance for task: {task}")
                results[task] = {"error": "No configurations with variance"}
                continue

            logger.info(f"Available configurations for {task}: {available_configs}")

            # Create formula
            fixed_effects_formula = " + ".join(available_configs)
            formula = f"{metric} ~ {fixed_effects_formula}"

            try:
                logger.info(f"Fitting LME model for {task}: {formula}")

                # Fit the model
                model = mixedlm(
                    formula, data=task_df, groups=task_df[self.COLUMN_EXPERIMENT_ID]
                )
                result = model.fit(method="powell", reml=True)  # type: ignore

                # Store results
                results[task] = {
                    "model": result,
                    "formula": formula,
                    "available_configs": available_configs,
                    "task_df": task_df,
                    "n_runs": len(task_df),
                    "n_experiments": task_df[self.COLUMN_EXPERIMENT_ID].nunique(),
                    "mean_metric": task_df[metric].mean(),
                    "std_metric": task_df[metric].std(),
                }

                logger.info(f"Successfully fitted model for {task}")

            except Exception as e:
                logger.error(f"Error fitting LME for task {task}: {e}")
                results[task] = {"error": str(e)}
                import traceback

                traceback.print_exc()

        return results

    def _fit_bayesian_models(
        self, metric: str, config_columns: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Fit Bayesian hierarchical models for each task using Bambi.

        Args:
            config_columns: List of configuration column names

        Returns:
            Dictionary mapping task names to their model results
        """

        logger.info("Fitting Bayesian hierarchical models...")
        results: dict[str, dict[str, Any]] = {}

        for task in self.tasks:
            task_df = self.df[self.df[self.COLUMN_TASK] == task].copy()

            if task_df.empty:
                logger.warning(f"No data found for task: {task}")
                results[task] = {"error": "No data available"}
                continue

            # Only include configs that have variance
            available_configs = [
                config for config in config_columns if task_df[config].nunique() >= 2
            ]

            if not available_configs:
                logger.warning(f"No configurations with variance for task: {task}")
                results[task] = {"error": "No configurations with variance"}
                continue

            logger.info(
                f"Available configurations for {task} (Bayesian): {available_configs}"
            )

            # Create formula with random intercept
            fixed_effects_formula = " + ".join(available_configs)
            formula = (
                f"{metric} ~ {fixed_effects_formula} + (1|{self.COLUMN_EXPERIMENT_ID})"
            )

            try:
                logger.info(f"Fitting Bayesian model for {task}: {formula}")

                # Build Bambi model
                model = bmb.Model(formula, data=task_df)

                # Fit the model with MCMC
                # Increase target_accept to reduce divergences
                idata = model.fit(  # type: ignore
                    draws=2000,
                    tune=1000,
                    chains=4,
                    random_seed=42,
                    target_accept=0.95,  # Increase from default 0.8 to reduce divergences
                )

                # Get posterior summary using arviz
                summary = az.summary(idata, hdi_prob=0.95)  # type: ignore

                # Store results
                results[task] = {
                    "model": model,
                    "idata": idata,
                    "summary": summary,
                    "formula": formula,
                    "available_configs": available_configs,
                    "task_df": task_df,
                    "n_runs": len(task_df),
                    "n_experiments": task_df[self.COLUMN_EXPERIMENT_ID].nunique(),
                    "mean_metric": task_df[metric].mean(),
                    "std_metric": task_df[metric].std(),
                }

                logger.info(f"Successfully fitted Bayesian model for {task}")

            except Exception as e:
                logger.error(f"Error fitting Bayesian model for task {task}: {e}")
                results[task] = {"error": str(e)}
                import traceback

                traceback.print_exc()

        return results

    def create(self, metric: str):
        """
        Perform both frequentist and Bayesian analyses, then generate a comprehensive report.
        """
        logger.info("Starting statistical analysis (Frequentist + Bayesian)...")

        # Get all configuration columns (exclude metadata and ALL metrics)
        config_columns = [
            col
            for col in self.df.columns
            if col not in [self.COLUMN_EXPERIMENT_ID, self.COLUMN_TASK] + self.METRICS
        ]

        logger.info(f"Configuration columns: {config_columns}")

        # Fit both types of models
        frequentist_results = self._fit_frequentist_models(metric, config_columns)
        bayesian_results = self._fit_bayesian_models(metric, config_columns)
        # bayesian_results = {}  # Temporarily disable Bayesian fitting for faster testing

        # Generate the report
        markdown_content = self._generate_report(
            metric, config_columns, frequentist_results, bayesian_results
        )

        # Write markdown to file
        output_md_file = "statistical_analysis_report.md"
        with open(output_md_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Markdown report saved to: {output_md_file}")

        # Generate PDF from markdown
        output_pdf_file = "statistical_analysis_report.pdf"
        try:
            self._generate_pdf_from_markdown(markdown_content, output_pdf_file)
            logger.info(f"PDF report generated: {output_pdf_file}")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")

        markdown_lines = markdown_content.split("\n")
        section_count = len([line for line in markdown_lines if line.startswith("## ")])
        print(f"   Total sections: {section_count}")
        print(f"   Total lines: {len(markdown_lines)}")

    def _generate_report(
        self,
        metric: str,
        config_columns: list[str],
        frequentist_results: dict[str, dict[str, Any]],
        bayesian_results: dict[str, dict[str, Any]],
    ) -> str:
        """
        Generate a comprehensive markdown report with both frequentist and Bayesian results.

        Args:
            config_columns: List of configuration column names
            frequentist_results: Results from frequentist LME models
            bayesian_results: Results from Bayesian hierarchical models

        Returns:
            Complete markdown report as a string
        """
        markdown_lines: list[str] = []

        # Header
        markdown_lines.append("# Statistical Analysis Report")
        markdown_lines.append("")
        markdown_lines.append(
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        markdown_lines.append("")

        # Overview
        markdown_lines.append("## Overview")
        markdown_lines.append("")
        markdown_lines.append(f"- **Total Runs Analyzed:** {len(self.df)}")
        markdown_lines.append(
            f"- **Total Experiments:** {self.df[self.COLUMN_EXPERIMENT_ID].nunique()}"
        )
        markdown_lines.append(f"- **Tasks:** {', '.join(self.tasks)}")
        markdown_lines.append(
            f"- **Configuration Options:** {', '.join(config_columns)}"
        )
        markdown_lines.append("")

        # Methodology
        markdown_lines.append("## Methodology")
        markdown_lines.append("")
        markdown_lines.append(
            "This report presents results from two complementary statistical approaches:"
        )
        markdown_lines.append("")
        markdown_lines.append("### 1. Frequentist Linear Mixed Effects (LME) Models")
        markdown_lines.append("")
        markdown_lines.append(
            "- **Fixed Effects:** Configuration options (binary indicators)"
        )
        markdown_lines.append(
            "- **Random Effects:** Random intercept for each experiment ID (5-fold CV)"
        )
        markdown_lines.append("- **Estimation:** REML (Restricted Maximum Likelihood)")
        markdown_lines.append("- **Inference:** p-values and confidence intervals")
        markdown_lines.append("")
        markdown_lines.append("### 2. Bayesian Hierarchical Models")
        markdown_lines.append("")
        markdown_lines.append(
            "- **Fixed Effects:** Configuration options (binary indicators)"
        )
        markdown_lines.append(
            "- **Random Effects:** Random intercept for each experiment ID (5-fold CV)"
        )
        markdown_lines.append(
            "- **Estimation:** MCMC sampling (2000 draws, 1000 tuning, 4 chains, target_accept=0.95)"
        )
        markdown_lines.append(
            "- **Inference:** Posterior distributions and 95% HDI (Highest Density Intervals)"
        )
        markdown_lines.append("")
        markdown_lines.append("**Outcome Variable:** Maximum validation F1 score")
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")

        # Results for each task
        for task in self.tasks:
            freq_result = frequentist_results.get(task, {})
            bayes_result = bayesian_results.get(task, {})

            markdown_lines.append(f"## Task: {task}")
            markdown_lines.append("")

            # Check if we have errors
            if "error" in freq_result and "error" in bayes_result:
                markdown_lines.append("**No analysis available for this task**")
                markdown_lines.append("")
                markdown_lines.append(f"- Frequentist: {freq_result['error']}")
                markdown_lines.append(f"- Bayesian: {bayes_result['error']}")
                markdown_lines.append("")
                markdown_lines.append("---")
                markdown_lines.append("")
                continue

            # Dataset summary (use whichever result is available)
            result_for_summary = (
                freq_result if "n_runs" in freq_result else bayes_result
            )
            if "n_runs" in result_for_summary:
                markdown_lines.append("### Dataset Summary")
                markdown_lines.append("")
                markdown_lines.append(
                    f"- **Total runs:** {result_for_summary['n_runs']}"
                )
                markdown_lines.append(
                    f"- **Total experiments:** {result_for_summary['n_experiments']}"
                )
                # markdown_lines.append(
                #     # f"- **Mean F1 score:** {result_for_summary['mean_f1']:.4f} ± {result_for_summary['std_f1']:.4f}"
                # )
                markdown_lines.append("")

            # Frequentist results
            if "model" in freq_result:
                markdown_lines.extend(
                    self._format_frequentist_results(metric, freq_result)
                )
            elif "error" in freq_result:
                markdown_lines.append("### Frequentist Analysis Error")
                markdown_lines.append("")
                markdown_lines.append(f"```\n{freq_result['error']}\n```")
                markdown_lines.append("")

            # Bayesian results
            if "model" in bayes_result:
                markdown_lines.extend(self._format_bayesian_results(bayes_result))
            elif "error" in bayes_result:
                markdown_lines.append("### Bayesian Analysis Error")
                markdown_lines.append("")
                markdown_lines.append(f"```\n{bayes_result['error']}\n```")
                markdown_lines.append("")

            markdown_lines.append("---")
            markdown_lines.append("")

        # Overall summary
        markdown_lines.append("## Overall Summary Across All Tasks")
        markdown_lines.append("")
        markdown_lines.append(
            "This section provides descriptive statistics aggregated across all tasks."
        )
        markdown_lines.append("")

        for config in config_columns:
            if self.df[config].nunique() < 2:
                continue

            markdown_lines.append(f"### {config}")
            markdown_lines.append("")
            markdown_lines.append("| Level | Mean F1 | Std Dev | N |")
            markdown_lines.append("|-------|---------|---------|---|")
            for level in sorted(self.df[config].unique()):  # type: ignore
                level_data = cast(pd.DataFrame, self.df[self.df[config] == level])
                level_label = "Yes" if level == 1 else "No"
                markdown_lines.append(
                    f"| {level_label} | {level_data[metric].mean():.4f} | "
                    f"{level_data[metric].std():.4f} | {len(level_data)} |"
                )
            markdown_lines.append("")

        return "\n".join(markdown_lines)

    def _format_frequentist_results(
        self, metric: str, result: dict[str, Any]
    ) -> list[str]:
        """Format frequentist LME results for the markdown report."""
        lines: list[str] = []

        model = result["model"]
        available_configs = result["available_configs"]
        task_df = result["task_df"]

        lines.append("### Frequentist Analysis (Linear Mixed Effects)")
        lines.append("")
        lines.append(f"**Model Specification:** `{result['formula']}`")
        lines.append("")
        lines.append(
            f"**Random Effect:** Random intercept by `{self.COLUMN_EXPERIMENT_ID}`"
        )
        lines.append("")

        # Fixed effects table
        lines.append("#### Fixed Effects")
        lines.append("")

        # Intercept
        if "Intercept" in model.params.index:
            intercept = model.params["Intercept"]
            intercept_se = model.bse["Intercept"]
            lines.append(
                f"**Intercept (Baseline F1):** {intercept:.4f} ± {intercept_se:.4f}"
            )
            lines.append("")
            lines.append(
                "> Expected F1 for baseline configuration (RESNET + no GNN + ATTENTION)."
            )
            lines.append("")

        # Configuration effects table
        lines.append(
            "| Configuration | Coefficient | Std Error | 95% CI | p-value | Sig |"
        )
        lines.append(
            "|---------------|-------------|-----------|--------|---------|-----|"
        )

        for config in available_configs:
            if config in model.params.index:
                coef = model.params[config]
                pval = model.pvalues[config]
                stderr = model.bse[config]
                ci_lower = model.conf_int().loc[config, 0]
                ci_upper = model.conf_int().loc[config, 1]
                significance = (
                    "***"
                    if pval < 0.001
                    else "**"
                    if pval < 0.01
                    else "*"
                    if pval < 0.05
                    else "ns"
                )

                lines.append(
                    f"| {config} | {coef:+.4f} | {stderr:.4f} | "
                    f"[{ci_lower:.4f}, {ci_upper:.4f}] | {pval:.4f} | {significance} |"
                )

        lines.append("")
        lines.append(
            "*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*"
        )
        lines.append("")

        # Significant effects interpretation
        lines.append("**Interpretation of Significant Effects:**")
        lines.append("")
        significant_found = False
        for config in available_configs:
            if config in model.params.index:
                coef = model.params[config]
                pval = model.pvalues[config]
                if pval < 0.05:
                    significant_found = True
                    direction = "increases" if coef > 0 else "decreases"
                    arrow = "⬆️" if coef > 0 else "⬇️"
                    lines.append(
                        f"- {arrow} **{config}:** {direction.capitalize()} F1 by "
                        f"{abs(coef):.4f} (p={pval:.4f})"
                    )

        if not significant_found:
            lines.append("*No statistically significant effects (p < 0.05)*")

        lines.append("")

        # Random effects
        lines.append("#### Random Effects")
        lines.append("")

        random_effect_var = (
            model.cov_re.iloc[0, 0] if hasattr(model, "cov_re") else model.scale
        )
        residual_var = model.scale
        total_var = random_effect_var + residual_var
        icc = random_effect_var / total_var

        lines.append("| Component | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Random intercept variance (τ²) | {random_effect_var:.6f} |")
        lines.append(f"| Residual variance (σ²) | {residual_var:.6f} |")
        lines.append(f"| **ICC** | **{icc:.4f}** |")
        lines.append("")
        lines.append(
            f"> **ICC:** {icc * 100:.2f}% of variance is between experiments. "
            f"Measures similarity within 5-fold CV."
        )
        lines.append("")

        # Descriptive stats
        lines.append("#### Descriptive Statistics")
        lines.append("")
        for config in available_configs:
            lines.append(f"**{config}:**")
            lines.append("")
            lines.append("| Level | Mean F1 | Std Dev | N |")
            lines.append("|-------|---------|---------|---|")
            for level in sorted(task_df[config].unique()):
                level_data = task_df[task_df[config] == level]
                level_label = "Yes" if level == 1 else "No"
                lines.append(
                    f"| {level_label} | {level_data[metric].mean():.4f} | "
                    f"{level_data[metric].std():.4f} | {len(level_data)} |"
                )
            lines.append("")

        return lines

    def _format_bayesian_results(self, result: dict[str, Any]) -> list[str]:
        """Format Bayesian hierarchical model results for the markdown report."""
        lines: list[str] = []

        summary = result["summary"]
        available_configs = result["available_configs"]
        idata = result["idata"]  # Get inference data for posterior samples

        lines.append("### Bayesian Analysis (Hierarchical Model)")
        lines.append("")
        lines.append(f"**Model Specification:** `{result['formula']}`")
        lines.append("")
        lines.append(
            "**MCMC Settings:** 2000 draws, 1000 tuning, 4 chains, target_accept=0.95"
        )
        lines.append("")

        # Fixed effects table
        lines.append("#### Posterior Distributions (Fixed Effects)")
        lines.append("")

        lines.append("| Parameter | Mean | SD | 95% HDI | R-hat |")
        lines.append("|-----------|------|----|---------|----|")

        for param in summary.index:
            if param == "Intercept" or param in available_configs:
                mean_val = summary.loc[param, "mean"]
                sd_val = summary.loc[param, "sd"]
                # arviz uses hdi_2.5% and hdi_97.5% by default
                hdi_lower = summary.loc[param, "hdi_2.5%"]
                hdi_upper = summary.loc[param, "hdi_97.5%"]
                r_hat = summary.loc[param, "r_hat"]

                lines.append(
                    f"| {param} | {mean_val:.4f} | {sd_val:.4f} | "
                    f"[{hdi_lower:.4f}, {hdi_upper:.4f}] | {r_hat:.3f} |"
                )

        lines.append("")
        lines.append("*HDI: Highest Density Interval (Bayesian credible interval)*")
        lines.append("*R-hat: Convergence diagnostic (should be < 1.01)*")
        lines.append("")

        # Interpretation with probabilities
        lines.append("**Interpretation:**")
        lines.append("")

        # Access posterior samples for probability calculations
        posterior = idata.posterior

        for config in available_configs:
            if config in summary.index:
                mean_val = summary.loc[config, "mean"]
                hdi_lower = summary.loc[config, "hdi_2.5%"]
                hdi_upper = summary.loc[config, "hdi_97.5%"]

                # Calculate probability of increase/decrease from posterior samples
                if config in posterior.data_vars:
                    # Get all posterior samples for this parameter
                    samples = posterior[config].values.flatten()
                    prob_positive = (
                        samples > 0
                    ).mean() * 100  # Probability of increase
                    prob_negative = (
                        samples < 0
                    ).mean() * 100  # Probability of decrease

                    # Check if credible interval excludes zero
                    if hdi_lower > 0 and hdi_upper > 0:
                        arrow = "⬆️"
                        lines.append(
                            f"- {arrow} **{config}:** Increases F1 by {abs(mean_val):.4f} "
                            f"(95% HDI excludes 0)"
                        )
                        lines.append(
                            f"  - Probability of increase: {prob_positive:.1f}%, "
                            f"decrease: {prob_negative:.1f}%"
                        )
                    elif hdi_lower < 0 and hdi_upper < 0:
                        arrow = "⬇️"
                        lines.append(
                            f"- {arrow} **{config}:** Decreases F1 by {abs(mean_val):.4f} "
                            f"(95% HDI excludes 0)"
                        )
                        lines.append(
                            f"  - Probability of increase: {prob_positive:.1f}%, "
                            f"decrease: {prob_negative:.1f}%"
                        )
                    else:
                        lines.append(
                            f"- **{config}:** Effect uncertain (95% HDI includes 0)"
                        )
                        lines.append(
                            f"  - Probability of increase: {prob_positive:.1f}%, "
                            f"decrease: {prob_negative:.1f}%"
                        )
                else:
                    # Fallback if samples not available
                    if (hdi_lower > 0 and hdi_upper > 0) or (
                        hdi_lower < 0 and hdi_upper < 0
                    ):
                        direction = "increases" if mean_val > 0 else "decreases"
                        arrow = "⬆️" if mean_val > 0 else "⬇️"
                        lines.append(
                            f"- {arrow} **{config}:** {direction.capitalize()} F1 by "
                            f"{abs(mean_val):.4f} (95% HDI excludes 0)"
                        )
                    else:
                        lines.append(
                            f"- **{config}:** Effect uncertain (95% HDI includes 0)"
                        )

        lines.append("")

        # Random effects variance
        lines.append("#### Random Effects (Group-Level)")
        lines.append("")

        # Find group-level standard deviation in summary
        group_params = [
            idx for idx in summary.index if self.COLUMN_EXPERIMENT_ID in idx
        ]
        if group_params:
            # Limit to top 10 by absolute mean to fit on page
            group_data: list[tuple[Any, ...]] = []
            for param in group_params:
                mean_val = summary.loc[param, "mean"]
                sd_val = summary.loc[param, "sd"]
                hdi_lower = summary.loc[param, "hdi_2.5%"]
                hdi_upper = summary.loc[param, "hdi_97.5%"]
                group_data.append((param, mean_val, sd_val, hdi_lower, hdi_upper))

            # Sort by absolute mean (largest effects first)
            group_data.sort(key=lambda x: abs(x[1]), reverse=True)

            # Show only top 10 to fit on page
            lines.append(
                f"**Top 10 Experiment Random Effects** (out of {len(group_params)} total):"
            )
            lines.append("")
            lines.append("| Experiment | Mean | SD | 95% HDI |")
            lines.append("|------------|------|----|---------|")
            for param, mean_val, sd_val, hdi_lower, hdi_upper in group_data[:10]:
                # Shorten parameter name - extract just the experiment ID
                # Format is typically like "1|EXPERIMENT_ID[experiment_name]"
                short_name = param
                if "[" in param and "]" in param:
                    # Extract text between brackets
                    short_name = param[param.find("[") + 1 : param.find("]")]
                elif "|" in param:
                    # Take last part after |
                    short_name = param.split("|")[-1]

                lines.append(
                    f"| {short_name} | {mean_val:.4f} | {sd_val:.4f} | "
                    f"[{hdi_lower:.4f}, {hdi_upper:.4f}] |"
                )
            lines.append("")
            lines.append(
                "*Showing top 10 experiments with largest random effects. "
                "Full results available in the inference data object.*"
            )
            lines.append("")

        return lines

    def _generate_pdf_from_markdown(self, markdown_content: str, output_file: str):
        """
        Convert markdown content to PDF using markdown_it and weasyprint.

        Args:
            markdown_content: The markdown text to convert
            output_file: Path to the output PDF file
        """

        # Initialize markdown-it with table support
        md = MarkdownIt("commonmark", {"breaks": True, "html": True}).enable("table")

        # Convert markdown to HTML
        html_body = md.render(markdown_content)

        # Create a complete HTML document with styling
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Statistical Analysis Report</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #2980b9;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 8px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        h4 {{
            color: #7f8c8d;
            margin-top: 15px;
            page-break-after: avoid;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 0.85em;
            page-break-inside: avoid;
            table-layout: auto;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 8px;
            text-align: left;
            font-weight: bold;
            word-wrap: break-word;
        }}
        td {{
            padding: 6px;
            border: 1px solid #ddd;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-width: 200px;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin: 15px 0;
            background-color: #ecf0f1;
            padding: 10px 15px;
            font-style: italic;
            page-break-inside: avoid;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 30px 0;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
        p {{
            margin: 10px 0;
            text-align: justify;
        }}
        .page-break {{
            page-break-after: always;
        }}
        em {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        strong {{
            color: #2c3e50;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

        # Convert HTML to PDF
        HTML(string=html_template).write_pdf(output_file)  # type: ignore
        logger.info(f"PDF successfully generated: {output_file}")
