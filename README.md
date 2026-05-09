# **`README.md`**

# Constructing and Comparing Sentiment Spillover Networks

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.26811-b31b1b.svg)](https://arxiv.org/abs/2604.26811)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2604.26811)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Discipline: Econometrics](https://img.shields.io/badge/Discipline-Information%20Econometrics-00529B)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Discipline: Network Science](https://img.shields.io/badge/Discipline-Computational%20Network%20Science-00529B)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Discipline: Topology](https://img.shields.io/badge/Discipline-Combinatorial%20Topology-00529B)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Data Sources](https://img.shields.io/badge/Data-Bloomberg%20Terminal%20%7C%20News%20%26%20Social%20Media-lightgrey)](https://www.bloomberg.com/professional/solution/bloomberg-terminal/)
[![Core Method: Transfer Entropy](https://img.shields.io/badge/Method-Transfer%20Entropy-orange)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Core Method: Markov Bootstrap](https://img.shields.io/badge/Method-Markov%20Bootstrap-orange)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Core Method: PageRank](https://img.shields.io/badge/Method-Weighted%20PageRank-orange)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Core Method: MSA](https://img.shields.io/badge/Method-Maximum%20Spanning%20Arborescence-orange)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Optimization](https://img.shields.io/badge/Optimization-Chu--Liu%2FEdmonds-red)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-%2300529B.svg?style=flat)](https://networkx.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-%233F51B5.svg?style=flat)](https://www.statsmodels.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks)

**Repository:** `https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Do News and Social Media Tell the Same Story? Constructing and Comparing Sentiment Spillover Networks"** by:

*   **Fan Wu**
*   **Anqi Liu**
*   **Jing Chen**
*   **Yuhua Li**

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the rigorous ingestion and stochastic imputation of high-frequency Bloomberg sentiment data, to the non-parametric estimation of Transfer Entropy, and the extraction of complex topological backbones via Maximum Spanning Arborescences. The pipeline culminates in a rigorous comparative analysis, demonstrating that authoritative news and speculative social media generate fundamentally distinct topologies of information contagion within the U.S. technology sector.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `deploy_sentinel_system`](#key-callable-deploy_sentinel_system)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Wu et al. (2026). The core of this repository is the iPython Notebook `constructing_and_comparing_sentiment_spillover_networks_draft.ipynb`, which contains a comprehensive suite of orchestrated tasks to replicate the paper's findings.

The pipeline addresses a foundational vulnerability in behavioral finance: the reliance on linear, mean-based models (like VAR or Granger Causality) to map information flow, which fail to capture the non-Gaussian, heavy-tailed nature of investor sentiment. 

The codebase operationalizes the proposed solution—a **High-Frequency "Sentinel" System for Information Topology**:
-   **Anchors** sentiment measurement in confidence-weighted aggregation and stochastic decay imputation, preserving the non-linear fade of investor memory.
-   **Enforces** model-free causality detection using Shannon Entropy and Normalized Transfer Entropy, capturing directional information flow without distributional assumptions.
-   **Propagates** the analysis to the systemic level by constructing dynamic, rolling-window adjacency matrices, filtered for statistical significance via rigorous Markov bootstrapping.
-   **Evaluates** the topological structure through recursive centrality (PageRank) to identify "Information Hubs" and combinatorial optimization (Chu-Liu/Edmonds) to extract the dominant paths of sentiment contagion.

## Theoretical Background

The implemented methods combine techniques from Information Econometrics, Computational Network Science, and Combinatorial Topology.

**1. Stochastic Imputation and Signal Synthesis:**
To handle asynchronous media volumes, missing data is modeled as an exponential decay function with additive Gaussian noise calibrated via an AR(1) process:
$$ x_t = B e^{-\lambda t} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2) $$

**2. Information-Theoretic Causality (Transfer Entropy):**
The framework abandons linear correlation in favor of Transfer Entropy, which quantifies the reduction in uncertainty of a target company's future sentiment ($Z$) provided by a source company's history ($Y$):
$$ T_{Y \to Z}^{k,l} = H(z_{t+1}|z_t^{(k)}) - H(z_{t+1}|z_t^{(k)}, y_t^{(l)}) $$
This is normalized to bound the intensity of information flow:
$$ \hat{T}_{Y \to Z} = \frac{T_{Y \to Z}}{H(z_{t+1}|z_t)} $$

**3. Systemic Influence and Centrality:**
The framework shifts the analytical unit from pairwise links to global influence using the Weighted PageRank algorithm, identifying nodes whose sentiment shifts have the most profound "ripple effect":
$$ PR(v_i) = \frac{1-f}{n} + f \sum_{v_j \in V_m} \frac{\hat{T}_{ji}}{d_{v_j}^{out}} PR(v_j) $$

**4. Dominant Path Extraction (Maximum Spanning Arborescence):**
To isolate the strongest chains of information diffusion, the framework solves a combinatorial optimization problem rooted at the node with peak out-degree:
$$ \max \sum_{(v_i, v_j) \in \mathbb{T}} w(v_i, v_j) $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks/blob/main/constructing_and_comparing_sentiment_spillover_networks_ipo_main.png" alt="Sentinel System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook implements the full research pipeline, including:

-   **Strict Temporal Invariants:** Enforces a mathematically guaranteed $T=1319$ master calendar using `pandas_market_calendars`, preventing the silent erasure of missing days.
-   **Single Source of Truth Indexing:** Utilizes an $O(1)$ `ticker_to_position` map to guarantee that matrix indices are perfectly synchronized with corporate entities across all centrality metrics and arborescences.
-   **Stochastic Reproducibility:** Employs cryptographic hashing (MD5) to generate deterministic seeds for the `PCG64` random number generator, ensuring perfect reproducibility of the Markov bootstrap across parallel workers.
-   **Exact Graph Optimization:** Replaces brittle structural heuristics with the `networkx` Edmonds algorithm, explicitly enforcing root constraints to guarantee 100% topological accuracy during MSA extraction.
-   **Rigorous Null Model Aggregation:** Computes empirical transition matrices without artificial smoothing, ensuring the Markov bootstrap relies solely on observed state dynamics.
-   **Configuration-Driven Design:** All study parameters, discretization thresholds, and topological policies are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preprocessing (Tasks 1-8):** Ingests raw Bloomberg data. Enforces strict temporal alignment, alias normalization, and missingness profiling across the 34-asset universe.
2.  **Stochastic Imputation (Tasks 9-12):** Diagnoses AR(1) plausibility, estimates innovation variance, and applies exponential decay imputation with additive Gaussian noise.
3.  **Information Quantification (Tasks 13-16):** Discretizes continuous signals via empirical quantiles, calibrates optimal lags, and computes raw and normalized Transfer Entropy for all 1,122 pairs.
4.  **Significance Filtering (Tasks 17-19):** Generates degree-preserving null networks via parallelized Markov Chains and filters the adjacency matrices at the $\alpha=0.10$ significance level.
5.  **Topological Analysis (Tasks 20-22):** Computes dynamic network density, weighted in/out-degrees, and recursive PageRank centrality to identify Information Hubs.
6.  **Regime & Path Optimization (Tasks 23-25):** Segments the timeline into three regimes, aggregates networks, and extracts the Maximum Spanning Arborescence (MSA) backbones.
7.  **Validation & Audit (Tasks 26-29):** Computes Jaccard similarity, evaluates spectral radius for convergence, generates publication-ready tables/figures, and programmatically audits the empirical outputs against the manuscript's core findings.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 29 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `deploy_sentinel_system`

The project is designed around a single, top-level user-facing interface function:

-   **`deploy_sentinel_system`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, stochastic imputation, TE estimation, topological extraction, and the final reproduction audit, while persisting all artifacts to disk.

## Prerequisites

-   Python 3.10+
-   Core Python dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `networkx`, `joblib`, `matplotlib`, `seaborn`, `pyyaml`, `pandas_market_calendars`, `Faker`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks.git
    cd constructing_and_comparing_sentiment_spillover_networks
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install numpy pandas scipy statsmodels networkx joblib matplotlib seaborn pyyaml pandas_market_calendars Faker
    ```

## Input Data Structure

The pipeline requires two strictly formatted `pandas.DataFrame` objects (`df_news_sentiment_raw` and `df_social_sentiment_raw`) containing the following columns:
-   `Date`: `datetime64[ns]` (Discrete daily timestamp index).
-   `Ticker`: `string` (Canonical ticker identifier).
-   `Sentiment_Score`: `float64` (Daily confidence-weighted score, range [-1, 1]).
-   `Story_Count_K` / `Post_Count_K`: `Int64` (Total volume of items, nullable integer).

*Note: The pipeline includes a high-fidelity synthetic data generator that simulates continuous sentiment scores and discrete item counts, injecting realistic missingness for testing purposes.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic benchmark data, and use the top-level orchestrator to execute the pipeline:

```python
import pandas as pd
import numpy as np
import yaml
from faker import Faker
from typing import Dict, Any, Tuple

# ==============================================================================
# Step 1: Synthetic Data Generation
# ==============================================================================
# Initialize Faker and Numpy RNG for perfect reproducibility
fake = Faker()
Faker.seed(123456)
np.random.seed(123456)

tech_universe_raw: Dict[str, Any] = {
    "Tickers": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN",
        "V", "MA", "ADBE", "PYPL", "CRM", "CSCO", "INTC",
        "AVGO", "ORCL", "ACN", "AMD", "TXN", "QCOM", "INTU",
        "NOW", "IBM", "SPGI", "AMAT", "ADP", "ADI", "LRCX",
        "SQ", "MU", "MCO", "FI", "FIS", "MCHP"
    ],
    "Ticker_Aliases": {"SQ": ["XYZ"]},
    "Adjacency_Index_Order": "Tickers"
}

def generate_synthetic_sentiment_panels(
    universe: Dict[str, Any], 
    start_date: str = "2019-09-30", 
    n_days: int = 1319
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Generates high-fidelity synthetic sentiment panels for News and Social Media."""
    tickers = universe["Tickers"]
    master_calendar = pd.bdate_range(start=start_date, periods=n_days)
    idx = pd.MultiIndex.from_product([master_calendar, tickers], names=["Date", "Ticker"])
    
    def build_panel(count_col_name: str) -> pd.DataFrame:
        df = pd.DataFrame(index=idx).reset_index()
        df["Sentiment_Score"] = np.random.uniform(-1.0, 1.0, size=len(df))
        df[count_col_name] = np.random.poisson(lam=50, size=len(df))
        
        missing_mask = np.random.rand(len(df)) < 0.15
        df.loc[missing_mask, "Sentiment_Score"] = np.nan
        df.loc[missing_mask, count_col_name] = 0
        
        df["Date"] = df["Date"].astype("datetime64[ns]")
        df["Ticker"] = df["Ticker"].astype(str)
        df["Sentiment_Score"] = df["Sentiment_Score"].astype("float64")
        df[count_col_name] = df[count_col_name].astype("Int64")
        return df

    return build_panel("Story_Count_K"), build_panel("Post_Count_K"), master_calendar

df_news_raw, df_social_raw, fallback_cal = generate_synthetic_sentiment_panels(tech_universe_raw)

# ==============================================================================
# Step 2: Loading the Configuration (config.yaml)
# ==============================================================================
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    """Loads the deterministic hyperparameters and topological metrics."""
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

study_config = load_study_configuration("config.yaml")

# ==============================================================================
# Step 3: Executing the Pipeline
# ==============================================================================
if __name__ == "__main__":
    print("\nInitiating High-Frequency Sentinel System Deployment...")
    output_dir = "./sentinel_paper_artifacts"
    
    try:
        # Execute the deployment script
        # Note: deploy_sentinel_system is defined in the Jupyter Notebook
        deploy_sentinel_system(
            tech_universe_raw=tech_universe_raw,
            df_sector_mapping=df_sector_mapping, # Assumed generated
            df_news_sentiment_raw=df_news_raw,
            df_social_sentiment_raw=df_social_raw,
            study_configuration=study_config,
            output_directory=output_dir
        )
        print(f"\nDEPLOYMENT COMPLETE. Artifacts saved to: {output_dir}")
    except RuntimeError as e:
        print(f"\nDeployment failed: {e}")
```

## Output Structure

The pipeline serializes artifacts to an `output_directory` containing:
-   **`Table_1_News_Stats.csv` & `Table_2_Social_Stats.csv`**: Descriptive statistics and stationarity diagnostics.
-   **`Table_3_Top_5_Influential.csv`**: The persistent influence rankings (PageRank, In/Out-Degree).
-   **`Figure_5_Density.pdf`**: Time-varying network density plots.
-   **`Figure_7_Jaccard.pdf`**: Structural similarity comparison over time.
-   **`Figure_9_Heatmaps_News.pdf` & `Figure_10_Heatmaps_Social.pdf`**: High-dimensional nodal intensity heatmaps.
-   **`fidelity_diagnostic_log.txt`**: A systematic audit report verifying the replication of qualitative invariants (e.g., the News density spike).

## Project Structure

```
constructing_and_comparing_sentiment_spillover_networks/
│
├── constructing_and_comparing_sentiment_spillover_networks_draft.ipynb    # Main implementation notebook
├── config.yaml                                                            # Master configuration file
├── requirements.txt                                                       # Python package dependencies
│
├── LICENSE                                                                # MIT Project License File
└── README.md                                                              # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Stochastic Settings:** Adjust the `decay_rate_lambda` or the `ar_model_order` for the imputation engine.
-   **Information Theory:** Modify the `optimal_lag_k` or `quantile_levels` to test structural sensitivity.
-   **Inference Rigor:** Increase the `n_boot` for the Markov bootstrap or tighten the `alpha` threshold for significance filtering.
-   **Topological Constraints:** Adjust the `pagerank.damping_factor_f` or modify the `regime_breakpoints` for temporal analysis.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions, as suggested by the authors, could include:
-   **Cross-Industry Spillovers:** Expanding the vertex set beyond the technology sector to map inter-industry sentiment contagion.
-   **Dynamic Lead-Lag Relationships:** Investigating whether the news spillover network systematically leads the social media network (or vice versa) using cross-correlation of the density time series.
-   **Systemic Risk Indices:** Integrating the aggregated network density metric as a real-time feature in asset pricing or market crash prediction models.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{wu2026sentiment,
  title={Do News and Social Media Tell the Same Story? Constructing and Comparing Sentiment Spillover Networks},
  author={Wu, Fan and Liu, Anqi and Chen, Jing and Li, Yuhua},
  journal={arXiv preprint arXiv:2604.26811},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Constructing and Comparing Sentiment Spillover Networks.
GitHub repository: https://github.com/chirindaopensource/constructing_and_comparing_sentiment_spillover_networks
```

## Acknowledgments

-   Credit to **Fan Wu, Anqi Liu, Jing Chen, and Yuhua Li** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **NumPy**, **Pandas**, **Statsmodels**, and **NetworkX** contributors.
```
