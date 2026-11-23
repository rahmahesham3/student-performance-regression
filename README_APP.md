# Student Performance Studio

A Streamlit companion to the Student Performance Regression notebook, by Rahma Hesham.

## Run locally

Use Python 3.11 or 3.12. Keep `app.py`, `requirements.txt`, and `StudentPerformanceFactors.csv` in the same folder.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. First launch trains both models and may take a little longer; subsequent interactions reuse the cached models.

## Deploy on Streamlit Community Cloud

1. Add `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the dataset to the repository root. The notebook can stay there too.
2. In Streamlit Community Cloud, choose the repository and branch, with `app.py` as the main file.
3. Select Python 3.11 or 3.12 and deploy.
4. Open the resulting app link on a phone or laptop. A GitHub source-code link alone does not run a Streamlit app.

Deployment is a separate step and is not included in these files.

## Workspace

- **Overview:** cleaned dataset size, computed model metrics, score distribution, and numeric correlations.
- **Predict a score:** all 19 model inputs, a choice of model, a score estimate, and a downloadable prediction.
- **Model lab:** live train/test metrics, actual-vs-predicted and residual plots, and a notebook-method explanation.
- **Explore data:** score filter, factor exploration, data table, data-quality report, and CSV export.

## Method and alignment

The app reads the supplied CSV locally. It removes duplicates and scores above 100, then uses the notebook's 80/20 split with `random_state=42`. Missing values are filled using training modes or medians only. It uses pandas one-hot encoding with `drop_first=True`, standardizes numeric columns, and fits Linear Regression and degree-2 Polynomial Regression. No saved model or API key is required.

For single-row inference, category levels are frozen to the training set before one-hot encoding. This avoids the single-row `get_dummies` pitfall where every category would otherwise disappear with `drop_first=True`.

Metrics are computed from the current model, not hardcoded from a notebook screenshot. Polynomial models have redundant terms, so small differences can occur with numerical-library versions. The exact tested package versions are pinned in requirements.txt.

## Responsible use

This is an educational demonstration. It is not a validated tool for grading, admissions, counseling, or high-stakes student decisions. The source includes personal and socioeconomic factors that may encode bias. Correlation does not imply causation. A model's test MAE or RMSE is not a per-person confidence interval, and individual errors can be much larger. Scores are not clipped; out-of-range regression estimates are flagged explicitly.

The app does not write submitted profiles to disk. CSV download buttons let the visitor export data explicitly. Avoid submitting identifying student information to a public deployment. Dataset licensing and source attribution should be confirmed before redistribution beyond the existing project.
