"""Student Performance Studio, a Streamlit companion to the regression notebook.
Run with: streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

st.set_page_config(page_title="Student Performance Studio", page_icon="🎓", layout="wide")
TEAL = "#087f8c"
GOLD = "#dba347"
TARGET = "Exam_Score"
DATA_PATH = Path(__file__).parent / "StudentPerformanceFactors.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#142d39;--muted:#657784;--teal:#087f8c;}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f7fa;color:var(--ink);}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.04em!important;color:var(--ink)!important;}
h1 {font-size:2.7rem!important;font-weight:800!important;line-height:1.15!important;}
h2 {font-size:1.55rem!important;}
h3 {font-size:1.1rem!important;}
.block-container {max-width:1450px;padding-top:2.4rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:#102d39;}
[data-testid="stSidebar"] * {color:#e9f0f2!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#a7bec8!important;}
[data-testid="stSidebar"] .stRadio label {padding:9px 2px;}
[data-testid="stSidebar"] hr {border-color:#33505c;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e9ed;border-radius:15px;padding:20px 22px;box-shadow:0 4px 16px #183b4e04;}
[data-testid="stMetricLabel"] {color:#657784;font-size:.85rem;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-weight:800;font-size:2rem;color:#142d39;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px!important;border-color:#e2e9ed!important;}
[data-testid="stForm"] {background:white;border-radius:16px;}
.stButton>button[kind="primary"],.stFormSubmitButton>button {background:#087f8c;border:none;border-radius:9px;padding:.65rem 1rem;color:white!important;}
.eyebrow {font-size:.72rem;letter-spacing:.17em;color:#087f8c;font-weight:800;margin-bottom:12px;}
.subtitle {font-size:1rem;color:#657784;max-width:760px;line-height:1.7;margin:0 0 25px;}
.hero {background:linear-gradient(120deg,#123644,#1a5260);padding:29px 32px;border-radius:18px;color:white;margin:22px 0;}
.hero h2,.hero p {color:white!important;margin:0;}
.hero p {color:#c1d9df!important;line-height:1.7;margin-top:9px;max-width:780px;}
.pill {display:inline-block;padding:5px 10px;border-radius:100px;background:#e5f4f1;color:#08766d;font-size:.73rem;font-weight:700;margin-bottom:9px;}
.score-card {background:linear-gradient(135deg,#103846,#087f8c);color:white;border-radius:18px;padding:30px;margin-top:10px;}
.score-card .big {font-size:4rem;font-weight:800;letter-spacing:-.06em;line-height:1.2;}
.score-card p {color:#d3e9ec;}
.foot {font-size:.76rem;color:#71838c;border-top:1px solid #dfe7eb;padding-top:18px;margin-top:30px;line-height:1.8;}
@media(max-width:640px) {h1{font-size:2rem!important;}.block-container{padding:1.3rem 1rem;}.hero{padding:22px;}[data-testid="stMetric"]{padding:15px;}.score-card .big{font-size:3rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Keep cleaning aligned with the supplied notebook."""
    raw = pd.read_csv(DATA_PATH)
    deduplicated = raw.drop_duplicates()
    clean = deduplicated[deduplicated[TARGET] <= 100].reset_index(drop=True)
    return raw, clean


@st.cache_resource(show_spinner="Training the two notebook models. This only happens once...")
def train_models():
    _, data = load_data()
    X, y = data.drop(columns=TARGET), data[TARGET]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = X_train.copy(), X_test.copy()
    fills = {}
    categories = {}
    for col in categorical:
        fills[col] = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(fills[col])
        X_test[col] = X_test[col].fillna(fills[col])
        # Freeze the training categories for single-row inference. Calling get_dummies
        # on one row with drop_first=True would otherwise erase its category.
        categories[col] = sorted(X_train[col].unique().tolist())
    for col in numeric:
        fills[col] = X_train[col].median()
        X_train[col] = X_train[col].fillna(fills[col])
        X_test[col] = X_test[col].fillna(fills[col])
    encoded_train = pd.get_dummies(X_train, columns=categorical, drop_first=True, dtype=int)
    encoded_test = pd.get_dummies(X_test, columns=categorical, drop_first=True, dtype=int)
    encoded_test = encoded_test.reindex(columns=encoded_train.columns, fill_value=0)
    scaler = StandardScaler()
    scaled_train, scaled_test = encoded_train.copy(), encoded_test.copy()
    scaled_train[numeric] = scaler.fit_transform(encoded_train[numeric])
    scaled_test[numeric] = scaler.transform(encoded_test[numeric])
    linear = LinearRegression().fit(scaled_train, y_train)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    poly_train = poly_features.fit_transform(scaled_train)
    poly_test = poly_features.transform(scaled_test)
    polynomial = LinearRegression().fit(poly_train, y_train)
    models = {"Linear Regression": linear, "Polynomial Regression": polynomial}
    predictions = {"Linear Regression": linear.predict(scaled_test),
                   "Polynomial Regression": polynomial.predict(poly_test)}
    rows = []
    for name, model in models.items():
        pred = predictions[name]
        train_matrix = scaled_train if name == "Linear Regression" else poly_train
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "MSE": mean_squared_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "Test R²": r2_score(y_test, pred),
                     "Train R²": r2_score(y_train, model.predict(train_matrix))})
    return dict(models=models, predictions=predictions, metrics=pd.DataFrame(rows),
                numeric=numeric, categorical=categorical, fills=fills, categories=categories,
                columns=encoded_train.columns.tolist(), scaler=scaler, poly=poly_features,
                y_test=y_test, X_test=X_test, train_rows=len(y_train), test_rows=len(y_test))


def encode_profile(frame, bundle):
    """Reuse fitted preprocessing, never refit on prediction input."""
    frame = frame.copy()
    for col in bundle["categorical"]:
        frame[col] = pd.Categorical(frame[col].fillna(bundle["fills"][col]),
                                    categories=bundle["categories"][col])
    for col in bundle["numeric"]:
        frame[col] = frame[col].fillna(bundle["fills"][col])
    encoded = pd.get_dummies(frame, columns=bundle["categorical"], drop_first=True, dtype=int)
    encoded = encoded.reindex(columns=bundle["columns"], fill_value=0)
    encoded[bundle["numeric"]] = bundle["scaler"].transform(encoded[bundle["numeric"]])
    return encoded


def chart_style(fig, height=310):
    fig.update_layout(template="plotly_white", height=height,
                      font=dict(family="DM Sans, sans-serif", color="#526773"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=15, t=15, b=10),
                      colorway=[TEAL, GOLD], legend_title_text="")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eaf0f3", zeroline=False)
    return fig


def plot(fig, height=310):
    st.plotly_chart(chart_style(fig, height), width="stretch",
                    config={"displayModeBar": False})


def heading(kicker, title, text):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtitle">{text}</p>', unsafe_allow_html=True)


if not DATA_PATH.exists():
    st.error("Dataset not found. Place StudentPerformanceFactors.csv beside app.py, then restart the app.")
    st.stop()
raw, data = load_data()
bundle = train_models()
metrics = bundle["metrics"]
best = metrics.loc[metrics["RMSE"].idxmin()]

with st.sidebar:
    st.markdown("### ◈ &nbsp; STUDENT STUDIO", unsafe_allow_html=True)
    st.caption("PERFORMANCE INTELLIGENCE")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Predict a score", "Model lab", "Explore data"], label_visibility="collapsed")
    st.divider()
    st.markdown("**THE PROJECT**")
    st.caption("Linear & Polynomial Regression\n\n19 input features · 1 exam score\n\n80/20 split · random state 42")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built by Rahma Hesham\n\nAn educational machine learning project.")

if page == "Overview":
    heading("STUDENT PERFORMANCE / OVERVIEW", "Small signals. Better understanding.",
            "Explore the factors behind exam performance, compare two regression models, and turn a student profile into a score estimate.")
    cols = st.columns(4)
    for col, label, value in zip(cols,
        ["Students in analysis", "Input features", "Best test R²", "Average absolute error"],
        [f"{len(data):,}", "19", f"{best['Test R²']:.3f}", f"{best['MAE']:.2f} pts"]):
        col.metric(label, value)
    st.markdown(f'<div class="hero"><div class="eyebrow" style="color:#81d6d5">THE MODEL VERDICT</div><h2>{best["Model"]} leads on the test set.</h2><p>More complexity does not always mean better predictions. Both models use the same held-out students; the winner has the lowest root mean squared error.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left, st.container(border=True):
        st.subheader("Where the scores land")
        st.caption("Distribution across the cleaned dataset")
        plot(px.histogram(data, x=TARGET, nbins=35, color_discrete_sequence=[TEAL],
                          labels={TARGET: "Exam score", "count": "Students"}))
    with right, st.container(border=True):
        st.subheader("Strongest numeric associations")
        st.caption("Pearson correlation with exam score, not cause and effect")
        correlations = data.select_dtypes("number").corr()[TARGET].drop(TARGET)
        correlations = correlations.reindex(correlations.abs().sort_values().index)
        corr = pd.DataFrame({"Feature": [x.replace("_", " ") for x in correlations.index], "Correlation": correlations.values})
        plot(px.bar(corr, x="Correlation", y="Feature", orientation="h", color_discrete_sequence=[TEAL]))
    st.caption(f"Data quality: {len(raw):,} original rows → {len(data):,} cleaned rows. Scores above 100 and duplicate rows are removed. Missing feature values are filled using training data only.")

elif page == "Predict a score":
    heading("STUDENT PERFORMANCE / PREDICTION", "Meet your next what-if.",
            "Set a student profile and estimate an exam score. Every input is used by the same models as the notebook.")
    st.info("For learning and demonstration only. Estimates are not guaranteed outcomes and must not be used for grading, admissions, or decisions about students.")
    with st.form("student_profile"):
        st.subheader("01 / Study habits")
        c1, c2, c3 = st.columns(3)
        profile = {}
        numeric_groups = [[("Hours_Studied", "Study hours per week", 1, 44, 20), ("Sleep_Hours", "Sleep hours per night", 4, 10, 7)],
                          [("Attendance", "Attendance (%)", 60, 100, 80), ("Tutoring_Sessions", "Tutoring sessions", 0, 8, 1)],
                          [("Previous_Scores", "Previous score", 50, 100, 75), ("Physical_Activity", "Physical activity hours", 0, 6, 3)]]
        for col, items in zip([c1, c2, c3], numeric_groups):
            with col:
                for key, label, low, high, value in items:
                    profile[key] = st.slider(label, low, high, value)
        st.subheader("02 / Learning environment")
        st.caption("Defaults are the most common values in the training set. Expand to change all 13 categorical features.")
        with st.expander("Personalize the learning environment"):
            columns = st.columns(3)
            for i, key in enumerate(bundle["categorical"]):
                options = bundle["categories"][key]
                with columns[i % 3]:
                    profile[key] = st.selectbox(key.replace("_", " "), options,
                                               index=options.index(bundle["fills"][key]))
        selected_model = st.selectbox("Prediction model", metrics["Model"].tolist())
        submitted = st.form_submit_button("Estimate exam score", type="primary", width="stretch")
    if submitted:
        frame = pd.DataFrame([profile])
        encoded = encode_profile(frame, bundle)
        matrix = encoded if selected_model == "Linear Regression" else bundle["poly"].transform(encoded)
        prediction = float(bundle["models"][selected_model].predict(matrix)[0])
        st.session_state["result"] = (prediction, selected_model, frame)
    if "result" in st.session_state:
        prediction, selected_model, frame = st.session_state["result"]
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.markdown(f'<div class="score-card"><div>ESTIMATED EXAM SCORE</div><div class="big">{prediction:.1f}<span style="font-size:1.2rem;letter-spacing:0"> / 100</span></div><p>{selected_model} · point estimate</p></div>', unsafe_allow_html=True)
        with col2:
            st.subheader("Read the estimate responsibly")
            row = metrics.set_index("Model").loc[selected_model]
            st.write(f"This model's test MAE is {row['MAE']:.2f} points and RMSE is {row['RMSE']:.2f} points. Individual errors can be much larger; these are evaluation metrics, not a confidence interval.")
            st.caption("The result stays tied to the last submitted profile. Submit again after changing inputs.")
            if not 0 <= prediction <= 100:
                st.warning("This regression estimate is outside the valid score range. It is shown without clipping so the limitation is visible.")
        export = frame.assign(Predicted_Score=prediction, Model=selected_model)
        st.download_button("Download this prediction (.csv)", export.to_csv(index=False), "student_prediction.csv", "text/csv")
    else:
        st.caption("Ready when you are. Adjust the profile, then select Estimate exam score.")

elif page == "Model lab":
    heading("STUDENT PERFORMANCE / MODEL LAB", "Evidence, before complexity.",
            "Compare both models on the exact same holdout set. All metrics below are calculated when the app trains, not typed in.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training students", f"{bundle['train_rows']:,}")
    c2.metric("Test students", f"{bundle['test_rows']:,}")
    c3.metric("Polynomial degree", "2")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(metrics.style.format({x: "{:.4f}" for x in metrics.columns if x != "Model"}), hide_index=True, width="stretch")
    st.caption("MAE, MSE, RMSE: lower is better. R²: higher is better. R² is explained variance relative to predicting the mean, not a percentage of correct predictions.")
    chosen = st.selectbox("Inspect a model", list(bundle["models"]))
    pred = bundle["predictions"][chosen]
    actual = bundle["y_test"].to_numpy()
    diagnostics = pd.DataFrame({"Actual score": actual, "Predicted score": pred, "Residual": actual - pred})
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Actual vs. predicted")
        fig = px.scatter(diagnostics, x="Actual score", y="Predicted score", opacity=.35, color_discrete_sequence=[TEAL])
        bounds = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", name="Perfect prediction", line=dict(color=GOLD, dash="dash")))
        plot(fig, 350)
    with right, st.container(border=True):
        st.subheader("Residual check")
        fig = px.scatter(diagnostics, x="Predicted score", y="Residual", opacity=.35, color_discrete_sequence=[TEAL])
        fig.add_hline(y=0, line_dash="dash", line_color=GOLD)
        plot(fig, 350)
    with st.expander("How the code matches the notebook"):
        st.markdown("""1. Remove duplicates and exam scores above 100.
2. Split **80% training / 20% testing**, with `random_state=42`.
3. Fill categorical values with the training mode and numerical values with the training median.
4. Apply `pd.get_dummies(drop_first=True)` and align the columns.
5. Fit `StandardScaler` to numerical training columns only.
6. Train `LinearRegression`, then a second model on `PolynomialFeatures(degree=2, include_bias=False)`.
7. Evaluate both on unseen test rows with MAE, MSE, RMSE, and R².

For a single new profile, categories are frozen to the training categories before encoding. No target score is used as an input. First-run training is cached; no model download is required.

Small changes in polynomial results can occur across numerical-library versions because one-hot and polynomial terms contain redundant columns. All displayed metrics come from the current fitted model.""")
    st.download_button("Download evaluation metrics (.csv)", metrics.to_csv(index=False), "model_metrics.csv", "text/csv")

else:
    heading("STUDENT PERFORMANCE / DATA EXPLORER", "Get closer to the data.",
            "Inspect the cleaned collection and explore associations. Charts and tables update together with the score filter.")
    score_range = st.slider("Exam score range", int(data[TARGET].min()), int(data[TARGET].max()),
                            (int(data[TARGET].min()), int(data[TARGET].max())))
    filtered = data[data[TARGET].between(*score_range)]
    st.caption(f"Showing {len(filtered):,} of {len(data):,} cleaned students. Source missing values remain visible here; model imputation happens separately after the split.")
    feature = st.selectbox("Explore a factor", bundle["numeric"] + bundle["categorical"])
    if filtered.empty:
        st.info("No students match this score range. Widen the filter to see data.")
    elif feature in bundle["numeric"]:
        plot(px.scatter(filtered, x=feature, y=TARGET, opacity=.25, color_discrete_sequence=[TEAL],
                        labels={feature: feature.replace("_", " "), TARGET: "Exam score"}), 350)
    else:
        plot(px.box(filtered, x=feature, y=TARGET, color_discrete_sequence=[TEAL],
                    labels={feature: feature.replace("_", " "), TARGET: "Exam score"}), 350)
    st.dataframe(filtered, hide_index=True, width="stretch", height=300)
    st.download_button("Download filtered data (.csv)", filtered.to_csv(index=False), "filtered_students.csv", "text/csv")
    with st.expander("Source data quality report"):
        st.write(f"Original rows: {len(raw):,} · duplicate rows: {raw.duplicated().sum():,} · scores above 100: {(raw[TARGET] > 100).sum():,}")
        missing = raw.isna().sum()
        st.dataframe(missing[missing > 0].rename("Missing values"), width="stretch")

st.markdown('<div class="foot">STUDENT PERFORMANCE STUDIO &nbsp; / &nbsp; Rahma Hesham<br>Academic demo · Association is not causation · A prediction is not a judgment of a student.</div>', unsafe_allow_html=True)
