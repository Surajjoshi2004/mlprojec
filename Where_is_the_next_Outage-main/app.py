import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

st.set_page_config(page_title="Power Outage Prediction", layout="wide")

st.title("Where is the next power outage?")
st.markdown("Predict whether a power outage occurs in the **West Climate region** using historical outage data (2000\u20132016).")

@st.cache_data
def load_and_clean_data():
    rows_to_skip = list(range(5))
    df = pd.read_csv('data/outage.csv', skiprows=rows_to_skip, index_col='OBS')
    column = np.array(df.columns).astype('str')
    unites = np.array(df.iloc[0].fillna('')).astype('str')
    unites = ["(" + i + ")" for i in unites]
    for i in range(len(unites)):
        if unites[i] == '()':
            unites[i] = ''
    combined_column = np.core.defchararray.add(column, unites)
    df.columns = combined_column
    df = df.reset_index().drop(0).drop('variables(Units)', axis=1).reset_index(drop=True)
    df['YEAR'] = df['YEAR'].astype('int')
    start_time = df['OUTAGE.START.TIME(Hour:Minute:Second (AM / PM))']
    start_date = df['OUTAGE.START.DATE(Day of the week, Month Day, Year)']
    df["OUTAGE.START"] = pd.to_datetime(start_date + " " + start_time)
    end_time = df['OUTAGE.RESTORATION.TIME(Hour:Minute:Second (AM / PM))']
    end_date = df['OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)']
    df["OUTAGE.RESTORATION"] = pd.to_datetime(end_date + " " + end_time)
    df['ANOMALY.LEVEL'] = df['ANOMALY.LEVEL(numeric)'].astype(float)
    df = df.drop(columns=[
        'ANOMALY.LEVEL(numeric)',
        'OUTAGE.START.DATE(Day of the week, Month Day, Year)',
        'OUTAGE.START.TIME(Hour:Minute:Second (AM / PM))',
        'OUTAGE.RESTORATION.DATE(Day of the week, Month Day, Year)',
        'OUTAGE.RESTORATION.TIME(Hour:Minute:Second (AM / PM))'
    ], axis=1)
    cols_float = [
        'OUTAGE.DURATION(mins)', 'DEMAND.LOSS.MW(Megawatt)',
        'RES.PRICE(cents / kilowatt-hour)', 'COM.PRICE(cents / kilowatt-hour)',
        'IND.PRICE(cents / kilowatt-hour)', 'TOTAL.PRICE(cents / kilowatt-hour)',
        'RES.SALES(Megawatt-hour)', 'COM.SALES(Megawatt-hour)',
        'IND.SALES(Megawatt-hour)', 'TOTAL.SALES(Megawatt-hour)',
        'RES.PERCEN(%)', 'COM.PERCEN(%)', 'IND.PERCEN(%)',
        'RES.CUST.PCT(%)', 'COM.CUST.PCT(%)', 'IND.CUST.PCT(%)',
        'PC.REALGSP.STATE(USD)', 'PC.REALGSP.USA(USD)',
        'PC.REALGSP.REL(fraction)', 'PC.REALGSP.CHANGE(%)',
        'UTIL.REALGSP(USD)', 'TOTAL.REALGSP(USD)', 'UTIL.CONTRI(%)',
        'PI.UTIL.OFUSA(%)', 'POPPCT_URBAN(%)', 'POPPCT_UC(%)',
        'POPDEN_URBAN(persons per square mile)', 'POPDEN_UC(persons per square mile)',
        'POPDEN_RURAL(persons per square mile)', 'AREAPCT_URBAN(%)', 'AREAPCT_UC(%)',
        'PCT_LAND(%)', 'PCT_WATER_TOT(%)', 'PCT_WATER_INLAND(%)', 'CUSTOMERS.AFFECTED'
    ]
    for c in cols_float:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.drop('OBS', axis=1)
    df = df.drop(columns=[
        'POSTAL.CODE', 'OUTAGE.START', 'OUTAGE.RESTORATION', 'U.S._STATE',
        'HURRICANE.NAMES', 'POPDEN_URBAN(persons per square mile)',
        'POPPCT_URBAN(%)', 'POPPCT_UC(%)', 'POPDEN_UC(persons per square mile)',
        'POPDEN_RURAL(persons per square mile)', 'AREAPCT_URBAN(%)',
        'PCT_LAND(%)', 'PCT_WATER_TOT(%)', 'PCT_WATER_INLAND(%)', 'AREAPCT_UC(%)'
    ])
    mean_dur = df.groupby('CLIMATE.REGION')['OUTAGE.DURATION(mins)'].mean()
    df['OUTAGE.DURATION(mins)'].fillna(df['CLIMATE.REGION'].map(mean_dur), inplace=True)
    df.at[1533, 'OUTAGE.DURATION(mins)'] = 1860
    df['CLIMATE.REGION'] = (df['CLIMATE.REGION'] == 'West').astype(int)
    return df

df = load_and_clean_data()

train_set = df[df['YEAR'] <= 2012]
test_set = df.drop(train_set.index)

X_train = train_set.drop('CLIMATE.REGION', axis=1)
y_train = train_set['CLIMATE.REGION']
X_test = test_set.drop('CLIMATE.REGION', axis=1)
y_test = test_set['CLIMATE.REGION']

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Data Exploration", "Baseline Model", "Final Model", "Fairness Analysis"
])

with tab1:
    st.header("Project Overview")
    st.markdown("""
    **Problem:** Predict whether a power outage occurs in the **West Climate region** using historical outage data.

    **Data:** Major power outage events in the continental U.S. from **January 2000 to July 2016** (1534 records, 56 features).

    **Approach:**
    - **Baseline:** Logistic Regression with 3 features (NERC.REGION, CAUSE.CATEGORY, OUTAGE.DURATION)
    - **Final Model:** Random Forest Classifier with tuned hyperparameters using 7 features

    **Key Results:**
    - Baseline Test Accuracy: **90.29%**
    - Final Model Test Accuracy: **~93.5%**
    - Final Model Recall: **~0.95**
    - Final Model Precision: **~0.67**
    """)

with tab2:
    st.header("Data Exploration")
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution (West Climate Region)")
        fig, ax = plt.subplots()
        df['CLIMATE.REGION'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_xticklabels(['Non-West', 'West'])
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Outages by Year")
        fig, ax = plt.subplots()
        year_counts = df['YEAR'].value_counts().sort_index()
        year_counts.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Climate Region Distribution")
    st.markdown("""
    - Northeast: 350
    - South: 229
    - West: 217
    - Central: 200
    - Southeast: 153
    - East North Central: 138
    - Northwest: 132
    - Southwest: 92
    - West North Central: 17
    """)

with tab3:
    st.header("Baseline Model: Logistic Regression")

    st.markdown("""
    **Features used:**
    - `NERC.REGION` (one-hot encoded)
    - `CAUSE.CATEGORY` (one-hot encoded)
    - `OUTAGE.DURATION(mins)` (passed through)

    **Rationale:** Simple, interpretable model to establish a performance benchmark.
    """)

    baseline_features = ['NERC.REGION', 'CAUSE.CATEGORY', 'OUTAGE.DURATION(mins)']
    X_train_base = train_set[baseline_features]
    X_test_base = test_set[baseline_features]
    y_train_base = train_set['CLIMATE.REGION']
    y_test_base = test_set['CLIMATE.REGION']

    categorical_cols_base = ['NERC.REGION', 'CAUSE.CATEGORY']
    preprocessor_base = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_cols_base)
    ], remainder='passthrough')

    pipe_base = Pipeline([
        ('prep', preprocessor_base),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_base.fit(X_train_base, y_train_base)
    y_pred_base = pipe_base.predict(X_test_base)

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Accuracy", f"{accuracy_score(y_train_base, pipe_base.predict(X_train_base)):.2%}")
    col2.metric("Test Accuracy", f"{accuracy_score(y_test_base, y_pred_base):.2%}")
    col3.metric("Test Recall", f"{recall_score(y_test_base, y_pred_base):.2%}")

    cm = confusion_matrix(y_test_base, y_pred_base)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-West', 'West'],
                yticklabels=['Non-West', 'West'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Baseline Model")
    st.pyplot(fig)

    st.markdown(f"""
    - True Positives: {cm[1,1]}
    - False Negatives: {cm[1,0]}
    - True Negatives: {cm[0,0]}
    - False Positives: {cm[0,1]}
    - Precision: **{precision_score(y_test_base, y_pred_base):.3f}**
    - Recall: **{recall_score(y_test_base, y_pred_base):.3f}**
    """)

with tab4:
    st.header("Final Model: Random Forest Classifier")

    st.markdown("""
    **Features used:**
    - `CLIMATE.CATEGORY` (one-hot encoded)
    - `CAUSE.CATEGORY.DETAIL` (one-hot encoded)
    - `CAUSE.CATEGORY` (one-hot encoded)
    - `PC.REALGSP.STATE(USD)` (standardized)
    - `OUTAGE.DURATION(mins)` (standardized)
    - `PI.UTIL.OFUSA(%)` (standardized)

    **Why Random Forest?**
    - Handles categorical features well
    - More tunable hyperparameters than Logistic Regression
    - Better with imbalanced data
    """)

    final_features = [
        'CLIMATE.CATEGORY', 'CAUSE.CATEGORY.DETAIL', 'CAUSE.CATEGORY',
        'PC.REALGSP.STATE(USD)', 'OUTAGE.DURATION(mins)', 'PI.UTIL.OFUSA(%)'
    ]
    available_feats = [f for f in final_features if f in train_set.columns]
    X_train_final = train_set[available_feats]
    y_train_final = train_set['CLIMATE.REGION']
    X_test_final = test_set[available_feats]
    y_test_final = test_set['CLIMATE.REGION']

    categorical_cols_final = ['CLIMATE.CATEGORY', 'CAUSE.CATEGORY.DETAIL', 'CAUSE.CATEGORY']
    categorical_cols_final = [c for c in categorical_cols_final if c in available_feats]
    numeric_cols_final = [c for c in ['PC.REALGSP.STATE(USD)', 'OUTAGE.DURATION(mins)', 'PI.UTIL.OFUSA(%)'] if c in available_feats]

    preprocessor_final = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_cols_final),
        ('scaler', StandardScaler(), numeric_cols_final)
    ])

    pipe_final = Pipeline([
        ('prep', preprocessor_final),
        ('rft', RandomForestClassifier(
            n_estimators=250, max_features=9, min_samples_split=2,
            random_state=42, n_jobs=-1
        ))
    ])
    pipe_final.fit(X_train_final, y_train_final)
    y_pred_final = pipe_final.predict(X_test_final)

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Score", f"{pipe_final.score(X_train_final, y_train_final):.2%}")
    col2.metric("Test Score (post-2013)", f"{accuracy_score(y_test_final, y_pred_final):.2%}")
    col3.metric("Recall", f"{recall_score(y_test_final, y_pred_final):.3f}")

    cm_final = confusion_matrix(y_test_final, y_pred_final)
    fig, ax = plt.subplots()
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Non-West', 'West'],
                yticklabels=['Non-West', 'West'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Final Model")
    st.pyplot(fig)

    st.markdown(f"""
    - True Positives: {cm_final[1,1]}
    - False Negatives: {cm_final[1,0]}
    - True Negatives: {cm_final[0,0]}
    - False Positives: {cm_final[0,1]}
    - Precision: **{precision_score(y_test_final, y_pred_final):.3f}**
    - Recall: **{recall_score(y_test_final, y_pred_final):.3f}**
    """)

    st.subheader("Try a Prediction")
    st.markdown("Adjust the sliders and selectors to see if an outage would be predicted in the West Climate region.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            category = st.selectbox("Cause Category", sorted(train_set['CAUSE.CATEGORY'].dropna().unique()))
            detail = st.selectbox("Cause Detail", sorted(train_set['CAUSE.CATEGORY.DETAIL'].dropna().unique()))
            climate_cat = st.selectbox("Climate Category", sorted(train_set['CLIMATE.CATEGORY'].dropna().unique()))
        with col_b:
            duration = st.slider("Outage Duration (mins)", 0, 10000, 500)
            gsp = st.slider("PC.REALGSP.STATE (USD)", 20000, 80000, 50000)
            util = st.slider("PI.UTIL.OFUSA (%)", 0.0, 10.0, 2.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            'CLIMATE.CATEGORY': climate_cat,
            'CAUSE.CATEGORY.DETAIL': detail,
            'CAUSE.CATEGORY': category,
            'PC.REALGSP.STATE(USD)': gsp,
            'OUTAGE.DURATION(mins)': duration,
            'PI.UTIL.OFUSA(%)': util
        }])
        input_df = input_df[available_feats]
        pred = pipe_final.predict(input_df)[0]
        proba = pipe_final.predict_proba(input_df)[0]
        if pred == 1:
            st.error(f"**West Climate Region** (confidence: {proba[1]:.1%})")
        else:
            st.success(f"**Non-West Climate Region** (confidence: {proba[0]:.1%})")

with tab5:
    st.header("Fairness Analysis")
    st.markdown("""
    **Groups:** Power outages affecting >50,000 customers (severe) vs \u226450,000 customers (less severe).

    **Metric 1: Accuracy Difference**
    - Null Hypothesis: Model accuracy is similar for both groups.
    - P-value from permutation test (5000 iterations): **0.1286**
    - Conclusion: Fail to reject null hypothesis \u2014 model appears fair based on accuracy.

    **Metric 2: R-squared Difference**
    - P-value from permutation test (5000 iterations): **0.0112**
    - Note: This is close to the significance level (0.01), suggesting potential concern.

    > Further testing with more data is recommended to verify fairness.
    """)

    st.caption("See the original notebook's HTML files (`asset/fairness1.html`, `asset/fairness2.html`) for detailed permutation test plots.")
