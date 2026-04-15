import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- Page Configuration ---
st.set_page_config(page_title="BodyFat ML Pipeline", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS for Aesthetics (Horizontal Stepper) ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        color: #31333F;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title(" Interactive ML Pipeline: Body Fat Prediction")

# --- Load Built-in Data ---
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "bodyfat.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Automatically drop Density to prevent data leakage
        if "Density" in df.columns:
            df = df.drop(columns=["Density"])
        return df
    else:
        st.error("⚠️ 'bodyfat.csv' not found! Please ensure it is in the same folder as app.py.")
        return pd.DataFrame()

# --- Session State Initialization ---
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = load_data()
if 'data' not in st.session_state:
    st.session_state['data'] = st.session_state['raw_data'].copy()
if 'target' not in st.session_state:
    st.session_state['target'] = 'BodyFat'

df = st.session_state['data']

# --- Horizontal Stepper using Tabs ---
tabs = st.tabs([
    "1. Problem Setup", "2. Data & PCA", "3. EDA", "4. Cleaning & Outliers", 
    "5. Feature Selection", "6. Split & Model", "7. Train & CV", "8. Metrics", "9. Tuning"
])

# ==========================================
# STEP 1: Problem Definition
# ==========================================
with tabs[0]:
    st.header("1. Problem Definition")
    st.info("Dataset loaded automatically: **BodyFat**")
    st.success("Pipeline set for **Regression** (Predicting continuous Body Fat Percentage).")
    st.session_state['problem_type'] = 'Regression'

# ==========================================
# STEP 2: Data & PCA
# ==========================================
with tabs[1]:
    st.header("2. Data Input & Visualization")
    
    if not df.empty:
        if st.button("Reload Original Data"):
            st.session_state['data'] = st.session_state['raw_data'].copy()
            st.rerun()
            
        st.dataframe(df.head(), use_container_width=True)
        
        st.divider()
        st.subheader("PCA Data Shape Visualization")
        
        # Feature selection for PCA
        columns = df.columns.tolist()
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        features_for_pca = [col for col in numeric_cols if col != 'BodyFat']
        
        features = st.multiselect("Select features for PCA (Numeric only):", features_for_pca, default=features_for_pca)
        
        if len(features) >= 2:
            try:
                df_pca = df[features + ['BodyFat']].dropna()
                X_pca = StandardScaler().fit_transform(df_pca[features])
                
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_pca)
                
                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                pca_df['BodyFat'] = df_pca['BodyFat'].values
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='BodyFat', title="2D PCA Projection", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate PCA plot. Error: {e}")
        else:
            st.info("Select at least 2 numeric features to view PCA.")

# ==========================================
# STEP 3: Exploratory Data Analysis (EDA)
# ==========================================
with tabs[2]:
    st.header("3. Exploratory Data Analysis")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            st.write("**Missing Values:**")
            st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
        st.subheader("Correlation Heatmap")
        fig_corr = px.imshow(df.corr(), text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# STEP 4: Data Engineering & Cleaning
# ==========================================
with tabs[3]:
    st.header("4. Data Engineering & Cleaning")
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.subheader("Missing Value Imputation")
        impute_col = st.selectbox("Select column to impute:", numeric_cols)
        impute_method = st.radio("Imputation Method:", ("Mean", "Median", "Mode"), horizontal=True)
        
        if st.button("Apply Imputation"):
            if impute_method == "Mean":
                df[impute_col].fillna(df[impute_col].mean(), inplace=True)
            elif impute_method == "Median":
                df[impute_col].fillna(df[impute_col].median(), inplace=True)
            else:
                df[impute_col].fillna(df[impute_col].mode()[0], inplace=True)
            st.session_state['data'] = df
            st.success(f"Applied {impute_method} imputation to {impute_col}.")
            
        st.divider()
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Select Algorithm:", ("Isolation Forest", "IQR", "DBSCAN"))
        
        if st.button("Detect Outliers"):
            df_clean = df.copy()
            df_num = df_clean[numeric_cols].dropna() 
            
            outliers = np.zeros(len(df_num))
            try:
                if outlier_method == "Isolation Forest":
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    outliers = iso.fit_predict(df_num)
                    outliers = [1 if x == -1 else 0 for x in outliers]
                elif outlier_method == "IQR":
                    Q1 = df_num.quantile(0.25)
                    Q3 = df_num.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)
                    outliers = outlier_mask.astype(int).values
                
                num_outliers = sum(outliers)
                st.warning(f"Detected {num_outliers} outliers using {outlier_method}.")
                
                st.session_state['outlier_mask'] = outliers
                st.session_state['df_num_index'] = df_num.index
            except Exception as e:
                st.error(f"Error in outlier detection: {e}")
                
        if 'outlier_mask' in st.session_state and sum(st.session_state['outlier_mask']) > 0:
            if st.button("Drop Outliers"):
                indices_to_keep = st.session_state['df_num_index'][np.array(st.session_state['outlier_mask']) == 0]
                st.session_state['data'] = df.loc[indices_to_keep]
                st.session_state['outlier_mask'] = []
                st.success("Outliers removed successfully! Data shape updated.")

# ==========================================
# STEP 5: Feature Selection
# ==========================================
with tabs[4]:
    st.header("5. Feature Selection")
    
    if st.button("Run Mutual Information Analysis"):
        X = df.drop(columns=['BodyFat'])
        y = df['BodyFat']
        
        mi = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        
        fig = px.bar(mi_series, title="Feature Importance (Information Gain)", color=mi_series.values, color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state['sorted_features'] = mi_series.index.tolist()

    st.divider()
    st.subheader("Select Features for Training")
    
    options = st.session_state.get('sorted_features', [col for col in df.columns if col != 'BodyFat'])
    
    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = options

    selected_features = st.multiselect(
        "Choose which measurements the AI will be allowed to use:",
        options=options,
        default=st.session_state['selected_features']
    )
    
    st.session_state['selected_features'] = selected_features
    
    if len(selected_features) > 0:
        st.success(f"✅ {len(selected_features)} features selected.")
    else:
        st.warning("⚠️ Please select at least one feature.")

# ==========================================
# STEP 6: Split & Model Selection
# ==========================================
with tabs[5]:
    st.header("6. Data Split & Model Selection")
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100.0
    st.session_state['test_size'] = test_size
    
    st.subheader("Select Model")
    # Restricted to Regression models only since BodyFat is continuous
    model_choice = st.selectbox("Model:", ["Random Forest Regressor", "Linear Regression", "SVR"])
    st.session_state['model_choice'] = model_choice
    
    if model_choice == "SVR":
        st.session_state['svm_kernel'] = st.selectbox("SVM Kernel:", ["linear", "poly", "rbf", "sigmoid"])

# ==========================================
# STEP 7: Training & K-Fold Validation
# ==========================================
with tabs[6]:
    st.header("7. Model Training & Validation")
    st.number_input("K-Folds for Cross Validation:", min_value=2, max_value=10, value=5, disabled=True)
    
    if st.button("Train Model"):
        if len(st.session_state.get('selected_features', [])) == 0:
            st.error("❌ You must select at least one feature in Step 5!")
        else:
            # Use ONLY the selected features
            X = df[st.session_state['selected_features']]
            y = df['BodyFat']
            
            if st.session_state['model_choice'] == "Linear Regression": 
                model = LinearRegression()
            elif st.session_state['model_choice'] == "Random Forest Regressor": 
                model = RandomForestRegressor(random_state=42)
            elif st.session_state['model_choice'] == "SVR": 
                model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
            
            st.session_state['model'] = model
            st.session_state['X'] = X
            st.session_state['y'] = y
            
            with st.spinner(f"Training on {len(st.session_state['selected_features'])} features with 5-Fold CV..."):
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                st.success("✅ Training Complete!")
                
                st.write("**Cross Validation R2 Scores:**")
                st.code([round(num, 4) for num in scores])
                st.metric("Average CV Score (R2)", f"{scores.mean():.4f}")

# ==========================================
# STEP 8: Performance Metrics
# ==========================================
with tabs[7]:
    st.header("8. Performance Metrics")
    if 'model' in st.session_state and 'X' in st.session_state:
        if st.button("Generate Final Metrics"):
            X, y = st.session_state['X'], st.session_state['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state['test_size'], random_state=42)
            
            model = st.session_state['model']
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Training R2 Score", f"{train_r2:.4f}")
            col2.metric("Testing R2 Score", f"{test_r2:.4f}")
            col3.metric("Test MSE", f"{mse:.2f}")
            
            if train_r2 - test_r2 > 0.15:
                st.warning("⚠️ Model might be Overfitting (High Training score, lower Test score).")
            else:
                st.success("✅ Model generalized well!")
                
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
            fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted Body Fat %", trendline="ols", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the model in Step 7 first.")

# ==========================================
# STEP 9: Hyperparameter Tuning
# ==========================================
with tabs[8]:
    st.header("9. Hyperparameter Tuning")
    if 'model' in st.session_state and 'X' in st.session_state:
        tune_method = st.radio("Tuning Method:", ("GridSearch", "RandomSearch"), horizontal=True)
        
        if st.button("Start Tuning"):
            model_name = st.session_state['model_choice']
            param_grid = {}
            
            if model_name == "Random Forest Regressor":
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            elif model_name == "SVR":
                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            else:
                st.warning(f"No tuning grid defined for {model_name} (Linear Regression does not require standard tuning).")
            
            if param_grid:
                with st.spinner(f"Running {tune_method}..."):
                    model = st.session_state['model']
                    X, y = st.session_state['X'], st.session_state['y']
                    
                    if tune_method == "GridSearch":
                        search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
                    else:
                        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, random_state=42, scoring='r2')
                        
                    search.fit(X, y)
                    
                    st.success("Tuning Complete!")
                    st.write("**Best Parameters Found:**", search.best_params_)
                    st.metric("Best Tuned CV Score (R2)", f"{search.best_score_:.4f}")
    else:
         st.info("Ensure the model is set up in previous steps.")
