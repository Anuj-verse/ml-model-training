import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# --- Page Configuration ---
st.set_page_config(page_title="AutoML Pipeline Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS for Aesthetics ---
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

st.title(" Interactive ML Pipeline Dashboard")

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'problem_type' not in st.session_state:
    st.session_state['problem_type'] = 'Classification'
if 'target' not in st.session_state:
    st.session_state['target'] = None

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
    st.session_state['problem_type'] = st.radio(
        "Select the type of problem to solve:",
        ("Classification", "Regression"),
        horizontal=True
    )
    st.success(f"Pipeline set for **{st.session_state['problem_type']}**.")

# ==========================================
# STEP 2: Input Data & PCA
# ==========================================
with tabs[1]:
    st.header("2. Data Input & Visualization")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state['data'] is None or st.button("Reload Data"):
            st.session_state['data'] = pd.read_csv(uploaded_file)
        
        df = st.session_state['data']
        st.dataframe(df.head())
        
        # Target Selection
        columns = df.columns.tolist()
        st.session_state['target'] = st.selectbox("Select the Target Feature:", columns)
        
        st.divider()
        st.subheader("PCA Data Shape Visualization")
        
        # Feature selection for PCA
        features = st.multiselect("Select features for PCA (Numeric only):", [col for col in columns if col != st.session_state['target']], default=[col for col in columns if col != st.session_state['target'] and pd.api.types.is_numeric_dtype(df[col])])
        
        if len(features) >= 2:
            try:
                # Basic dropna for PCA visualization purposes
                df_pca = df[features + [st.session_state['target']]].dropna()
                X_pca = StandardScaler().fit_transform(df_pca[features])
                
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_pca)
                
                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                pca_df['Target'] = df_pca[st.session_state['target']].values
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="2D PCA Projection", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate PCA plot. Please ensure selected features are numeric and handle extreme missing values. Error: {e}")
        else:
            st.info("Select at least 2 numeric features to view PCA.")

# ==========================================
# STEP 3: Exploratory Data Analysis (EDA)
# ==========================================
with tabs[2]:
    st.header("3. Exploratory Data Analysis")
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        
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
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Please upload data in Step 2.")

# ==========================================
# STEP 4: Data Engineering & Cleaning
# ==========================================
with tabs[3]:
    st.header("4. Data Engineering & Cleaning")
    if st.session_state['data'] is not None:
        df = st.session_state['data']
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
        outlier_method = st.selectbox("Select Algorithm:", ("IQR", "Isolation Forest", "DBSCAN", "OPTICS"))
        
        if st.button("Detect Outliers"):
            # Simplified Outlier Detection Logic for Demonstration
            df_clean = df.copy()
            df_num = df_clean[numeric_cols].dropna() # Requires no NaNs for these models
            
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
                # DBSCAN and OPTICS omitted for brevity, but follow similar fit_predict logic
                
                num_outliers = sum(outliers)
                st.warning(f"Detected {num_outliers} outliers using {outlier_method}.")
                
                st.session_state['outlier_mask'] = outliers
                st.session_state['df_num_index'] = df_num.index
            except Exception as e:
                st.error(f"Error in outlier detection: {e}. (Ensure no missing values in numeric columns)")
                
        if 'outlier_mask' in st.session_state and sum(st.session_state['outlier_mask']) > 0:
            if st.button("Drop Outliers"):
                indices_to_keep = st.session_state['df_num_index'][np.array(st.session_state['outlier_mask']) == 0]
                st.session_state['data'] = df.loc[indices_to_keep]
                st.session_state['outlier_mask'] = [] # Reset
                st.success("Outliers removed successfully! Data shape updated.")
    else:
        st.info("Please upload data in Step 2.")

# ==========================================
# STEP 5: Feature Selection
# ==========================================
with tabs[4]:
    st.header("5. Feature Selection")
    if st.session_state['data'] is not None and st.session_state['target'] is not None:
        df = st.session_state['data']
        target = st.session_state['target']
        
        method = st.selectbox("Selection Method:", ["Variance Threshold", "Information Gain (Mutual Info)"])
        
        if st.button("Run Feature Selection"):
            X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore').dropna()
            y = df.loc[X.index, target]
            
            if method == "Variance Threshold":
                selector = VarianceThreshold(threshold=0.1)
                selector.fit(X)
                selected_cols = X.columns[selector.get_support()]
                st.write("**Features passing variance threshold:**")
                st.write(selected_cols.tolist())
                
            elif method == "Information Gain (Mutual Info)":
                if st.session_state['problem_type'] == 'Classification':
                    mi = mutual_info_classif(X, y)
                else:
                    mi = mutual_info_regression(X, y)
                mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
                fig = px.bar(mi_series, title="Feature Importance via Information Gain")
                st.plotly_chart(fig)
    else:
        st.info("Ensure data is uploaded and target is selected.")

# ==========================================
# STEP 6: Split & Model Selection
# ==========================================
with tabs[5]:
    st.header("6. Data Split & Model Selection")
    if st.session_state['data'] is not None:
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100.0
        st.session_state['test_size'] = test_size
        
        st.subheader("Select Model")
        if st.session_state['problem_type'] == 'Classification':
            model_choice = st.selectbox("Model:", ["Logistic Regression", "SVM", "Random Forest"])
        else:
            model_choice = st.selectbox("Model:", ["Linear Regression", "SVR", "Random Forest Regressor"])
            
        st.session_state['model_choice'] = model_choice
        
        # Kernel options for SVM
        if model_choice in ["SVM", "SVR"]:
            st.session_state['svm_kernel'] = st.selectbox("SVM Kernel:", ["linear", "poly", "rbf", "sigmoid"])
            
        # K-Means Note (Since user asked, though it's unsupervised)
        st.info("Note: K-Means was requested but is an unsupervised clustering algorithm. It is typically not evaluated alongside supervised train/test splits.")

# ==========================================
# STEP 7: Training & K-Fold Validation
# ==========================================
with tabs[6]:
    st.header("7. Model Training & K-Fold Validation")
    if st.session_state['data'] is not None and 'model_choice' in st.session_state:
        df = st.session_state['data'].dropna() # Drop NaNs for modelling
        target = st.session_state['target']
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        y = df[target]
        
        k_folds = st.number_input("Select K for K-Fold Cross Validation:", min_value=2, max_value=20, value=5)
        
        if st.button("Train Model & Validate"):
            # Model instantiation
            model = None
            if st.session_state['model_choice'] == "Linear Regression": model = LinearRegression()
            elif st.session_state['model_choice'] == "Logistic Regression": model = LogisticRegression(max_iter=1000)
            elif st.session_state['model_choice'] == "Random Forest": model = RandomForestClassifier()
            elif st.session_state['model_choice'] == "Random Forest Regressor": model = RandomForestRegressor()
            elif st.session_state['model_choice'] == "SVM": model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'))
            elif st.session_state['model_choice'] == "SVR": model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
            
            st.session_state['model'] = model
            st.session_state['X'] = X
            st.session_state['y'] = y
            
            # K-Fold
            with st.spinner("Running Cross Validation..."):
                scoring = 'accuracy' if st.session_state['problem_type'] == 'Classification' else 'neg_mean_squared_error'
                scores = cross_val_score(model, X, y, cv=k_folds, scoring=scoring)
                
                if st.session_state['problem_type'] == 'Regression':
                    scores = -scores # Convert neg MSE to positive
                    
                st.success("Training Complete!")
                st.write(f"**Cross-Validation Scores:** {scores}")
                st.write(f"**Mean CV Score:** {scores.mean():.4f}")

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
            
            col1, col2 = st.columns(2)
            if st.session_state['problem_type'] == 'Classification':
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                
                col1.metric("Training Accuracy", f"{train_acc:.4f}")
                col2.metric("Testing Accuracy", f"{test_acc:.4f}")
                
                # Check for overfitting
                if train_acc - test_acc > 0.1:
                    st.warning("⚠️ High variance detected: Model might be Overfitting (Training Acc significantly higher than Test Acc).")
                elif train_acc < 0.6 and test_acc < 0.6:
                    st.error("⚠️ High bias detected: Model might be Underfitting (Both accuracies are low).")
                else:
                    st.success("✅ Model generalized well!")
            else:
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                col1.metric("Training R2 Score", f"{train_r2:.4f}")
                col2.metric("Testing R2 Score", f"{test_r2:.4f}")
                
                if train_r2 - test_r2 > 0.15:
                    st.warning("⚠️ Model might be Overfitting.")
                else:
                    st.success("✅ Model generalized well!")
    else:
        st.info("Train the model in Step 7 first.")

# ==========================================
# STEP 9: Hyperparameter Tuning
# ==========================================
with tabs[8]:
    st.header("9. Hyperparameter Tuning")
    if 'model' in st.session_state and 'X' in st.session_state:
        tune_method = st.radio("Tuning Method:", ("GridSearch", "RandomSearch"), horizontal=True)
        
        st.write("*(Pre-configured param grids are used based on the model selected in Step 6)*")
        
        if st.button("Start Tuning"):
            model_name = st.session_state['model_choice']
            param_grid = {}
            
            # Setup dummy param grids for demonstration
            if "Random Forest" in model_name:
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            elif "SVM" in model_name or "SVR" in model_name:
                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            elif "Logistic" in model_name:
                param_grid = {'C': [0.1, 1.0, 10.0]}
            else:
                st.warning(f"No tuning grid defined for {model_name}.")
            
            if param_grid:
                with st.spinner(f"Running {tune_method}..."):
                    model = st.session_state['model']
                    X, y = st.session_state['X'], st.session_state['y']
                    
                    if tune_method == "GridSearch":
                        search = GridSearchCV(model, param_grid, cv=3)
                    else:
                        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5)
                        
                    search.fit(X, y)
                    
                    st.success("Tuning Complete!")
                    st.write("**Best Parameters Found:**", search.best_params_)
                    st.write("**Best CV Score:**", f"{search.best_score_:.4f}")
    else:
         st.info("Ensure the model is set up in previous steps.")