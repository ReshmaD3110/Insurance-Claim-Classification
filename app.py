import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to run the model
def run_model(data, model_name, target_variable, hyperparams):
    # Vectorization of the claim_description
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['claim_description']).toarray()
    
    # Preparing target variable based on user selection
    y = data[target_variable]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Model selection
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=10, **hyperparams)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=10, **hyperparams)

    model.fit(X_train_balanced, y_train_balanced)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'train_precision': precision_score(y_train, y_pred_train, average='weighted'),
        'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
        'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_test, y_pred_test, average='weighted')
    }

    return results

# Streamlit application
st.title("Model Runner")

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

# Model selection
model_name = st.selectbox("Choose the model to run", ("Random Forest", "XGBoost"))

# Target variable selection
target_variable = st.selectbox("Select target variable to predict", 
                                 ("coverage_code_encoded", "accident_source_encoded"))

# Hyperparameter inputs
if model_name == "Random Forest":
    st.write("### Random Forest Hyperparameters")
    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100, 
                                    help="The number of trees in the forest. Increasing this number can improve performance but also increases computation time.")
    max_depth = st.number_input("Max Depth", min_value=1, value=None, 
                                 help="The maximum depth of the tree. Deeper trees can model more complex relationships but may lead to overfitting.")
    hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth}
else:
    st.write("### XGBoost Hyperparameters")
    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100, 
                                    help="The number of trees in the model. More trees can lead to better accuracy but will increase training time.")
    learning_rate = st.number_input("Learning Rate", min_value=0.01, value=0.1, format="%.2f", 
                                     help="The step size at each iteration while moving toward a minimum of the loss function. Smaller values can lead to better performance but require more trees.")
    hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Run the model and display results
    if st.button("Run Model"):
        results = run_model(data, model_name, target_variable, hyperparams)
        st.success("Model run successfully!")

        # Display results
        st.write("### Results:")
        st.write(f"Train Precision: {results['train_precision']:.4f}")
        st.write(f"Train Recall: {results['train_recall']:.4f}")
        st.write(f"Test Precision: {results['test_precision']:.4f}")
        st.write(f"Test Recall: {results['test_recall']:.4f}")

        # Save the results to Excel
        if st.button("Save Results to Excel"):
            results_df = pd.DataFrame([results])
            results_df.to_excel("model_results.xlsx", index=False)
            st.success("Results saved to 'model_results.xlsx'")

# Run the app
if __name__ == "__main__":
    st.write("Upload a dataset to get started!")
