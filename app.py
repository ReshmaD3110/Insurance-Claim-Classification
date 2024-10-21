import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Function to run the model
def run_model(data, model_name):
    X = data.drop(columns=['coverage_code_encoded'])
    y = data['coverage_code_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=10)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=10)

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

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Run the model and display results
    if st.button("Run Model"):
        results = run_model(data, model_name)
        st.success("Model run successfully!")

        # Display results
        st.write("### Results:")
        st.write(f"Train Precision: {results['train_precision']:.4f}")
        st.write(f"Train Recall: {results['train_recall']:.4f}")
        st.write(f"Test Precision: {results['test_precision']:.4f}")
        st.write(f"Test Recall: {results['test_recall']:.4f}")

        # Optionally, save the results to Excel
        if st.button("Save Results to Excel"):
            results_df = pd.DataFrame([results])
            results_df.to_excel("model_results.xlsx", index=False)
            st.success("Results saved to 'model_results.xlsx'")

# Run the app
if __name__ == "__main__":
    st.write("Upload a dataset to get started!")


