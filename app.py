import streamlit as st
import pandas as pd
import joblib
try:
    model = joblib.load("best_salary_model.pkl")
except FileNotFoundError:
    st.error("The model file 'best_salary_model.pkl' was not found. Please train the model first.")
    st.stop()
st.set_page_config(
    page_title="Data Science Salary Prediction",
    page_icon="💰",
    layout="centered"
)
st.title("💰 Data Science Salary Prediction App")
st.markdown("Predict the salary for a data science role based on its key features.")
st.sidebar.header("Input Job Details")
exp_level_options = ['EN', 'MI', 'SE', 'EX']
emp_type_options = ['FT', 'PT', 'CT', 'FL']
comp_size_options = ['S', 'M', 'L']
remote_ratio_options = [0, 50, 100]
experience_level = st.sidebar.selectbox("Experience Level", exp_level_options)
employment_type = st.sidebar.selectbox("Employment Type", emp_type_options)
company_size = st.sidebar.selectbox("Company Size", comp_size_options)
remote_ratio = st.sidebar.selectbox("Remote Ratio (%)", remote_ratio_options)
expected_columns = [
    'remote_ratio', 'experience_level_EX', 'experience_level_MI',
    'experience_level_SE', 'employment_type_FL', 'employment_type_FT',
    'employment_type_PT', 'company_size_M', 'company_size_S'
]
input_data = {
    'experience_level': [experience_level],
    'employment_type': [employment_type],
    'company_size': [company_size],
    'remote_ratio': [remote_ratio]
}
input_df = pd.DataFrame(input_data)

st.write("### 🔎 Input Data")
st.write(input_df)
if st.button("Predict Salary"):
    processed_input = pd.get_dummies(input_df)
    processed_input = processed_input.reindex(columns=expected_columns, fill_value=0)
    prediction = model.predict(processed_input)
    st.success(f"✅ Predicted Salary: ${prediction[0]:,.2f} USD")
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())


    required_cols = ['experience_level', 'employment_type', 'company_size', 'remote_ratio']
    if all(col in batch_data.columns for col in required_cols):
      a
        processed_batch = pd.get_dummies(batch_data[required_cols])
        processed_batch = processed_batch.reindex(columns=expected_columns, fill_value=0)


        batch_preds = model.predict(processed_batch)
        batch_data['PredictedSalaryUSD'] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())


        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name='predicted_salaries.csv',
            mime='text/csv'
        )
    else:
        st.error(f"The uploaded CSV must contain the following columns: {required_cols}")
