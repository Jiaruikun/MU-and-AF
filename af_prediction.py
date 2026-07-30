# af_prediction_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Atrial Fibrillation Recurrence Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Application title and description
st.title("🏥 Atrial Fibrillation Recurrence Prediction System")
st.markdown("""
This system predicts the risk of early recurrence after catheter ablation in atrial fibrillation patients 
based on clinical data. The system first calculates metabolic unhealthy status (MU), then performs recurrence risk prediction.
""")

# Define metabolic unhealthy calculation function
def calculate_metabolically_unhealthy(row):
    """
    Calculate metabolic unhealthy status
    """
    unhealthy_conditions = 0
    
    # Impaired glucose control: FBG ≥5.6 or using anti-diabetic drugs
    if row['FBG(mmol/L)'] >= 5.6 or row.get('DM', 0) == 1:
        unhealthy_conditions += 1
    
    # High blood pressure: SBP ≥130 or DBP ≥85 or using anti-hypertensive drugs
    if row['SBP(mmHg)'] >= 130 or row['DBP(mmHg)'] >= 85 or row.get('HT', 0) == 1:
        unhealthy_conditions += 1
    
    # Elevated TG: TG ≥1.7
    if row['TG(mmol/L)'] >= 1.7:
        unhealthy_conditions += 1
    
    # Low HDL-C: Male HDL-C <1.03 or Female <1.29 or using lipid-lowering drugs
    if (row['Female'] == 0 and row['HDL-c(mmol/L)'] < 1.03) or \
       (row['Female'] == 1 and row['HDL-c(mmol/L)'] < 1.29) or \
       row.get('Hyperlipidemia', 0) == 1:
        unhealthy_conditions += 1
    
    # Determine metabolic unhealthy (at least 2 conditions)
    return unhealthy_conditions >= 2

# Load model function with absolute path
@st.cache_resource
def load_prediction_model():
    """
    Load prediction model
    """
    try:
        model_path = r"G:\APP\MU\tabpfn_model.joblib"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        model_data = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        
        # Debug: show model features
        st.sidebar.header("🔧 Model Features")
        st.sidebar.write(f"Feature columns: {model_data['feature_columns']}")
        
        return model_data
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Create input form
st.header("📋 Patient Clinical Data Input")

# Use two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=60)
    female = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    
    st.subheader("Medical History")
    duration_af = st.number_input("AF Duration (months)", min_value=0.0, max_value=240.0, value=12.0, step=1.0)
    early_recurrence = st.selectbox("Early Recurrence History", options=[0, 1], 
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="History of early recurrence after previous ablation")
    ht = st.selectbox("Hypertension History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dm = st.selectbox("Diabetes History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    hyperlipidemia = st.selectbox("Hyperlipidemia History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    st.subheader("Metabolic Parameters")
    fbg = st.number_input("Fasting Blood Glucose (FBG, mmol/L)", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
    tg = st.number_input("Triglycerides (TG, mmol/L)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

with col2:
    st.subheader("Blood Pressure Parameters")
    sbp = st.number_input("Systolic Blood Pressure (SBP, mmHg)", min_value=80, max_value=200, value=120)
    dbp = st.number_input("Diastolic Blood Pressure (DBP, mmHg)", min_value=50, max_value=130, value=80)
    
    st.subheader("Echocardiography Parameters")
    la = st.number_input("Left Atrial Diameter (LA, mm)", min_value=20, max_value=60, value=40)
    ra = st.number_input("Right Atrial Diameter (RA, mm)", min_value=20, max_value=60, value=38)
    emv = st.number_input("Mitral Valve E-wave Velocity (Emv, m/s)", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    laa_velocity = st.number_input("Left Atrial Appendage Emptying Velocity (m/s)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
    hdl = st.number_input("HDL Cholesterol (HDL-c, mmol/L)", min_value=0.1, max_value=3.0, value=1.2, step=0.1)

# Calculate button
if st.button("Calculate Metabolic Status and Predict Recurrence Risk", type="primary"):
    
    # Create data row
    patient_data = {
        'Age (years)': age,
        'Female': female,
        'LA(mm)': la,
        'RA(mm)': ra,
        'Duration of AF(month)': duration_af,
        'Emv(m/s)': emv,
        'Left Atrial Appendage Emptying Velocity(m/s)': laa_velocity,
        'FBG(mmol/L)': fbg,
        'TG(mmol/L)': tg,
        'HDL-c(mmol/L)': hdl,
        'SBP(mmHg)': sbp,
        'DBP(mmHg)': dbp,
        'HT': ht,
        'DM': dm,
        'Hyperlipidemia': hyperlipidemia,
        'Early Recurrence': early_recurrence  # 添加早期复发历史
    }
    
    # Calculate metabolic unhealthy status
    st.header("🔍 Metabolic Unhealthy Status Analysis")
    
    mu_result = calculate_metabolically_unhealthy(patient_data)
    patient_data['Metabolically_unhealthy'] = mu_result
    patient_data['MU'] = 1 if mu_result else 0
    
    # Display MU results
    if mu_result:
        st.error("**Metabolic Unhealthy Status: Yes** 🚨")
        st.warning("This patient meets metabolic unhealthy criteria. Metabolic syndrome intervention is recommended.")
    else:
        st.success("**Metabolic Unhealthy Status: No** ✅")
        st.info("This patient has relatively healthy metabolic status")
    
    # Perform prediction
    st.header("📊 AF Recurrence Risk Prediction")
    
    # Load model
    model_data = load_prediction_model()
    
    if model_data is not None:
        try:
            # Prepare prediction data - include all features including 'Early Recurrence'
            feature_columns = model_data['feature_columns']
            
            st.sidebar.header("🔧 Prediction Features")
            st.sidebar.write(f"Required features: {feature_columns}")
            st.sidebar.write(f"Available patient data: {list(patient_data.keys())}")
            
            # Create prediction data with all required features
            prediction_dict = {}
            missing_features = []
            
            for feature in feature_columns:
                if feature in patient_data:
                    prediction_dict[feature] = patient_data[feature]
                else:
                    # Handle missing features
                    missing_features.append(feature)
                    st.warning(f"Feature '{feature}' not found in input data. Using default value 0.")
                    prediction_dict[feature] = 0
            
            if missing_features:
                st.error(f"Missing features in input data: {missing_features}")
            
            prediction_data = pd.DataFrame([prediction_dict])
            
            # Ensure feature consistency
            prediction_data = prediction_data[feature_columns]
            
            # Perform prediction
            model = model_data['model']
            prediction = model.predict(prediction_data)[0]
            probability = model.predict_proba(prediction_data)[0]
            
            # Display prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Results")
                if prediction == 1:
                    st.error(f"**High Risk: Likely to Recur** (Probability: {probability[1]:.1%})")
                else:
                    st.success(f"**Low Risk: Unlikely to Recur** (Probability: {probability[0]:.1%})")
                
                # Display probability distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                classes = ['No Recurrence', 'Recurrence']
                colors = ['#2ecc71', '#e74c3c']
                bars = ax.bar(classes, probability, color=colors, alpha=0.8)
                ax.set_ylabel('Probability')
                ax.set_title('Recurrence Risk Probability Distribution')
                ax.set_ylim(0, 1)
                
                # Display values on bars
                for bar, prob in zip(bars, probability):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                st.subheader("Clinical Recommendations")
                if prediction == 1:
                    st.markdown("""
                    **High-Risk Patient Management Recommendations:**
                    - 🔴 **Close Follow-up**: Regular ECG monitoring at 1, 3, 6 months post-procedure
                    - 🔴 **Medication Optimization**: Consider intensified anti-arrhythmic therapy
                    - 🔴 **Risk Factor Control**: Strict control of blood pressure, glucose, and lipids
                    - 🔴 **Lifestyle Intervention**: Weight management, alcohol restriction, regular exercise
                    - 🔴 **Early Intervention Consideration**: Consider re-ablation if recurrence occurs
                    """)
                else:
                    st.markdown("""
                    **Low-Risk Patient Management Recommendations:**
                    - 🟢 **Routine Follow-up**: Regular check-ups at 6, 12 months post-procedure
                    - 🟢 **Maintenance Therapy**: Continue current medication as prescribed
                    - 🟢 **Preventive Measures**: Continue cardiovascular risk factor control
                    - 🟢 **Healthy Lifestyle**: Maintain healthy diet and appropriate exercise
                    - 🟢 **Symptom Monitoring**: Seek medical attention if palpitations occur
                    """)
            
            # Display feature importance or input summary
            st.subheader("📋 Input Summary")
            input_summary = pd.DataFrame({
                'Feature': feature_columns,
                'Value': [prediction_dict[feature] for feature in feature_columns]
            })
            st.dataframe(input_summary, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error("Please check the model features and input data compatibility.")
    else:
        st.error("Unable to load prediction model. Please check if the model file exists.")

# Sidebar information
st.sidebar.header("ℹ️ System Information")
st.sidebar.markdown("""
**Model Feature Description:**
- AF Duration: Months of AF history
- Early Recurrence: History of early recurrence after ablation
- Left Atrial Diameter: Echocardiography measurement  
- Right Atrial Diameter: Echocardiography measurement
- Metabolic Unhealthy: Based on 4 metabolic criteria
- Mitral E-wave Velocity: Diastolic function indicator
- LAA Emptying Velocity: Thrombosis risk indicator
- Age, Gender, Hypertension, Diabetes status

**Metabolic Unhealthy Criteria:**
Meet at least 2 of the following 4 criteria:
1. Fasting glucose ≥5.6 mmol/L or diabetes
2. Blood pressure ≥130/85 mmHg or hypertension  
3. Triglycerides ≥1.7 mmol/L
4. Low HDL-C (male<1.03, female<1.29)
""")

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: The prediction results from this system are for reference only and cannot replace professional medical diagnosis. Clinical decisions should be made based on comprehensive patient evaluation.")