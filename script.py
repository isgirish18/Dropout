import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib  # Import joblib to load the fitted scaler

# Load the trained model
model = load_model('student_classification_model.h5')

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

# Load the label encoder (replace with your actual label encoder classes)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Dropout', 'Graduate', 'Enrolled'])  # Modify with actual classes

# Streamlit form for getting user input
def get_input():
    st.title("Student Classification Model")

    # Create form for input
    with st.form("input_form"):
        marital_status = st.radio(
            "Marital Status", 
            options=[1, 2, 3, 4, 5, 6], 
            format_func=lambda x: {
                1: "Single", 
                2: "Married", 
                3: "Widower", 
                4: "Divorced", 
                5: "Common-law marriage", 
                6: "Legally separated"
            }[x]
        )
        application_mode = st.radio(
            "Application Mode", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
            format_func=lambda x: {
                1: "1st phase—general contingent", 2: "Ordinance No. 612/93", 3: "1st phase—special contingent (Azores Island)", 
                4: "Holders of other higher courses", 5: "Ordinance No. 854-B/99", 6: "International student (bachelor)", 
                7: "1st phase—special contingent (Madeira Island)", 8: "2nd phase—general contingent", 
                9: "3rd phase—general contingent", 10: "Ordinance No. 533-A/99, item b2) (Different Plan)", 
                11: "Ordinance No. 533-A/99, item b3 (Other Institution)", 12: "Over 23 years old", 
                13: "Transfer", 14: "Change in course", 15: "Technological specialization diploma holders", 
                16: "Change in institution/course", 17: "Short cycle diploma holders", 
                18: "Change in institution/course (International)"
            }[x]
        )
        application_order = st.radio("Application Order", options=[1, 2, 3, 4, 5, 6, 7], help="1-7: Application order")
        course = st.radio(
            "Course", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 
            format_func=lambda x: {
                1: "Biofuel Production Technologies", 2: "Animation and Multimedia Design", 3: "Social Service (evening attendance)", 
                4: "Agronomy", 5: "Communication Design", 6: "Veterinary Nursing", 7: "Informatics Engineering", 
                8: "Equiniculture", 9: "Management", 10: "Social Service", 11: "Tourism", 12: "Nursing", 
                13: "Oral Hygiene", 14: "Advertising and Marketing Management", 15: "Journalism and Communication", 
                16: "Basic Education", 17: "Management (evening attendance)"
            }[x]
        )
        daytime_attendance = st.radio(
            "Daytime/Evening Attendance", 
            options=[0, 1], 
            format_func=lambda x: "Evening" if x == 0 else "Daytime"
        )
        previous_qualification = st.radio(
            "Previous Qualification", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 
            format_func=lambda x: {
                1: "Secondary education", 2: "Higher education—bachelor’s degree", 3: "Higher education—degree", 
                4: "Higher education—master’s degree", 5: "Higher education—doctorate", 6: "Frequency of higher education", 
                7: "12th year of schooling—not completed", 8: "11th year of schooling—not completed", 
                9: "Other—11th year of schooling", 10: "10th year of schooling", 11: "10th year of schooling—not completed", 
                12: "Basic education 3rd cycle (9th/10th/11th year) or equivalent", 13: "Basic education 2nd cycle (6th/7th/8th year) or equivalent", 
                14: "Technological specialization course", 15: "Higher education—degree (1st cycle)", 
                16: "Professional higher technical course", 17: "Higher education—master’s degree (2nd cycle)"
            }[x]
        )
        nationality = st.radio(
            "Nationality", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 
            format_func=lambda x: {
                1: "Portuguese", 2: "German", 3: "Spanish", 4: "Italian", 5: "Dutch", 
                6: "English", 7: "Lithuanian", 8: "Angolan", 9: "Cape Verdean", 10: "Guinean", 
                11: "Mozambican", 12: "Santomean", 13: "Turkish", 14: "Brazilian", 15: "Romanian", 
                16: "Moldova (Republic of)", 17: "Mexican", 18: "Ukrainian", 19: "Russian", 
                20: "Cuban", 21: "Colombian"
            }[x]
        )
        mothers_qualification = st.radio(
            "Mother's Qualification", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
            format_func=lambda x: {
                1: "Secondary Education—12th Year of Schooling or Equivalent", 2: "Higher Education—bachelor’s degree", 
                3: "Higher Education—degree", 4: "Higher Education—master’s degree", 5: "Higher Education—doctorate", 
                6: "Frequency of Higher Education", 7: "12th Year of Schooling—not completed", 8: "11th Year of Schooling—not completed", 
                9: "7th Year (Old)", 10: "Other—11th Year of Schooling", 11: "2nd year complementary high school course", 
                12: "10th Year of Schooling", 13: "General commerce course", 14: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent", 
                15: "Complementary High School Course", 16: "Technical-professional course", 17: "Complementary High School Course—not concluded", 
                18: "7th year of schooling", 19: "2nd cycle of the general high school course", 20: "9th Year of Schooling—not completed", 
                21: "8th year of schooling", 22: "General Course of Administration and Commerce", 23: "Supplementary Accounting and Administration", 
                24: "Unknown", 25: "Cannot read or write"
            }[x]
        )
        fathers_qualification = st.radio(
            "Father's Qualification", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
            format_func=lambda x: {
                1: "Secondary Education—12th Year of Schooling or Equivalent", 2: "Higher Education—bachelor’s degree", 
                3: "Higher Education—degree", 4: "Higher Education—master’s degree", 5: "Higher Education—doctorate", 
                6: "Frequency of Higher Education", 7: "12th Year of Schooling—not completed", 8: "11th Year of Schooling—not completed", 
                9: "7th Year (Old)", 10: "Other—11th Year of Schooling", 11: "2nd year complementary high school course", 
                12: "10th Year of Schooling", 13: "General commerce course", 14: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent", 
                15: "Complementary High School Course", 16: "Technical-professional course", 17: "Complementary High School Course—not concluded", 
                18: "7th year of schooling", 19: "2nd cycle of the general high school course", 20: "9th Year of Schooling—not completed", 
                21: "8th year of schooling", 22: "General Course of Administration and Commerce", 23: "Supplementary Accounting and Administration", 
                24: "Unknown", 25: "Cannot read or write"
            }[x]
        )
        mothers_occupation = st.radio(
            "Mother's Occupation", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], 
            format_func=lambda x: {
                1: "Student", 2: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
                3: "Specialists in Intellectual and Scientific Activities", 4: "Intermediate Level Technicians and Professions", 5: "Administrative staff", 
                6: "Personal Services, Security and Safety Workers, and Sellers", 7: "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry", 
                8: "Skilled Workers in Industry, Construction, and Craftsmen", 9: "Installation and Machine Operators and Assembly Workers", 
                10: "Unskilled Workers", 11: "Armed Forces Professions", 12: "Other Situation", 13: "Armed Forces Officers", 
                14: "Armed Forces Sergeants", 15: "Other Armed Forces personnel", 16: "Directors of administrative and commercial services", 
                17: "Hotel, catering, trade, and other services directors", 18: "Specialists in the physical sciences, mathematics, engineering, and related techniques", 
                19: "Health professionals", 20: "Teachers", 21: "Specialists in finance, accounting, administrative organization, and public and commercial relations", 
                22: "Intermediate level science and engineering technicians and professions", 23: "Technicians and professionals of intermediate level of health", 
                24: "Intermediate level technicians from legal, social, sports, cultural, and similar services", 25: "Information and communication technology technicians", 
                26: "Office workers, secretaries in general, and data processing operators", 27: "Data, accounting, statistical, financial services, and registry-related operators", 
                28: "Other administrative support staff", 29: "Personal service workers", 30: "Sellers", 31: "Personal care workers and the like", 
                32: "Protection and security services personnel", 33: "Market-oriented farmers and skilled agricultural and animal production workers", 
                34: "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence", 35: "Skilled construction workers and the like, except electricians", 
                36: "Skilled workers in metallurgy, metalworking, and similar", 37: "Skilled workers in electricity and electronics", 38: "Workers in food processing, woodworking, and clothing and other industries and crafts", 
                39: "Fixed plant and machine operators", 40: "Assembly workers", 41: "Vehicle drivers and mobile equipment operators", 
                42: "Unskilled workers in agriculture, animal production, and fisheries and forestry", 43: "Unskilled workers in extractive industry, construction, manufacturing, and transport", 
                44: "Meal preparation assistants", 45: "Street vendors (except food) and street service provider"
            }[x]
        )
        fathers_occupation = st.radio(
            "Father's Occupation", 
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], 
            format_func=lambda x: {
                1: "Student", 2: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
                3: "Specialists in Intellectual and Scientific Activities", 4: "Intermediate Level Technicians and Professions", 5: "Administrative staff", 
                6: "Personal Services, Security and Safety Workers, and Sellers", 7: "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry", 
                8: "Skilled Workers in Industry, Construction, and Craftsmen", 9: "Installation and Machine Operators and Assembly Workers", 
                10: "Unskilled Workers", 11: "Armed Forces Professions", 12: "Other Situation", 13: "Armed Forces Officers", 
                14: "Armed Forces Sergeants", 15: "Other Armed Forces personnel", 16: "Directors of administrative and commercial services", 
                17: "Hotel, catering, trade, and other services directors", 18: "Specialists in the physical sciences, mathematics, engineering, and related techniques", 
                19: "Health professionals", 20: "Teachers", 21: "Specialists in finance, accounting, administrative organization, and public and commercial relations", 
                22: "Intermediate level science and engineering technicians and professions", 23: "Technicians and professionals of intermediate level of health", 
                24: "Intermediate level technicians from legal, social, sports, cultural, and similar services", 25: "Information and communication technology technicians", 
                26: "Office workers, secretaries in general, and data processing operators", 27: "Data, accounting, statistical, financial services, and registry-related operators", 
                28: "Other administrative support staff", 29: "Personal service workers", 30: "Sellers", 31: "Personal care workers and the like", 
                32: "Protection and security services personnel", 33: "Market-oriented farmers and skilled agricultural and animal production workers", 
                34: "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence", 35: "Skilled construction workers and the like, except electricians", 
                36: "Skilled workers in metallurgy, metalworking, and similar", 37: "Skilled workers in electricity and electronics", 38: "Workers in food processing, woodworking, and clothing and other industries and crafts", 
                39: "Fixed plant and machine operators", 40: "Assembly workers", 41: "Vehicle drivers and mobile equipment operators", 
                42: "Unskilled workers in agriculture, animal production, and fisheries and forestry", 43: "Unskilled workers in extractive industry, construction, manufacturing, and transport", 
                44: "Meal preparation assistants", 45: "Street vendors (except food) and street service provider"
            }[x]
        )
        displaced = st.radio(
            "Displaced", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        educational_special_needs = st.radio(
            "Education Special Needs", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        debtor = st.radio(
            "Debtor", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        tuition_fees_up_to_date = st.radio(
            "Tuition Fees Up to Date", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        gender = st.radio(
            "Gender", 
            options=[0, 1], 
            format_func=lambda x: "Female" if x == 0 else "Male"
        )
        scholarship_holder = st.radio(
            "Scholarship Holder", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        age_at_enrollment = st.number_input("Age at Enrollment", value=20, min_value=18, max_value=100)
        international = st.radio("International", options=[0, 1], help="0: No, 1: Yes")
        curricular_units_1st_sem_credited = st.number_input("Curricular Units 1st Sem (Credited)", value=0)
        curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", value=0)
        curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem (Evaluations)", value=0)
        curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (Approved)", value=0)
        curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem (Grade)", value=0)
        curricular_units_1st_sem_without_evaluations = st.number_input("Curricular Units 1st Sem (Without Evaluations)", value=0)
        curricular_units_2nd_sem_credited = st.number_input("Curricular Units 2nd Sem (Credited)", value=0)
        curricular_units_2nd_sem_enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", value=0)
        curricular_units_2nd_sem_evaluations = st.number_input("Curricular Units 2nd Sem (Evaluations)", value=0)
        curricular_units_2nd_sem_approved = st.number_input("Curricular Units 2nd Sem (Approved)", value=0)
        curricular_units_2nd_sem_grade = st.number_input("Curricular Units 2nd Sem (Grade)", value=0)
        curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular Units 2nd Sem (Without Evaluations)", value=0)
        unemployment_rate = st.number_input("Unemployment Rate", value=0.0)
        inflation_rate = st.number_input("Inflation Rate", value=0.0)
        gdp = st.number_input("GDP", value=0.0)

        # Submit button
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Prepare input data (make sure all features are included)
        input_data = np.array([marital_status, application_mode, application_order, course, daytime_attendance,
                               previous_qualification, nationality, mothers_qualification, fathers_qualification,
                               mothers_occupation, fathers_occupation, displaced, educational_special_needs, debtor,
                               tuition_fees_up_to_date, gender, scholarship_holder, age_at_enrollment, international,
                               curricular_units_1st_sem_credited, curricular_units_1st_sem_enrolled,
                               curricular_units_1st_sem_evaluations, curricular_units_1st_sem_approved,
                               curricular_units_1st_sem_grade, curricular_units_1st_sem_without_evaluations,
                               curricular_units_2nd_sem_credited, curricular_units_2nd_sem_enrolled,
                               curricular_units_2nd_sem_evaluations, curricular_units_2nd_sem_approved,
                               curricular_units_2nd_sem_grade, curricular_units_2nd_sem_without_evaluations,
                               unemployment_rate, inflation_rate, gdp])

        # Scale the input data using the same fitted scaler
        input_data_scaled = scaler.transform([input_data])

        # Make prediction
        y_pred_prob = model.predict(input_data_scaled)
        predicted_class = np.argmax(y_pred_prob, axis=1)  # Get the index of the highest probability

        # Decode predicted class to string
        predicted_class_label = label_encoder.inverse_transform(predicted_class)

        # Show the result
        st.write(f"Predicted Class: {predicted_class_label[0]}")
        # st.write(f"Prediction Probabilities: {y_pred_prob}")

if __name__ == "__main__":
    get_input()
