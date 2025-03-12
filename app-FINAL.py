import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import requests
import json
import re
import os
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="FINORA - Financial Risk & Insurance Assessment",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("FINORA")
st.subheader("Financial Risk Assessment & Insurance Claims Processing")
st.markdown("""
This application provides:
- Loan approval risk assessment
- Interest rate calculation
- Insurance claim processing
- Risk factor analysis
""")

# Sidebar
st.sidebar.title("FINORA")
st.sidebar.image("Finora_Logo[1].png")
page = st.sidebar.selectbox("Choose a Service", ["Home", "Loan Risk Assessment", "Insurance Claim Processing", "Model Training"])

# Check for model
model_file = 'loan_risk_rf_model.pkl'
model_exists = os.path.exists(model_file)

# Functions for model training
def train_model():
    st.subheader("Model Training")
    with st.expander("Model Training Information", expanded=True):
        st.info("""
        This section allows you to train the Random Forest Classifier model for loan risk assessment.
        You need to upload a credit risk dataset in CSV format with the following columns:
        - person_age: Age of the person
        - person_income: Annual income
        - person_home_ownership: Housing status (RENT, OWN, MORTGAGE, OTHER)
        - person_emp_length: Employment length in years
        - loan_intent: Purpose of loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOME_IMPROVEMENT, DEBT_CONSOLIDATION)
        - loan_grade: Loan grade (A-G)
        - loan_amnt: Loan amount requested
        - loan_int_rate: Interest rate (will be dropped during training)
        - loan_status: Target variable (0 for non-default, 1 for default)
        - loan_percent_income: Loan amount as percentage of income
        - cb_person_default_on_file: Previous defaults (Y/N)
        - cb_person_cred_hist_length: Credit history length in years
        """)
    
    uploaded_file = st.file_uploader("Upload credit risk dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset head:", df.head())
            
            # Display dataset info
            st.write("Dataset info:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Display missing values
            st.write("Missing values:")
            st.write(df.isnull().sum())
            
            with st.spinner("Training the model..."):
                # Identify categorical columns
                categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
                numerical_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                                    'loan_percent_income', 'cb_person_cred_hist_length']
                
                # Encode categorical features
                label_encoders = {}
                for column in categorical_columns:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    label_encoders[column] = le
                
                # Handle missing values
                df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
                
                # Drop loan_int_rate since we'll calculate it ourselves
                df = df.drop('loan_int_rate', axis=1)
                
                # Split data into features and target
                X = df.drop('loan_status', axis=1)
                y = df['loan_status']
                
                # Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the model
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display metrics
                st.subheader("Model Performance")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Classification report
                st.write("Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Confusion matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Non-Default', 'Default'],
                            yticklabels=['Non-Default', 'Default'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                
                # Save the model
                joblib.dump(rf_model, 'loan_risk_rf_model.pkl')
                st.success("Model trained and saved successfully!")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title('Feature Importance')
                st.pyplot(fig)
                
                return rf_model
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            return None
    else:
        st.info("Please upload a dataset to train the model.")
        return None

# Calculate interest rate
def calculate_interest_rate(risk_level, loan_amount, income, cibil_score, age, loan_purpose):
    """
    Calculate interest rate based on multiple factors using Indian lending standards:
    - Risk level from the model
    - CIBIL score (credit score)
    - Loan amount relative to income
    - Age of applicant
    - Purpose of loan
    
    Special handling for education loans where income can be 0.
    
    Parameters:
    risk_level (str): Risk level from model ('Low', 'Medium', 'High')
    loan_amount (float): Amount of the loan in INR
    income (float): Annual income of the applicant in INR
    cibil_score (int): CIBIL credit score (300-900 range in India)
    age (int): Age of the applicant
    loan_purpose (str): Purpose of the loan (e.g., 'education', 'debt_consolidation', etc.)
    
    Returns:
    float: Calculated interest rate
    """
    # Base rate based on risk level (Indian bank rates)
    if risk_level == "Low":
        base_rate = 8.5  # Typical low-risk personal loan rate in India
    elif risk_level == "Medium":
        base_rate = 12.0  # Average personal loan rate in India
    else:  # High risk
        base_rate = 16.0  # Higher risk personal loan rate in India
    
    # CIBIL score adjustment (Indian CIBIL ranges from 300-900)
    if cibil_score >= 800:
        score_adjustment = -2.0  # Excellent credit in Indian context
    elif cibil_score >= 750:
        score_adjustment = -1.5
    elif cibil_score >= 700:
        score_adjustment = -0.75
    elif cibil_score >= 650:
        score_adjustment = 0
    elif cibil_score >= 600:
        score_adjustment = 1.0
    elif cibil_score >= 550:
        score_adjustment = 2.0
    else:
        score_adjustment = 3.5  # Poor credit in Indian context
    
    # Special handling for education loans - income can be 0
    if loan_purpose == "education":
        # For education loans, we don't penalize for low/no income
        lpi_adjustment = -0.5  # Favorable adjustment for education loans
    else:
        # Loan-to-income ratio adjustment (Indian banks typically prefer ratio below 0.5)
        loan_percent_income = loan_amount / income if income > 0 else 0.6
        if loan_percent_income <= 0.2:
            lpi_adjustment = -0.5
        elif loan_percent_income <= 0.3:
            lpi_adjustment = -0.25
        elif loan_percent_income <= 0.4:
            lpi_adjustment = 0
        elif loan_percent_income <= 0.5:
            lpi_adjustment = 0.5
        else:
            lpi_adjustment = 1.5  # Higher adjustment for high LTI ratios in Indian context
    
    # Age adjustment (Indian banks often prefer 25-50 age range)
    if age < 25:
        age_adjustment = 0.75  # Higher rates for younger borrowers
    elif age >= 25 and age <= 35:
        age_adjustment = -0.25  # Prime borrowing age
    elif age > 35 and age <= 55:
        age_adjustment = 0
    else:
        age_adjustment = 1.0  # Higher rates for elderly borrowers
    
    # Loan purpose adjustment (based on Indian lending preferences)
    purpose_adjustments = {
        "home_improvement": -0.5,    # Secured by property value
        "debt_consolidation": 0.75,  # Higher risk in Indian context
        "education": -1.0,           # Education loans have subsidized rates in India
        "medical": -0.5,             # Often given preference
        "personal": 1.0,             # General purpose loans have higher rates
        "venture": 2.0               # Business/venture loans seen as higher risk
    }
    
    purpose_adjustment = purpose_adjustments.get(loan_purpose, 0)
    
    # Calculate final rate
    final_rate = (
        base_rate 
        + score_adjustment 
        + lpi_adjustment 
        + age_adjustment
        + purpose_adjustment
    )
    
    # Special case: Education loans typically have lower caps in India
    if loan_purpose == "education":
        return max(min(final_rate, 15.0), 6.0)  # Education loan range in India
    else:
        # Ensure rate is within reasonable bounds for Indian loans
        # Personal loans in India typically range from 7% to 24%
        return (max(min(final_rate, 24.0), 7.0)*2)


# Mistral API helper function
def get_mistral_response(prompt, api_key, model="mistral-large-latest"):
    """
    Get a response from the Mistral API
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        # Check if the API returned an error
        if "error" in result:
            st.error(f"API Error: {result.get('error')}")
            return "Error in API response"
        
        # Check if the response has the expected structure
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            # If we can't get a proper response, use a fallback approach
            st.warning(f"Unexpected API response format")
            
            # For loan queries, return a default JSON
            if "loan" in prompt.lower() or "financial" in prompt.lower():
                return """
                {
                    "person_age": 35,
                    "person_income": 75000,
                    "person_home_ownership": "own",
                    "person_emp_length": 8,
                    "loan_intent": "home_improvement",
                    "loan_grade": "B",
                    "loan_amnt": 20000,
                    "loan_percent_income": 26.7,
                    "cb_person_default_on_file": "N",
                    "cb_person_cred_hist_length": 10,
                    "cibil_score": 720
                }
                """
                
            # For intent classification, make a best guess
            elif "classify" in prompt.lower():
                if "loan" in prompt.lower() or "financial" in prompt.lower():
                    return "financial_risk"
                elif "insurance" in prompt.lower() or "claim" in prompt.lower():
                    return "insurance_claim"
                else:
                    return "financial_risk"
                    
            # Generic fallback
            else:
                return "Unable to process this request"
                
    except Exception as e:
        st.error(f"Exception in Mistral API call: {str(e)}")
        return f"API error: {str(e)}"

# Base Agent class
class Agent:
    def __init__(self, name):
        self.name = name
        
    def process(self, query, context=None):
        raise NotImplementedError("Subclasses must implement this method")

# Financial Risk Agent
class FinancialRiskAgent(Agent):
    def __init__(self, model_path="loan_risk_rf_model.pkl"):
        super().__init__("FinancialRiskAgent")
        self.model = joblib.load(model_path)
        # Remove loan_int_rate from feature list
        self.feature_names = ['person_age', 'person_income', 'person_home_ownership',
                             'person_emp_length', 'loan_intent', 'loan_grade',
                             'loan_amnt', 'loan_percent_income',
                             'cb_person_default_on_file', 'cb_person_cred_hist_length']
        
    def preprocess_input(self, data):
        processed = {}
        
        home_ownership_map = {"rent": 0, "mortgage": 1, "own": 2, "other": 3}
        loan_intent_map = {"personal": 0, "education": 1, "medical": 2,
                          "venture": 3, "home_improvement": 4, "debt_consolidation": 5}
        loan_grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        default_map = {"Y": 1, "N": 0}
        
        # Handle categorical features with None check
        home_ownership = data.get("person_home_ownership", "")
        if home_ownership is not None:
            processed["person_home_ownership"] = home_ownership_map.get(str(home_ownership).lower(), 0)
        else:
            processed["person_home_ownership"] = 0
            
        loan_intent = data.get("loan_intent", "")
        if loan_intent is not None:
            processed["loan_intent"] = loan_intent_map.get(str(loan_intent).lower(), 0)
        else:
            processed["loan_intent"] = 0
            
        loan_grade = data.get("loan_grade", "")
        if loan_grade is not None:
            processed["loan_grade"] = loan_grade_map.get(str(loan_grade).upper(), 0)
        else:
            processed["loan_grade"] = 0
            
        default_on_file = data.get("cb_person_default_on_file", "")
        if default_on_file is not None:
            processed["cb_person_default_on_file"] = default_map.get(str(default_on_file).upper(), 0)
        else:
            processed["cb_person_default_on_file"] = 0
            
        # Handle numerical features
        for feature in ["person_age", "person_income", "person_emp_length",
                      "loan_amnt", "loan_percent_income",
                      "cb_person_cred_hist_length"]:
            try:
                processed[feature] = float(data.get(feature, 0) or 0)
            except (ValueError, TypeError):
                processed[feature] = 0
                
        return processed
        
    def process(self, query, context=None):
        if context:
            # Check age restrictions
            age = context.get("person_age", 0)
            cibil_score = context.get("cibil_score", 650)  # Default CIBIL score
            
            # Underage applicant
            if age and age < 18:
                return {
                    "prediction": 1,  # Default/Reject
                    "default_probability": 1.0,
                    "risk_level": "Ineligible",
                    "recommendation": "Guardian Required",
                    "message": "We cannot process loan applications for individuals under 18 years old. Please have a parent or legal guardian apply on your behalf."
                }
            
            # Features for the model (no interest rate)
            features = self.preprocess_input(context)
            feature_df = pd.DataFrame({feature: [features.get(feature, 0)] for feature in self.feature_names})
            
            prediction = self.model.predict(feature_df)[0]
            probability = self.model.predict_proba(feature_df)[0]
            
            prob_default = probability[1]
            if prob_default < 0.3:
                risk_level = "Low"
            elif prob_default < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Adjust approval threshold based on age
            approval_threshold = 0.5
            if age > 60:
                approval_threshold = 0.4
            
            recommendation = "Approved" if prob_default < approval_threshold else "Rejected"
            
            # Calculate interest rate only if approved
            interest_rate = None
            if recommendation == "Approved":
                loan_purpose = context.get("loan_intent", "personal")
                interest_rate = calculate_interest_rate(
                    risk_level=risk_level,
                    loan_amount=context.get("loan_amnt", 0),
                    income=context.get("person_income", 1),
                    cibil_score=cibil_score,
                    age=age,
                    loan_purpose=loan_purpose
                )
            
            # Special message for elderly applicants
            message = None
            if age > 60:
                message = "We've carefully evaluated your application. Given your age, we recommend reviewing the loan terms carefully and considering how this loan fits into your long-term financial plans."
            
            result = {
                "prediction": int(prediction),
                "default_probability": float(prob_default),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "message": message
            }
            
            # Add interest rate only if approved
            if recommendation == "Approved":
                result["interest_rate"] = interest_rate
                
            return result
            
        else:
            return {"error": "No context provided"}

# Insurance Claims Agent
class InsuranceClaimsAgent(Agent):
    def __init__(self):
        super().__init__("InsuranceClaimsAgent")
        self.claims_data = [
            {"id": 1, "description": "Car accident on highway, rear-ended by another vehicle", "risk_factors": ["distracted driving", "tailgating"], "severity": "moderate"},
            {"id": 2, "description": "Home water damage from burst pipe during winter", "risk_factors": ["freezing weather", "old plumbing"], "severity": "severe"},
            {"id": 3, "description": "Slip and fall in grocery store", "risk_factors": ["wet floor", "no warning sign"], "severity": "minor"},
            {"id": 4, "description": "Theft of laptop from car", "risk_factors": ["visible valuables", "unsecured vehicle"], "severity": "minor"},
            {"id": 5, "description": "House fire started in kitchen", "risk_factors": ["unattended cooking", "electrical fault"], "severity": "severe"}
        ]
        
    def similarity_search(self, query, api_key, top_k=2):
        prompt = f"""
        I have a claim description: "{query}"

        Compare it to the following claims and identify the 2 most similar based on content and risk factors:

        1. Car accident on highway, rear-ended by another vehicle
        2. Home water damage from burst pipe during winter
        3. Slip and fall in grocery store
        4. Theft of laptop from car
        5. House fire started in kitchen

        Return ONLY the numbers of the two most similar claims, separated by a comma.
        For example: "1,3"
        """
        
        response = get_mistral_response(prompt, api_key)
        
        try:
            claim_ids = [int(id.strip()) for id in response.strip().split(",")]
            similar_claims = [self.claims_data[id-1] for id in claim_ids if 1 <= id <= len(self.claims_data)]
            return similar_claims
        except:
            return self.claims_data[:2]
            
    def process(self, query, context=None):
        # Extract API key if available in context
        api_key = context.get("api_key", "") if context else ""
        claim_description = context.get("claim_description", query) if context else query
        
        similar_claims = self.similarity_search(claim_description, api_key)
        
        all_risk_factors = []
        for claim in similar_claims:
            all_risk_factors.extend(claim.get("risk_factors", []))
            
        risk_factor_counts = Counter(all_risk_factors)
        common_risk_factors = [factor for factor, count in risk_factor_counts.most_common(3)]
        
        severity_scores = {"minor": 1, "moderate": 2, "severe": 3}
        avg_severity = sum(severity_scores.get(claim.get("severity", "moderate"), 2)
                          for claim in similar_claims) / len(similar_claims)
                          
        if avg_severity < 1.5:
            estimated_severity = "minor"
        elif avg_severity < 2.5:
            estimated_severity = "moderate"
        else:
            estimated_severity = "severe"
            
        recommendation_prompt = f"""
        Based on an insurance claim described as: "{claim_description}"

        Similar past claims involved these risk factors: {', '.join(common_risk_factors)}
        The estimated severity is: {estimated_severity}

        Provide a brief recommendation for handling this claim, including:
        1. Initial assessment steps
        2. Documentation requirements
        3. Processing timeline

        Keep it concise (3-4 sentences maximum).
        """
        
        recommendation = get_mistral_response(recommendation_prompt, api_key)
        
        return {
            "similar_claims": similar_claims,
            "common_risk_factors": common_risk_factors,
            "estimated_severity": estimated_severity,
            "recommendation": recommendation
        }

# Orchestrator Agent
class OrchestratorAgent(Agent):
    def __init__(self, api_key=""):
        super().__init__("OrchestratorAgent")
        self.api_key = api_key
        self.financial_agent = FinancialRiskAgent()
        self.insurance_agent = InsuranceClaimsAgent()
        
    def classify_intent(self, query):
        prompt = f"""
        Analyze this query and classify it as EITHER 'financial_risk' OR 'insurance_claim'.

        If it's about loans, loan approval, credit risk, or debt, classify as 'financial_risk'.
        If it's about insurance claims, damage, accidents, or coverage, classify as 'insurance_claim'.

        Query: "{query}"

        Return ONLY ONE word: 'financial_risk' or 'insurance_claim'
        """
        
        intent = get_mistral_response(prompt, self.api_key).strip().lower()
        
        if "financial" in intent or "loan" in intent or "risk" in intent:
            return "financial_risk"
        elif "insurance" in intent or "claim" in intent:
            return "insurance_claim"
        else:
            return "financial_risk"
            
    def process(self, query, context=None, force_intent=None):
        # Add API key to context
        if context is None:
            context = {}
        context["api_key"] = self.api_key
        
        # Use forced intent if provided, else get from context or classify
        intent = force_intent or (context.get("intent") if context and "intent" in context else self.classify_intent(query))
        
        if intent == "financial_risk":
            result = self.financial_agent.process(query, context)
            
            # Special handling for age-related messages
            if "message" in result and result["message"]:
                formatted_response = f"""
                Financial Risk Assessment:

                Risk Level: {result.get('risk_level', 'Unknown')}
                Default Probability: {result.get('default_probability', 0) * 100:.1f}%
                Recommendation: {result.get('recommendation', 'Unknown')}
                """
                
                if "interest_rate" in result:
                    formatted_response += f"Interest Rate: {result['interest_rate']:.2f}%\n"
                    
                formatted_response += f"\nNote: {result['message']}"
            else:
                formatted_response = f"""
                Financial Risk Assessment:

                Risk Level: {result.get('risk_level', 'Unknown')}
                Default Probability: {result.get('default_probability', 0) * 100:.1f}%
                Recommendation: {result.get('recommendation', 'Unknown')}
                """
                
                if "interest_rate" in result:
                    formatted_response += f"Interest Rate: {result['interest_rate']:.2f}%"
                    
            result["formatted_response"] = formatted_response
            return result
            
        elif intent == "insurance_claim":
            result = self.insurance_agent.process(query, context)
            
            risk_factors = ", ".join(result.get('common_risk_factors', ['Unknown']))
            formatted_response = f"""
            Insurance Claim Assessment:

            Estimated Severity: {result.get('estimated_severity', 'Unknown')}
            Common Risk Factors: {risk_factors}

            Recommendation:
            {result.get('recommendation', 'No recommendation available.')}
            """
            
            result["formatted_response"] = formatted_response
            return result
            
        else:
            return {
                "error": "Could not determine intent",
                "raw_query": query
            }

# Loan Eligibility Checker
class LoanEligibilityChecker:
    @staticmethod
    def is_eligible(query=None, data=None):
        if data is not None:
            # Using direct data from form
            age = data.get("person_age")
            income = data.get("person_income")
            employment = data.get("person_emp_length")
            credit_history = data.get("cb_person_cred_hist_length")
            cibil_score = data.get("cibil_score")
            
            # Eligibility checks
            if age < 18:
                return "Error: Applicants must be at least 18 years old to apply for a loan."
            if age > 65:
                return "Error: Loans are only available for individuals aged 18 to 65."
            if income < 20000:
                return "Error: Minimum income requirement for a loan is $20,000."
            if employment < 1:
                return "Error: Applicants must have at least 1 year of employment."
            if credit_history < 3:
                return "Error: Applicants must have a credit history of at least 3 years."
            if cibil_score < 700:
                return f"Error: CIBIL score {cibil_score} is too low. Minimum required CIBIL score is 700."
            if cibil_score > 900:
                return f"Error: Invalid CIBIL score {cibil_score}. Maximum CIBIL score is 900."

            # Advanced Checks
            if age < 20:
                if employment > 2:
                    return f"Error: At age {age}, employment cannot exceed 2 years."
                if credit_history > 3:
                    return f"Error: At age {age}, credit history cannot exceed 3 years."

            if age < 25 and credit_history > 7:
                return f"Error: At age {age}, credit history cannot exceed 7 years."

            if employment > (age - 18):  # People can't work before 18
                return f"Error: Employment duration of {employment} years is unrealistic for a {age}-year-old."

            if credit_history > (age - 18):  # Credit history cannot be more than age minus 18
                return f"Error: Credit history of {credit_history} years is unrealistic for a {age}-year-old."

            return "Eligible"
            
        elif query:
            # Updated regex patterns for more flexible matching
            age_match = re.search(r"I[''m am]+ (\d+) years old", query, re.IGNORECASE)
            income_match = re.search(r"income of \$?([\d,]+)", query, re.IGNORECASE)
            employment_match = re.search(r"employed for (\d+) years", query, re.IGNORECASE)
            credit_match = re.search(r"credit history of (\d+) years", query, re.IGNORECASE)
            cibil_match = re.search(r"CIBIL score is (\d+)", query, re.IGNORECASE)

            # Identify missing information
            missing_fields = []
            if not age_match:
                missing_fields.append("Age")
            if not income_match:
                missing_fields.append("Income")
            if not employment_match:
                missing_fields.append("Employment years")
            if not credit_match:
                missing_fields.append("Credit history")
            if not cibil_match:
                missing_fields.append("CIBIL score")

            # If any required field is missing, return an error message
            if missing_fields:
                return f"Error: Missing information. Please provide {', '.join(missing_fields)}."

            # Convert extracted values
            age = int(age_match.group(1))
            income = int(income_match.group(1).replace(",", ""))
            employment = int(employment_match.group(1))
            credit_history = int(credit_match.group(1))
            cibil_score = int(cibil_match.group(1))

            # Then perform the same checks as in the data method
            if age < 18:
                return "Error: Applicants must be at least 18 years old to apply for a loan."
            elif age > 65:
                return "Error: Loans are only available for individuals aged 18 to 65."
            if income < 20000:
                return "Error: Minimum income requirement for a loan is $20,000."
            if employment < 1:
                return "Error: Applicants must have at least 1 year of employment."
            if credit_history < 3:
                return "Error: Applicants must have a credit history of at least 3 years."
            if cibil_score < 700:
                return f"Error: CIBIL score {cibil_score} is too low. Minimum required CIBIL score is 700."
            if cibil_score > 900:
                return f"Error: Invalid CIBIL score {cibil_score}. Maximum CIBIL score is 900."

            # Advanced Checks
            if age < 20:
                if employment > 2:
                    return f"Error: At age {age}, employment cannot exceed 2 years."
                if credit_history > 3:
                    return f"Error: At age {age}, credit history cannot exceed 3 years."

            if age < 25 and credit_history > 7:
                return f"Error: At age {age}, credit history cannot exceed 7 years."

            if employment > (age - 18):  # People can't work before 18
                return f"Error: Employment duration of {employment} years is unrealistic for a {age}-year-old."

            if credit_history > (age - 17):  # Credit history cannot be more than age minus 18
                return f"Error: Credit history of {credit_history} years is unrealistic for a {age}-year-old."
            
            return "Eligible"
        
        else:
            return "Error: No data or query provided for eligibility check."

# Home page
def home_page():
    st.header("Welcome to FINORA")
    st.markdown("""
    FINORA is a comprehensive platform for financial risk assessment and insurance claims processing.
    
    ### Features:
    - **Loan Risk Assessment**: Evaluate loan applications for approval and determine interest rates
    - **Insurance Claim Processing**: Analyze insurance claims and provide recommendations
    - **Advanced ML Model**: Uses Random Forest classification for accurate risk predictions
    - **LLM Integration**: Leverages Mistral AI for enhanced analysis capabilities
    
    ### Getting Started:
    - Select a service from the sidebar
    - For LLM features, you'll need to provide your Mistral API key
    - Enter your information to receive an assessment
    """)
    
    # Display sample dashboard
    st.subheader("Loan Approval Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Approval Rate", "78.5%", "2.1%")
    with col2:
        st.metric("Average Interest Rate", "9.2%", "-0.5%")
    with col3:
        st.metric("Risk Level", "Medium", "")
        
    # Sample chart
    st.subheader("Monthly Loan Approvals")
    chart_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Approved': [65, 72, 80, 75, 82, 85],
        'Rejected': [35, 28, 20, 25, 18, 15]
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_data.plot(x='Month', y=['Approved', 'Rejected'], kind='bar', stacked=True, ax=ax)
    plt.ylabel('Number of Applications')
    plt.title('Monthly Loan Application Results')
    st.pyplot(fig)
    
    st.info("Note: This application uses a machine learning model to provide financial assessments. The model has been trained on sample data and should be used for demonstration purposes only.")

# Loan risk assessment page
# Loan risk assessment page
def loan_risk_page():
    st.header("Loan Risk Assessment")
    
    if not model_exists:
        st.warning("No trained model found. Please go to the Model Training page to train a model first.")
        if st.button("Go to Model Training"):
            st.session_state['page'] = "Model Training"
            st.experimental_rerun()
        return
    
    # API key for Mistral (optional)
    with st.expander("Mistral API Settings (Optional)"):
        api_key = st.text_input("Enter your Mistral API key (optional)", type="password")
        st.markdown("An API key is optional, but enables enhanced features.")
    
    # User can input loan information
    st.subheader("Loan Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=75000)
        emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=8.0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=20000)
        
    with col2:
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_intent = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"])
        loan_grade = st.selectbox("Loan Grade (if known)", ["A", "B", "C", "D", "E", "F", "G"])
        default_history = st.selectbox("Previous Defaults?", ["No", "Yes"])
    
    credit_history = st.slider("Credit History Length (years)", 0, 30, 10)
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)
    
    # Calculate loan percent income
    loan_percent = (loan_amount / income * 100) if income > 0 else 0
    st.write(f"Loan Amount is {loan_percent:.1f}% of annual income")
    
    # Define process_loan_application function first (IMPORTANT!)
    def process_loan_application():
        # Prepare data for model
        loan_data = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home_ownership.lower(),
            "person_emp_length": emp_length,
            "loan_intent": loan_intent.lower().replace(" ", "_"),
            "loan_grade": loan_grade,
            "loan_amnt": loan_amount,
            "loan_percent_income": loan_percent / 100,  # Convert to decimal
            "cb_person_default_on_file": "Y" if default_history == "Yes" else "N",
            "cb_person_cred_hist_length": credit_history,
            "cibil_score": cibil_score
        }
        
        # Create agent and process
        try:
            financial_agent = FinancialRiskAgent()
            result = financial_agent.process("", loan_data)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Assessment")
                risk_level = result.get("risk_level")
                
                # Color code based on risk level
                if risk_level == "Low":
                    st.success(f"Risk Level: {risk_level}")
                elif risk_level == "Medium":
                    st.warning(f"Risk Level: {risk_level}")
                else:
                    st.error(f"Risk Level: {risk_level}")
                    
                st.write(f"Default Probability: {result.get('default_probability', 0) * 100:.1f}%")
                recommendation = result.get('recommendation', 'Unknown')
                
                if recommendation == "Approved":
                    st.success(f"Recommendation: {recommendation}")
                else:
                    st.error(f"Recommendation: {recommendation}")
            
            with col2:
                if "interest_rate" in result:
                    st.subheader("Loan Terms")
                    
                    interest_rate = result.get("interest_rate", 0)
                    # Color code interest rate
                    if interest_rate < 8:
                        st.success(f"Interest Rate: {interest_rate:.2f}%")
                    elif interest_rate < 15:
                        st.warning(f"Interest Rate: {interest_rate:.2f}%")
                    else:
                        st.error(f"Interest Rate: {interest_rate:.2f}%")
                    
                    # Calculate monthly payment (simple)
                    years = 5  # Assuming 5-year loan
                    monthly_rate = interest_rate / 100 / 12
                    num_payments = years * 12
                    monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
                    
                    st.write(f"Estimated Monthly Payment (5-year term): ${monthly_payment:.2f}")
                    st.write(f"Total Interest Paid: ${(monthly_payment * num_payments - loan_amount):.2f}")
            
            # Special message if any
            if "message" in result and result["message"]:
                st.info(result["message"])
                
            # Show feature importance if approved
            if recommendation == "Approved" and "interest_rate" in result:
                st.subheader("Factors Influencing Your Rate")
                factors = {
                    "Credit Score": f"{(900 - cibil_score) / 6:.0f}%",
                    "Income Level": f"{(100000 - min(income, 100000)) / 1000:.0f}%",
                    "Loan Amount": f"{min(loan_amount / 1000, 30):.0f}%",
                    "Employment History": f"{max(10 - emp_length, 0) * 5:.0f}%",
                    "Credit History": f"{max(20 - credit_history, 0) * 3:.0f}%"
                }
                
                # Create factor influence chart
                factor_df = pd.DataFrame({
                    'Factor': list(factors.keys()),
                    'Influence': [float(v.replace('%', '')) for v in factors.values()]
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(factor_df['Factor'], factor_df['Influence'], color='skyblue')
                ax.set_xlabel('Influence on Interest Rate (%)')
                ax.set_title('Factors Influencing Your Interest Rate')
                
                # Add values at the end of each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.0f}%', 
                            ha='left', va='center')
                
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Submit button (AFTER the function definition)
    if st.button("Assess Loan Application"):
        # Prepare data for eligibility check
        loan_data = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp_length,
            "cb_person_cred_hist_length": credit_history,
            "cibil_score": cibil_score
        }
        
        # Check eligibility first
        eligibility_result = LoanEligibilityChecker.is_eligible(data=loan_data)
        
        if "Error" in eligibility_result:
            st.error(eligibility_result)
        elif "Warning" in eligibility_result:
            st.warning(eligibility_result)
            # Continue with assessment
            process_loan_application()
        else:
            # Eligible, proceed with assessment
            process_loan_application()
# Insurance claim page
def insurance_claim_page():
    st.header("Insurance Claim Processing")
    
    # API key for Mistral (required for insurance claims)
    api_key = st.text_input("Enter your Mistral API key (required for LLM features)", type="password")
    
    if not api_key:
        st.warning("A Mistral API key is required for full insurance claim processing features.")
    
    # Claim information form
    st.subheader("Claim Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claim_type = st.selectbox("Claim Type", ["Auto Insurance", "Home Insurance", "Health Insurance", "Travel Insurance", "Other"])
        incident_date = st.date_input("Incident Date")
    
    with col2:
        policy_number = st.text_input("Policy Number (optional)")
        claim_amount = st.number_input("Estimated Claim Amount ($)", min_value=0, value=5000)
    
    claim_description = st.text_area("Detailed Claim Description", placeholder="Please provide a detailed description of the incident and damages...")
    
    # Submit button
    if st.button("Process Claim"):
        if not claim_description:
            st.error("Please provide a claim description.")
        else:
            try:
                with st.spinner("Processing claim..."):
                    # If API key is not provided, use simpler processing
                    if not api_key:
                        st.warning("Processing with limited features (no LLM integration)")
                        
                        # Simple severity estimation based on claim amount
                        if claim_amount < 2000:
                            severity = "minor"
                        elif claim_amount < 10000:
                            severity = "moderate"
                        else:
                            severity = "severe"
                            
                        # Basic risk factors based on claim type
                        risk_factors = {
                            "Auto Insurance": ["vehicle damage", "collision"],
                            "Home Insurance": ["property damage", "structural issue"],
                            "Health Insurance": ["medical expenses", "treatment"],
                            "Travel Insurance": ["trip cancellation", "lost items"],
                            "Other": ["unexpected loss", "damage"]
                        }.get(claim_type, ["unknown"])
                        
                        result = {
                            "estimated_severity": severity,
                            "common_risk_factors": risk_factors,
                            "recommendation": "Please submit all relevant documentation including photos, receipts, and incident reports. Our claims adjuster will review your case within 7-10 business days.",
                            "similar_claims": []
                        }
                    else:
                        # Full processing with LLM
                        insurance_agent = InsuranceClaimsAgent()
                        result = insurance_agent.process(claim_description, {"api_key": api_key})
                
                # Display results
                st.subheader("Claim Assessment Results")
                
                # Severity indicator with color
                severity = result.get("estimated_severity", "Unknown")
                if severity == "minor":
                    st.success(f"Estimated Severity: {severity.title()}")
                elif severity == "moderate":
                    st.warning(f"Estimated Severity: {severity.title()}")
                else:
                    st.error(f"Estimated Severity: {severity.title()}")
                
                # Risk factors
                st.subheader("Identified Risk Factors")
                risk_factors = result.get("common_risk_factors", [])
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor.title()}")
                else:
                    st.write("No specific risk factors identified.")
                
                # Similar claims (only shown with API key)
                if api_key:
                    st.subheader("Similar Past Claims")
                    similar_claims = result.get("similar_claims", [])
                    for i, claim in enumerate(similar_claims):
                        with st.expander(f"Similar Claim #{i+1}"):
                            st.write(f"Description: {claim.get('description')}")
                            st.write(f"Severity: {claim.get('severity', 'Unknown').title()}")
                            st.write("Risk Factors: " + ", ".join([f.title() for f in claim.get('risk_factors', [])]))
                
                # Recommendations
                st.subheader("Recommendations")
                recommendation = result.get("recommendation", "No specific recommendations available.")
                st.info(recommendation)
                
                # Estimated timeline
                st.subheader("Estimated Processing Timeline")
                
                # Timeline based on severity
                if severity == "minor":
                    timeline_days = "3-5 business days"
                    required_docs = ["Claim form", "Photos of damage", "Cost estimates"]
                elif severity == "moderate":
                    timeline_days = "7-10 business days"
                    required_docs = ["Claim form", "Photos of damage", "Cost estimates", "Police report (if applicable)", "Medical records (if applicable)"]
                else:
                    timeline_days = "14-21 business days"
                    required_docs = ["Claim form", "Detailed photos of damage", "Professional damage assessment", "Cost estimates from licensed contractors", "Police report", "Medical records (if applicable)", "Witness statements (if applicable)"]
                
                st.write(f"Estimated processing time: {timeline_days}")
                
                # Required documentation
                st.subheader("Required Documentation")
                for doc in required_docs:
                    st.write(f"• {doc}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Import missing module
import io

# Main app logic
if page == "Home":
    home_page()
elif page == "Loan Risk Assessment":
    loan_risk_page()
elif page == "Insurance Claim Processing":
    insurance_claim_page()
elif page == "Model Training":
    train_model()