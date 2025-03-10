import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import requests
from typing import Dict, Any, List, Union
from collections import Counter
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="FINORA",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with black background
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .main {
        background-color: #121212;
    }
    .main-header {
        font-size: 2.5rem;
        color: #64B5F6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #90CAF9;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #333;
    }
    .highlight {
        color: #90CAF9;
        font-weight: bold;
    }
    .approved {
        color: #4CAF50;
        font-weight: bold;
    }
    .rejected {
        color: #F44336;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #9E9E9E;
        font-style: italic;
    }
    .stApp {
        background-color: #121212;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    .Widget>label {
        color: #E0E0E0;
    }
    .stTextInput>div>div>input {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .stSelectbox>div>div>div {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    button {
        background-color: #1976D2;
    }
    .reportview-container .main .block-container {
        background-color: #121212;
    }
    div[data-testid="stForm"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Functions
def get_mistral_response(prompt: str, api_key: str, model: str = "mistral-large-latest") -> str:
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

        if "error" in result:
            st.error(f"API Error: {result.get('error')}")
            return "Error in API response"

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            st.warning(f"Unexpected API response format")
            
            # Fallback responses
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
                    "cb_person_cred_hist_length": 10
                }
                """
            elif "classify" in prompt.lower():
                if "loan" in prompt.lower() or "financial" in prompt.lower():
                    return "financial_risk"
                elif "insurance" in prompt.lower() or "claim" in prompt.lower():
                    return "insurance_claim"
                else:
                    return "financial_risk"
            else:
                return "Unable to process this request"

    except Exception as e:
        st.error(f"Exception in Mistral API call: {str(e)}")
        return f"API error: {str(e)}"

# Calculate interest rate based on risk factors and CIBIL score
def calculate_interest_rate(risk_level: str, loan_amount: float, income: float, 
                           cibil_score: int, age: int, loan_purpose: str) -> float:
    # Base rate based on risk level
    if risk_level == "Low":
        base_rate = 6.5
    elif risk_level == "Medium":
        base_rate = 9.0
    else:  # High risk
        base_rate = 12.0
    
    # Adjustments based on CIBIL score
    if cibil_score >= 750:
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
        score_adjustment = 3.0
    
    # Loan-to-income ratio adjustment
    lti_ratio = loan_amount / income if income > 0 else 1
    if lti_ratio > 0.5:
        lti_adjustment = 1.0
    elif lti_ratio > 0.3:
        lti_adjustment = 0.5
    else:
        lti_adjustment = 0
    
    # Age adjustment - higher rates for very young or elderly applicants
    if age < 25:
        age_adjustment = 0.5
    elif age > 60:
        age_adjustment = 0.75
    else:
        age_adjustment = 0
    
    # Loan purpose adjustment
    purpose_adjustments = {
        "home_improvement": -0.25,
        "debt_consolidation": 0.25,
        "education": -0.5,
        "medical": -0.25,
        "personal": 0.5,
        "venture": 1.0
    }
    purpose_adjustment = purpose_adjustments.get(loan_purpose, 0)
    
    # Calculate final rate
    final_rate = base_rate + score_adjustment + lti_adjustment + age_adjustment + purpose_adjustment
    
    # Ensure rate is within reasonable bounds
    return max(min(final_rate, 24.0), 4.0)

# Agent classes
class Agent:
    def __init__(self, name: str):
        self.name = name

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

class FinancialRiskAgent(Agent):
    def __init__(self, model_path: str = "loan_risk_rf_model.pkl", api_key: str = None):
        super().__init__("FinancialRiskAgent")
        self.model = joblib.load(model_path)
        self.feature_names = ['person_age', 'person_income', 'person_home_ownership',
                            'person_emp_length', 'loan_intent', 'loan_grade',
                            'loan_amnt', 'loan_percent_income',
                            'cb_person_default_on_file', 'cb_person_cred_hist_length']
        self.api_key = api_key

    def preprocess_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
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

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if context:
            # Check age restrictions
            age = context.get("person_age", 0)
            cibil_score = context.get("cibil_score", 650)
            
            # Underage applicant
            if age and age < 18:
                return {
                    "prediction": 1,  # Default/Reject
                    "default_probability": 1.0,
                    "risk_level": "Ineligible",
                    "recommendation": "Guardian Required",
                    "message": "We cannot process loan applications for individuals under 18 years old. Please have a parent or legal guardian apply on your behalf."
                }
            
            # Prepare features (excluding interest rate which we'll calculate)
            features = self.preprocess_input(context)
            feature_df = pd.DataFrame({feature: [features.get(feature, 0)] for feature in self.feature_names})
            
            prediction = self.model.predict(feature_df)[0]
            probability = self.model.predict_proba(feature_df)[0]
            prob_default = probability[1]
            
            # Determine risk level
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
            
            # Calculate interest rate based on risk and CIBIL score
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
            
            return {
                "prediction": int(prediction),
                "default_probability": float(prob_default),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "interest_rate": interest_rate,
                "message": message
            }
            
        else:
            parameter_prompt = f"""
            Extract the following financial parameters from this query. Return a JSON object.
            If a value is not present, use a reasonable default or null.

            Query: "{query}"

            Parameters needed:
            - person_age: Age of the person (number)
            - person_income: Annual income (number)
            - person_home_ownership: Housing status (rent, mortgage, own, other)
            - person_emp_length: Employment length in years (number)
            - loan_intent: Purpose of loan (personal, education, medical, venture, home_improvement, debt_consolidation)
            - loan_grade: Loan grade if known (A-G)
            - loan_amnt: Loan amount requested (number)
            - loan_percent_income: Loan amount as percentage of income (number)
            - cb_person_default_on_file: Previous defaults (Y/N)
            - cb_person_cred_hist_length: Credit history length in years (number)

            Return ONLY a valid JSON object with these fields.
            """

            param_response = get_mistral_response(parameter_prompt, self.api_key)

            try:
                # Extract JSON from response
                json_match = re.search(r'(\{.*\})', param_response, re.DOTALL)
                if json_match:
                    params = json.loads(json_match.group(1))
                else:
                    params = json.loads(param_response)

                st.info(f"Extracted parameters: {params}")

                return self.process(query, params)

            except json.JSONDecodeError:
                st.error(f"Failed to parse JSON: {param_response}")
                return {
                    "error": "Could not extract financial parameters",
                    "raw_response": param_response
                }

class InsuranceClaimsAgent(Agent):
    def __init__(self, api_key: str = None):
        super().__init__("InsuranceClaimsAgent")
        self.api_key = api_key
        self.claims_data = [
            {"id": 1, "description": "Car accident on highway, rear-ended by another vehicle", "risk_factors": ["distracted driving", "tailgating"], "severity": "moderate"},
            {"id": 2, "description": "Home water damage from burst pipe during winter", "risk_factors": ["freezing weather", "old plumbing"], "severity": "severe"},
            {"id": 3, "description": "Slip and fall in grocery store", "risk_factors": ["wet floor", "no warning sign"], "severity": "minor"},
            {"id": 4, "description": "Theft of laptop from car", "risk_factors": ["visible valuables", "unsecured vehicle"], "severity": "minor"},
            {"id": 5, "description": "House fire started in kitchen", "risk_factors": ["unattended cooking", "electrical fault"], "severity": "severe"}
        ]

    def similarity_search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
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

        response = get_mistral_response(prompt, self.api_key)

        try:
            claim_ids = [int(id.strip()) for id in response.strip().split(",")]
            similar_claims = [self.claims_data[id-1] for id in claim_ids if 1 <= id <= len(self.claims_data)]
            return similar_claims
        except:
            return self.claims_data[:2]

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        claim_description = context.get("claim_description", query) if context else query

        similar_claims = self.similarity_search(claim_description)

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

        recommendation = get_mistral_response(recommendation_prompt, self.api_key)

        return {
            "similar_claims": similar_claims,
            "common_risk_factors": common_risk_factors,
            "estimated_severity": estimated_severity,
            "recommendation": recommendation
        }

class OrchestratorAgent(Agent):
    def __init__(self, api_key: str = None, model_path: str = "loan_risk_rf_model.pkl"):
        super().__init__("OrchestratorAgent")
        self.api_key = api_key
        self.financial_agent = FinancialRiskAgent(model_path=model_path, api_key=api_key)
        self.insurance_agent = InsuranceClaimsAgent(api_key=api_key)

    def classify_intent(self, query: str) -> str:
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

    def process(self, query: str, context: Dict[str, Any] = None, force_intent: str = None) -> Dict[str, Any]:
        # Use forced intent if provided, else get from context or classify
        intent = force_intent or (context.get("intent") if context and "intent" in context else self.classify_intent(query))

        if intent == "financial_risk":
            result = self.financial_agent.process(query, context)
            
            # Special handling for age-related messages
            if "message" in result and result["message"]:
                formatted_response = f"""
                <div class="result-box">
                    <h3>Financial Risk Assessment</h3>
                    <p><span class="highlight">Risk Level:</span> {result.get('risk_level', 'Unknown')}</p>
                    <p><span class="highlight">Default Probability:</span> {result.get('default_probability', 0) * 100:.1f}%</p>
                    <p><span class="highlight">Recommendation:</span> <span class="{'approved' if result.get('recommendation') == 'Approved' else 'rejected'}">{result.get('recommendation', 'Unknown')}</span></p>
                    {f'<p><span class="highlight">Offered Interest Rate:</span> {result.get("interest_rate", 0):.2f}%</p>' if result.get('recommendation') == 'Approved' else ''}
                    <p><b>Note:</b> {result.get('message', '')}</p>
                </div>
                """
            else:
                formatted_response = f"""
                <div class="result-box">
                    <h3>Financial Risk Assessment</h3>
                    <p><span class="highlight">Risk Level:</span> {result.get('risk_level', 'Unknown')}</p>
                    <p><span class="highlight">Default Probability:</span> {result.get('default_probability', 0) * 100:.1f}%</p>
                    <p><span class="highlight">Recommendation:</span> <span class="{'approved' if result.get('recommendation') == 'Approved' else 'rejected'}">{result.get('recommendation', 'Unknown')}</span></p>
                    {f'<p><span class="highlight">Offered Interest Rate:</span> {result.get("interest_rate", 0):.2f}%</p>' if result.get('recommendation') == 'Approved' else ''}
                </div>
                """

            result["formatted_response"] = formatted_response
            return result

        elif intent == "insurance_claim":
            result = self.insurance_agent.process(query, context)

            risk_factors = ", ".join(result.get('common_risk_factors', ['Unknown']))
            formatted_response = f"""
            <div class="result-box">
                <h3>Insurance Claim Assessment</h3>
                <p><span class="highlight">Estimated Severity:</span> {result.get('estimated_severity', 'Unknown')}</p>
                <p><span class="highlight">Common Risk Factors:</span> {risk_factors}</p>
                <h4>Recommendation:</h4>
                <p>{result.get('recommendation', 'No recommendation available.')}</p>
            </div>
            """

            result["formatted_response"] = formatted_response
            return result

        else:
            return {
                "error": "Could not determine intent",
                "raw_query": query
            }

# Main Streamlit app
def main():
    st.markdown("<h1 class='main-header'>FINORA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Loan & Insurance Approval System</h2>", unsafe_allow_html=True)
    
    # API Key Input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Mistral API Key", type="password")
        st.markdown("---")
        st.write("**About FINORA**")
        st.write("FINORA is an AI-powered platform for financial risk assessment and insurance claim processing.")
        st.markdown("---")
        st.write("**How to use**")
        st.write("1. Enter your Mistral API key")
        st.write("2. Select a service type")
        st.write("3. Fill in the relevant information")
        st.write("4. Get your assessment results")
        
        # Model file checker
        model_file = "loan_risk_rf_model.pkl"
        if not os.path.exists(model_file):
            st.error(f"Model file '{model_file}' not found!")
            st.warning("Please make sure the model file is in the same directory as this app.")
        else:
            st.success("Model loaded successfully!")
    
    if not api_key:
        st.warning("Please enter your Mistral API key in the sidebar to continue.")
        return
    
    # Main content
    service_type = st.selectbox(
        "Select service type",
        ["Loan Assessment", "Insurance Claim Processing"]
    )
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(api_key=api_key)
    
    if service_type == "Loan Assessment":
        st.markdown("<h3 class='sub-header'>Loan Application Assessment</h3>", unsafe_allow_html=True)
        
        # Create three columns for input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=75000, step=1000)
            home_ownership = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own", "Other"])
            
        with col2:
            employment_length = st.number_input("Years Employed", min_value=0, max_value=50, value=8)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=20000, step=1000)
            loan_intent = st.selectbox(
                "Loan Purpose", 
                ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"]
            )
            
        with col3:
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
            default_history = st.selectbox("Previous Defaults?", ["No", "Yes"])
            credit_history_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)
        
        # Calculate loan percent income
        loan_percent = (loan_amount / income) * 100 if income > 0 else 0
        
        # Submit button
        if st.button("Assess Loan Application"):
            with st.spinner("Analyzing your loan application..."):
                # Prepare the context
                context = {
                    "person_age": age,
                    "person_income": income,
                    "person_home_ownership": home_ownership.lower(),
                    "person_emp_length": employment_length,
                    "loan_intent": loan_intent.lower().replace(" ", "_"),
                    "loan_grade": "", # We'll determine this based on CIBIL score
                    "loan_amnt": loan_amount,
                    "loan_percent_income": loan_percent,
                    "cb_person_default_on_file": "Y" if default_history == "Yes" else "N",
                    "cb_person_cred_hist_length": credit_history_length,
                    "cibil_score": cibil_score
                }
                
                # Process with the financial agent directly
                result = orchestrator.process("", context=context, force_intent="financial_risk")
                
                # Display results
                st.markdown(result.get("formatted_response", ""), unsafe_allow_html=True)
                
                # Show additional details in an expander
                with st.expander("See loan application details"):
                    st.write(f"**Income to Loan Ratio:** {loan_percent:.1f}%")
                    
                    # Only show monthly payment if approved
                    if result.get("recommendation") == "Approved":
                        interest_rate = result.get("interest_rate", 7.5)
                        monthly_payment = (loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12) ** (12*5)) / ((1 + interest_rate/100/12) ** (12*5) - 1)
                        st.write(f"**Monthly Payment (5-year term):** ${monthly_payment:.2f}")
                    
                    # CIBIL score interpretation
                    if cibil_score >= 750:
                        cibil_rating = "Excellent"
                    elif cibil_score >= 700:
                        cibil_rating = "Good"
                    elif cibil_score >= 650:
                        cibil_rating = "Fair"
                    elif cibil_score >= 600:
                        cibil_rating = "Poor"
                    else:
                        cibil_rating = "Very Poor"
                    
                    st.write(f"**CIBIL Score Rating:** {cibil_rating} ({cibil_score})")
                    
                    # Feature importance display
                    feature_importance = dict(zip(
                        orchestrator.financial_agent.feature_names, 
                        orchestrator.financial_agent.model.feature_importances_
                    ))
                    
                    # Sort by importance
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    
                    st.write("**Key factors in decision (in order of importance):**")
                    for feature, importance in sorted_features[:5]:
                        readable_feature = feature.replace("_", " ").replace("person", "applicant").replace("cb", "credit")
                        st.write(f"- {readable_feature.title()}: {importance:.3f}")
    
    else:  # Insurance Claim Processing
        st.markdown("<h3 class='sub-header'>Insurance Claim Assessment</h3>", unsafe_allow_html=True)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            claim_type = st.selectbox(
                "Claim Type", 
                ["Auto Insurance", "Home Insurance", "Health Insurance", "Property Insurance", "Other"]
            )
            incident_date = st.date_input("Incident Date")
            
        with col2:
            policy_number = st.text_input("Policy Number (if available)")
            filing_date = st.date_input("Filing Date")
        
        # Full width for description
        claim_description = st.text_area(
            "Claim Description", 
            height=150,
            placeholder="Please describe your claim in detail. Include what happened, where it happened, and any relevant details about the incident."
        )
        
        # Documentation upload (note: in this demo, we don't actually process uploads)
        st.write("**Supporting Documentation**")
        col1, col2 = st.columns(2)
        
        with col1:
            photos = st.file_uploader("Upload Photos (if any)", accept_multiple_files=True)
            
        with col2:
            documents = st.file_uploader("Upload Documents (if any)", accept_multiple_files=True)
        
        # Submit button
        if st.button("Process Claim"):
            if not claim_description:
                st.error("Please provide a description of your claim.")
            else:
                with st.spinner("Analyzing your insurance claim..."):
                    # Create context with all the info
                    context = {
                        "claim_description": claim_description,
                        "claim_type": claim_type,
                        "incident_date": str(incident_date),
                        "policy_number": policy_number,
                        "filing_date": str(filing_date)
                    }
                    
                    # Process with the insurance agent directly
                    result = orchestrator.process(claim_description, context=context, force_intent="insurance_claim")
                    
                    # Display results
                    st.markdown(result.get("formatted_response", ""), unsafe_allow_html=True)
                    
                    # Show similar claims in an expander
                    with st.expander("See similar claims in our database"):
                        similar_claims = result.get("similar_claims", [])
                        for i, claim in enumerate(similar_claims):
                            st.write(f"**Similar Claim #{i+1}:** {claim.get('description')}")
                            st.write(f"Severity: {claim.get('severity')}")
                            st.write(f"Risk Factors: {', '.join(claim.get('risk_factors', []))}")
                            st.markdown("---")
    
    # Disclaimer at bottom of page
    st.markdown("<p class='disclaimer'>Disclaimer: This tool provides estimates and guidance based on the information provided. The assessments should not be considered as final decisions. For actual loan applications or insurance claims, please contact the appropriate financial or insurance representatives.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()