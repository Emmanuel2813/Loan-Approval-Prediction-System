import streamlit as st
import pandas as pd
import joblib
import dice_ml
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CATEGORICAL MAPPINGS (From app.py)
# -------------------------------------------------
age_map = {"None": 0,"<25": 1,"25-34": 2,"35-44": 3,"45-54": 4,"55-64": 5,"65-74": 6,">74": 7}
loan_limit_map =  {'cf':2, 'None':0, 'ncf':1} 
approv_in_adv_map = {'nopre': 1, 'pre': 2, 'None':0} 
Neg_ammortization_map = {'not_neg':1, 'neg_amm':2, 'None':0} 
submission_of_application_map = {'to_inst':2, 'not_inst':1, 'None':0} 
total_units_map = {'1U':1, '2U':2, '3U':3, '4U':4} 
yes_no_map = {"Yes":1, "No":0}
construction_type_map = {'sb':0, 'mh':1}
secured_by_map = {'home':0, 'land':1}
co_applicant_credit_type_map = {'EXP':1, 'CIB':0}
Security_Type_map = {'direct':0, 'Indirect':1}

# -------------------------------------------------
# PREMIUM RESPONSIVE CSS & THEME ADAPTATION
# -------------------------------------------------
st.markdown("""
<style>
/* Base container: dynamically matches the user's Streamlit Theme (Light/Dark) */
.main {
    background-color: var(--background-color); 
    color: var(--text-color);
    transition: all 0.3s ease-in-out;
}

/* Premium Button Styling */
.stButton>button {
    width: 100%;
    background-color: #0a66c2;
    color: white;
    height: 3.5em;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.05rem;
    border: none;
    box-shadow: 0 4px 6px rgba(10, 102, 194, 0.2);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.stButton>button:hover {
    background-color: #004182;
    box-shadow: 0 7px 14px rgba(10, 102, 194, 0.4);
    transform: translateY(-2px);
    color: #ffffff;
}

.stButton>button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 4px rgba(10, 102, 194, 0.2);
}

/* Smooth Form Wrappers & Input Fields */
div[data-testid="stForm"] {
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(128, 128, 128, 0.2);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    background-color: transparent;
    transition: border 0.3s ease;
}

div[data-testid="stForm"]:hover {
    border: 1px solid rgba(10, 102, 194, 0.3);
}

/* Glassmorphic Callout Banner matching any theme */
.premium-callout {
    background: rgba(10, 102, 194, 0.08);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 20px 25px;
    border-radius: 12px;
    border-left: 6px solid #0a66c2;
    margin-bottom: 2rem;
    font-size: 1.1rem;
    color: var(--text-color);
    transition: all 0.3s ease;
}

/* Mobile Responsiveness Auto-Padding */
@media (max-width: 768px) {
    div[data-testid="stForm"] {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL (ONLY ONCE)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🏦 Smart Loan Approval Predictor")
st.markdown("""
<div class='premium-callout'>
✨ <b>Welcome!</b> This AI tool evaluates your comprehensive financial profile to predict your loan approval probability.<br>
<i>Please complete the details below carefully for an accurate forecast.</i>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FORM START
# -------------------------------------------------
with st.form("loan_form"):

    st.subheader("📌 Section 1: Personal & Financial Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Monthly Income", 0.00, 1_000_000_000.00, 100000.00, step=10000.00)
        age = st.selectbox("Select Age Group", ["None", "<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"])
        age = age_map[age]
        credit_score = st.number_input("Credit Score", 0.00, 1_000_000_000.00, 650.00, step=50.00)
        credit_worthiness = st.selectbox("Willing/Able to repay debt? (Credit Worthiness)", ["Yes", "No"])
        credit_worthiness = yes_no_map[credit_worthiness]

    with col2:
        gender = st.selectbox("What is your Gender?", ["Male", "Female", "Joint", "Sex Not Available"])
        Gender_Male = True if gender == "Male" else False
        Gender_Female = True if gender == "Female" else False
        Gender_Joint = True if gender == "Joint" else False
        Gender_Sex_Not_Available = True if gender == "Sex Not Available" else False
        
        region = st.selectbox("Select Region", ["North", "South", "North-East", "central"])
        Region_North = True if region == "North" else False
        Region_south = True if region == "South" else False
        Region_North_East = True if region == "North-East" else False
        Region_central = True if region == "central" else False

        credit_type = st.selectbox("Credit Bureau Type", ["EXP", "EQUI", "CRIF", "CIB"])
        credit_type_EXP = True if credit_type == "EXP" else False
        credit_type_EQUI = True if credit_type == "EQUI" else False
        credit_type_CRIF = True if credit_type == "CRIF" else False
        credit_type_CIB = True if credit_type == "CIB" else False

    with col3:
        co_applicant_credit_type = st.selectbox("Co-applicant Credit Bureau", ["EXP", "CIB"])
        co_applicant_credit_type = co_applicant_credit_type_map[co_applicant_credit_type]
        open_credit = st.selectbox("Existing Open Credit?", ["Yes", "No"])
        open_credit = yes_no_map[open_credit]
        business_use = st.selectbox("Business/Commercial Purpose?", ["Yes", "No"])
        business_use = yes_no_map[business_use]


    st.markdown("---")
    st.subheader("📊 Section 2: Loan Setup")
    col4, col5, col6 = st.columns(3)

    with col4:
        loan_amount = st.number_input("Loan Amount", 1000.00, 1_000_000_000.00, 100000.00, step=10000.00)
        term = st.number_input("Loan Term (Days)", 1, 365000, 360, step=30)
        rate_of_interest = st.number_input("Interest Rate (%)", 0.0, 100.0, 5.0, step=0.1)
        upfront_charges = st.number_input("Upfront Charges", 0.00, loan_amount, 0.00, step=1000.00)

    with col5:
        loan_limit = st.selectbox("Loan Limit Type", ["cf", "ncf", "None"])
        loan_limit = loan_limit_map[loan_limit]
        approv_in_adv = st.selectbox("Pre-Approval Status", ["nopre", "pre", "None"])
        approv_in_adv = approv_in_adv_map[approv_in_adv]
        
        loan_type = st.selectbox("Loan Type (1=Fixed, 2=Variable, 3=Adjustable)", ["type1", "type2", "type3"])
        loan_type_type1 = True if loan_type == "type1" else False
        loan_type_type2 = True if loan_type == "type2" else False
        loan_type_type3 = True if loan_type == "type3" else False

    with col6:
        loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4", "None"])
        loan_purpose_p1 = True if loan_purpose == "p1" else False
        loan_purpose_p2 = True if loan_purpose == "p2" else False
        loan_purpose_p3 = True if loan_purpose == "p3" else False
        loan_purpose_p4 = True if loan_purpose == "p4" else False
        loan_purpose_None = True if loan_purpose == "None" else False
        
        interest_only = st.selectbox("Interest Only Payment?", ["Yes", "No"])
        interest_only = yes_no_map[interest_only]
        lump_sum_payment = st.selectbox("Lump Sum Payment Intended?", ["Yes", "No"])
        lump_sum_payment = yes_no_map[lump_sum_payment]
        neg_ammortization = st.selectbox("Negative Amortization?", ["not_neg", "neg_amm", "None"])
        neg_ammortization = Neg_ammortization_map[neg_ammortization]


    st.markdown("---")
    st.subheader("🏡 Section 3: Property Details")
    col7, col8 = st.columns(2)

    with col7:
        property_value = st.number_input("Property Value", 1000.00, 1_000_000_000.00, 200000.00, step=10000.00)
        construction_type = st.selectbox("Construction Type (sb=Storey, mh=Multi-Housing)", ["sb", "mh"])
        construction_type = construction_type_map[construction_type]
        secured_by = st.selectbox("Secured By", ["home", "land"])
        secured_by = secured_by_map[secured_by]
        total_units = st.selectbox("Total Housing Units", ["1", "2", "3", "4"])
        total_units = int(total_units)

    with col8:
        occupancy_type = st.selectbox("Occupancy Type (pr=Primary, sr=Secondary, ir=Investment)", ["pr", "sr", "ir"])
        occupancy_type_pr = True if occupancy_type == "pr" else False
        occupancy_type_sr = True if occupancy_type == "sr" else False
        occupancy_type_ir = True if occupancy_type == "ir" else False
        
        security_type = st.selectbox("Security Type", ["direct", "Indirect"])
        security_type = Security_Type_map[security_type]
        submission = st.selectbox("Submission of Application", ["to_inst", "not_inst", "None"])
        submission = submission_of_application_map[submission]

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if submitted:

    with st.spinner("Analyzing financial profile..."):
        time.sleep(0.5)

        # Feature Engineering (Matching app.py exactly)
        LTV = (loan_amount / property_value) * 100 if property_value > 0 else 0
        Interest_rate_spread = (rate_of_interest / 100) - (term / 100)

        def calculate_monthly_payment(loan_amount, annual_rate, term_months):
            if annual_rate == 0:
                return loan_amount / term_months
            monthly_rate = annual_rate / 100 / 12
            factor = (1 + monthly_rate) ** term_months
            payment = loan_amount * monthly_rate * factor / (factor - 1)
            return payment

        monthly_payment = calculate_monthly_payment(loan_amount, rate_of_interest, term)

        if income > 0 and monthly_payment > income:
            dtir1 = (monthly_payment / income) * 100
        else:
            dtir1 = 0

        # DataFrame matching app.py precisely
        user_input_df = pd.DataFrame({
            'loan_limit': [loan_limit],
            'approv_in_adv': [approv_in_adv],
            'Credit_Worthiness': [credit_worthiness],
            'open_credit': [open_credit],
            'business_or_commercial': [business_use],
            'loan_amount': [loan_amount],
            'rate_of_interest': [rate_of_interest],
            'Interest_rate_spread': [Interest_rate_spread],
            'Upfront_charges': [upfront_charges],
            'term': [term],
            'Neg_ammortization': [neg_ammortization],
            'interest_only': [interest_only],
            'lump_sum_payment': [lump_sum_payment],
            'property_value': [property_value],
            'construction_type': [construction_type],
            'Secured_by': [secured_by],
            'total_units': [total_units],
            'income': [income],
            'Credit_Score': [credit_score],
            'co_applicant_credit_type': [co_applicant_credit_type],
            'age': [age],
            'submission_of_application': [submission],
            'LTV': [LTV],
            'Security_Type': [security_type],
            'dtir1': [dtir1],
            'Gender_Female': [Gender_Female],
            'Gender_Joint': [Gender_Joint],
            'Gender_Male': [Gender_Male],
            'Gender_Sex_Not_Available': [Gender_Sex_Not_Available],
            'loan_type_type1': [loan_type_type1],
            'loan_type_type2': [loan_type_type2],
            'loan_type_type3': [loan_type_type3],
            'loan_purpose_None': [loan_purpose_None],
            'loan_purpose_p1': [loan_purpose_p1],
            'loan_purpose_p2': [loan_purpose_p2],
            'loan_purpose_p3': [loan_purpose_p3],
            'loan_purpose_p4': [loan_purpose_p4],
            'occupancy_type_ir': [occupancy_type_ir],
            'occupancy_type_pr': [occupancy_type_pr],
            'occupancy_type_sr': [occupancy_type_sr],
            'credit_type_CIB': [credit_type_CIB],
            'credit_type_CRIF': [credit_type_CRIF],
            'credit_type_EQUI': [credit_type_EQUI],
            'credit_type_EXP': [credit_type_EXP],
            'Region_North': [Region_North],
            'Region_North_East': [Region_North_East],
            'Region_central': [Region_central],
            'Region_south': [Region_south]
        })

        try:
            # Reindex and typecast identically to app.py
            model_columns = joblib.load("model_columns.pkl")
            model_dtypes = joblib.load("model_dtypes.pkl")
            user_input_df = user_input_df.reindex(columns=model_columns, fill_value=0)

            for col, dtype in model_dtypes.items():
                user_input_df[col] = user_input_df[col].astype(dtype)

            prediction = model.predict(user_input_df)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(user_input_df)[0][1] * 100
            else:
                probability = 100 if prediction[0] == 1 else 0

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # -------------------------------------------------
        # RESULTS DASHBOARD
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Loan Approval Analysis")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if prediction[0] == 1:
                st.metric("Loan Status", "Approved ✅")
            else:
                st.metric("Loan Status", "Rejected ❌")

        with col_res2:
            st.write(f"### Confidence: {probability:.1f}%")
            st.progress(int(probability))

            if prediction[0] == 1:
                st.success("Your loan is likely to be approved.")
            else:
                st.error("Your loan is likely to be rejected.")

        # -------------------------------------------------
        # COUNTERFACTUAL SECTION (DiCE ML EXACTLY LIKE APP.PY)
        # -------------------------------------------------
        if prediction[0] == 0:
            with st.expander("💡 See How To Improve Approval Chances"):
                with st.spinner("Generating suggestions to improve your application... this might take a minute."):
                    try:
                        df = pd.read_csv("Preprocessed_Loan_dataset.csv", index_col=0)
                        df.drop(['ID', 'year'], axis = 1, inplace = True)

                        # Enforce column dtypes strictly like app.py
                        for col, dtype in model_dtypes.items():
                            df[col] = df[col].astype(dtype)

                        # Explicit Boolean safeguard against DiCE Model Crashing
                        for col in df.columns:
                            if df[col].dtype == bool:
                                df[col] = df[col].astype(int)
                        for col in user_input_df.columns:
                            if user_input_df[col].dtype == bool:
                                user_input_df[col] = user_input_df[col].astype(int)

                        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
                        d = dice_ml.Data(
                            dataframe=df,
                            continuous_features=["loan_amount", "rate_of_interest", "Interest_rate_spread", "Upfront_charges", "term", "property_value", "income", "Credit_Score", "LTV", "dtir1"],
                            outcome_name="Status"
                        )
                        exp = dice_ml.Dice(d, m)

                        e1 = exp.generate_counterfactuals(user_input_df, total_CFs=5, desired_class="opposite")
                        st.success("Counterfactuals generated successfully!")
                        
                        st.write("### Your Original Profile")
                        st.caption("These were the details you submitted.")
                        st.dataframe(user_input_df)

                        st.write("### Recommended Alternative Profiles")
                        st.caption("Any row below represents a profile that would have been APPROVED. Focus on changing the fields that differ from your original.")
                        cf_df = e1.cf_examples_list[0].final_cfs_df
                        st.dataframe(cf_df)

                    except Exception as e:
                        st.error(f"Could not calculate exact differences. Error: {e}")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.info("ℹ️ About This App")
    st.markdown("""
    This AI tool uses a trained Machine Learning model 
    to predict loan approval likelihood.
    
    • No data is stored  
    • For educational purposes only  
    """)
    st.write("---")
    st.caption("Loan AI v2.0")
