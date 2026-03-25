"""Streamlit dashboard for single-customer churn prediction."""

import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💻",
    layout="wide"
)

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


# Make project root importable when running: streamlit run dashboard/app.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_customer, load_model, preprocess_input


DEFAULT_INPUTS = {
	"gender": "Female",
	"senior_citizen": 0,
	"partner": "No",
	"dependents": "No",
	"tenure": 12,
	"phone_service": "No",
	"multiple_lines": "No",
	"internet_service": "DSL",
	"online_security": "No",
	"online_backup": "No",
	"device_protection": "No",
	"tech_support": "No",
	"streaming_tv": "No",
	"streaming_movies": "No",
	"contract": "Month-to-month",
	"paperless_billing": "No",
	"payment_method": "Electronic check",
	"monthly_charges": 70.0,
	"total_charges": 1200.0,
}


def init_form_state() -> None:
	"""Initialize form session state with defaults once."""
	for key, value in DEFAULT_INPUTS.items():
		if key not in st.session_state:
			st.session_state[key] = value


def reset_form_state() -> None:
	"""Reset all form inputs to default values."""
	for key, value in DEFAULT_INPUTS.items():
		st.session_state[key] = value


def get_top_feature_impacts(input_data: dict, top_n: int = 5) -> tuple[pd.DataFrame, str]:
	"""Return top feature impacts for a single prediction.

	Uses model feature importances weighted by the absolute transformed feature value
	for the current input row to approximate which features most affected the score.
	"""
	model, artifacts = load_model()
	processed_input = preprocess_input(input_data, artifacts)

	if not hasattr(model, "feature_importances_"):
		return pd.DataFrame(), "Feature importance is unavailable for this model type."

	feature_names = list(processed_input.columns)
	importances = np.array(model.feature_importances_, dtype=float)
	row_values = np.abs(processed_input.iloc[0].to_numpy(dtype=float))

	impact_scores = importances * row_values
	if np.allclose(impact_scores, 0):
		impact_scores = importances

	importance_df = pd.DataFrame(
		{
			"Feature": feature_names,
			"Impact": impact_scores,
		}
	)

	importance_df = importance_df.sort_values("Impact", ascending=False).head(top_n)

	return importance_df, "Approximate impact based on model importance and this customer's transformed values."


def get_churn_reasons(input_data: dict, max_reasons: int = 3) -> list[str]:
	"""Generate human-readable churn reasons from customer input signals."""
	reasons = []

	if input_data.get("Contract") == "Month-to-month":
		reasons.append("Month-to-month contract typically has higher churn risk than longer commitments.")

	if int(input_data.get("tenure", 0)) <= 12:
		reasons.append("Low tenure customers are generally more likely to churn in early months.")

	if input_data.get("PaymentMethod") == "Electronic check":
		reasons.append("Electronic check payment method is often associated with higher churn patterns.")

	if float(input_data.get("MonthlyCharges", 0.0)) >= 80:
		reasons.append("Higher monthly charges can increase price sensitivity and churn likelihood.")

	if (
		input_data.get("InternetService") in {"DSL", "Fiber optic"}
		and input_data.get("TechSupport") == "No"
	):
		reasons.append("Customers without tech support are more likely to leave when issues occur.")

	if (
		input_data.get("InternetService") in {"DSL", "Fiber optic"}
		and input_data.get("OnlineSecurity") == "No"
	):
		reasons.append("Lack of online security add-on is linked with increased churn risk.")

	if (
		input_data.get("InternetService") == "Fiber optic"
		and float(input_data.get("MonthlyCharges", 0.0)) >= 70
	):
		reasons.append("Fiber optic customers with higher bills may churn due to cost-value tradeoffs.")

	# Keep unique reasons in insertion order and ensure at least 2 insights.
	unique_reasons = list(dict.fromkeys(reasons))

	if len(unique_reasons) < 2:
		unique_reasons.append("Billing and service profile still shows non-zero churn probability from the model.")
	if len(unique_reasons) < 2:
		unique_reasons.append("Customer lifecycle stage and subscription mix can still contribute to churn behavior.")

	return unique_reasons[:max_reasons]


st.set_page_config(
	page_title="Customer Churn Prediction",
	page_icon="📊",
	layout="wide",
)

st.markdown(
	"""
	<style>
		.block-container {
			padding-top: 2.2rem;
			padding-bottom: 2.4rem;
		}
		.app-header {
			margin-bottom: 0.7rem;
		}
		.app-title {
			margin: 0;
			line-height: 1.2;
			font-size: 2.25rem;
			font-weight: 700;
		}
		.app-subtitle {
			margin: 0.35rem 0 0 0;
			font-size: 1rem;
			color: #4B5563;
		}
		div[data-testid="stForm"] {
			padding-top: 0.35rem;
			padding-bottom: 0.4rem;
		}
		div[data-testid="stForm"] hr {
			margin: 0.45rem 0 0.6rem 0;
		}
		div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
			gap: 0.35rem;
		}
		div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
			margin-top: 0.1rem;
		}
		h3 {
			margin-bottom: 0.25rem;
		}
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
	<div class="app-header">
		<h1 class="app-title">Customer Churn Prediction System</h1>
		<p class="app-subtitle">AI-powered churn risk analyzer for telecom customers</p>
	</div>
	""",
	unsafe_allow_html=True,
)

init_form_state()


with st.form("prediction_form"):
	st.subheader("Customer Info")
	col1, col2 = st.columns(2, gap="small")
	with col1:
		gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
		senior_citizen = st.selectbox("Senior Citizen", [0, 1], key="senior_citizen")
		partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
	with col2:
		dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
		tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, key="tenure")

	st.divider()
	st.subheader("Services")
	col3, col4 = st.columns(2, gap="small")
	with col3:
		phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_service")
		multiple_lines = st.selectbox(
			"Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines"
		)
		internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")
		online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
		online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
	with col4:
		device_protection = st.selectbox(
			"Device Protection", ["No", "Yes", "No internet service"], key="device_protection"
		)
		tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
		streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
		streaming_movies = st.selectbox(
			"Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies"
		)

	st.divider()
	st.subheader("Billing")
	col5, col6 = st.columns(2, gap="small")
	with col5:
		contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
		paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless_billing")
	with col6:
		payment_method = st.selectbox(
			"Payment Method",
			[
				"Electronic check",
				"Mailed check",
				"Bank transfer (automatic)",
				"Credit card (automatic)",
			],
			key="payment_method",
		)
		monthly_charges = st.number_input(
			"Monthly Charges", min_value=0.0, max_value=200.0, step=0.01, key="monthly_charges"
		)
		total_charges = st.number_input(
			"Total Charges", min_value=0.0, step=0.01, key="total_charges"
		)

	button_container = st.container()
	with button_container:
		btn_col1, btn_col2 = st.columns([1, 1], gap="small")
		with btn_col1:
			submit = st.form_submit_button("Predict Churn", use_container_width=True)
		with btn_col2:
			reset = st.form_submit_button("Reset Inputs", use_container_width=True)

if reset:
	reset_form_state()
	st.rerun()


if submit:
	input_data = {
		"gender": gender,
		"SeniorCitizen": int(senior_citizen),
		"Partner": partner,
		"Dependents": dependents,
		"tenure": int(tenure),
		"PhoneService": phone_service,
		"MultipleLines": multiple_lines,
		"InternetService": internet_service,
		"OnlineSecurity": online_security,
		"OnlineBackup": online_backup,
		"DeviceProtection": device_protection,
		"TechSupport": tech_support,
		"StreamingTV": streaming_tv,
		"StreamingMovies": streaming_movies,
		"Contract": contract,
		"PaperlessBilling": paperless_billing,
		"PaymentMethod": payment_method,
		"MonthlyCharges": float(monthly_charges),
		"TotalCharges": float(total_charges),
	}

	try:
		result = predict_customer(input_data)
		churn_probability = float(result["probability"])
		risk_level = result['risk_level']
		top_features_df, impact_note = get_top_feature_impacts(input_data, top_n=5)
		reasons = get_churn_reasons(input_data, max_reasons=3)

		# Determine colors and emoji based on risk level
		if risk_level == "High":
			emoji = "⚠️"
			heading = "High Churn Risk"
			color = "#DC2626"  # Red
			bg_color = "#FEE2E2"
			text_color = "#991B1B"
		elif risk_level == "Medium":
			emoji = "⏸️"
			heading = "Medium Churn Risk"
			color = "#F59E0B"  # Amber/Yellow
			bg_color = "#FEF3C7"
			text_color = "#92400E"
		else:  # Low
			emoji = "✅"
			heading = "Low Churn Risk"
			color = "#10B981"  # Green
			bg_color = "#DCFCE7"
			text_color = "#065F46"

		result_col1, result_col2 = st.columns([1, 1], gap="small")

		with result_col1:
			# Display big heading and probability summary
			st.markdown(f"<h1 style='margin: 0; color: {color};'>{emoji} {heading}</h1>", unsafe_allow_html=True)
			percentage = churn_probability * 100
			st.markdown(f"<h2 style='margin: 0.15rem 0 0.5rem 0; color: {text_color};'>{percentage:.1f}%</h2>", unsafe_allow_html=True)
			st.progress(churn_probability)

		with result_col2:
			# Display detailed result box
			st.markdown(
				f"""
				<div style="
					border-radius: 12px;
					padding: 16px;
					border: 2px solid {color};
					background-color: {bg_color};
					margin-top: 4px;
				">
					<p style="margin: 0; font-size: 16px; color: {text_color};">
						<strong>Risk Level:</strong> {risk_level}
					</p>
					<p style="margin: 6px 0 0 0; font-size: 14px; color: {text_color};">
						<strong>Churn Probability:</strong> {churn_probability:.2%}
					</p>
				</div>
				""",
				unsafe_allow_html=True,
			)

			# Display contextual message
			if risk_level == "High":
				st.error("🚨 High churn risk detected. Consider a retention action plan immediately.")
			elif risk_level == "Medium":
				st.warning("⚠️ Medium churn risk detected. Monitor this customer closely.")
			else:
				st.success("✅ Low churn risk detected. Customer appears stable.")

		st.divider()
		analysis_col1, analysis_col2 = st.columns([1, 1], gap="medium")

		with analysis_col1:
			st.subheader("Top 5 Important Features")
			if not top_features_df.empty:
				top_features_df = top_features_df.copy()
				total_impact = float(top_features_df["Impact"].sum())
				if total_impact > 0:
					top_features_df["ImpactPercent"] = (top_features_df["Impact"] / total_impact) * 100
				else:
					top_features_df["ImpactPercent"] = 0.0

				chart_data = top_features_df.set_index("Feature")["ImpactPercent"]
				st.bar_chart(chart_data)

				for _, row in top_features_df.iterrows():
					st.write(f"- {row['Feature']}: {row['ImpactPercent']:.1f}% influence")

				st.caption(impact_note)
			else:
				st.info("Top feature importance is not available for the current model.")

		with analysis_col2:
			st.subheader("Why This Customer Might Churn")
			for reason in reasons:
				st.write(f"- {reason}")

	except FileNotFoundError:
		st.error("Model file not found. Train the model first from src/train_model.py.")
	except Exception as exc:
		st.error(f"Prediction failed: {exc}")
