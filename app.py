
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Customer Churn Analysis Dashboard")
st.caption("Telecom Customer Retention | Python • Pandas • Plotly")

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        data = pd.read_csv(BytesIO(file_bytes))
    else:
        data = pd.read_csv(DATA_PATH)

    data.columns = data.columns.str.strip()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.drop_duplicates(subset=["customerID"])

    data = data[data["Churn"].isin(["Yes", "No"])].copy()

    data["Tenure Group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49+ Months"
        ],
        ordered=True
    )

    return data


uploaded_file = None

if not DATA_PATH.exists():
    uploaded_file = st.sidebar.file_uploader(
        "Upload the Telco Customer Churn CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info(
            "Place the dataset in the data folder or upload it "
            "using the sidebar to continue."
        )
        st.stop()

try:
    df = load_data(
        uploaded_file.getvalue() if uploaded_file else None
    )
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.error("No valid customer records were found.")
    st.stop()


st.sidebar.header("Filter Customers")

def filter_options(column):
    return sorted(df[column].dropna().unique().tolist())

contract_options = filter_options("Contract")
payment_options = filter_options("PaymentMethod")
internet_options = filter_options("InternetService")

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    contract_options,
    default=contract_options
)

selected_payments = st.sidebar.multiselect(
    "Payment Method",
    payment_options,
    default=payment_options
)

selected_internet = st.sidebar.multiselect(
    "Internet Service",
    internet_options,
    default=internet_options
)

senior_options = ["All", "Senior Citizens", "Non-Senior Citizens"]

selected_senior = st.sidebar.selectbox(
    "Customer Age Group",
    senior_options
)

filtered = df[
    df["Contract"].isin(selected_contracts)
    & df["PaymentMethod"].isin(selected_payments)
    & df["InternetService"].isin(selected_internet)
].copy()

if selected_senior == "Senior Citizens":
    filtered = filtered[filtered["SeniorCitizen"] == 1]
elif selected_senior == "Non-Senior Citizens":
    filtered = filtered[filtered["SeniorCitizen"] == 0]


total_customers = filtered["customerID"].nunique()
churned_customers = filtered.loc[
    filtered["Churn"] == "Yes", "customerID"
].nunique()

retained_customers = max(total_customers - churned_customers, 0)

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers else 0
)

avg_monthly_charges = (
    filtered["MonthlyCharges"].mean()
    if not filtered.empty else 0
)

st.subheader("Customer Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Churned Customers", f"{churned_customers:,}")
k3.metric("Churn Rate", f"{churn_rate:.2f}%")
k4.metric("Avg. Monthly Charges", f"${avg_monthly_charges:,.2f}")

st.caption(
    f"Showing {len(filtered):,} customer records after applying filters."
)


def churn_by(column):
    if filtered.empty:
        return pd.DataFrame(
            columns=[column, "Total Customers", "Churned Customers", "Churn Rate"]
        )

    result = (
        filtered.groupby(column, observed=False)
        .agg(
            **{
                "Total Customers": ("customerID", "nunique"),
                "Churned Customers": (
                    "Churn",
                    lambda values: (values == "Yes").sum()
                )
            }
        )
        .reset_index()
    )

    result = result[result["Total Customers"] > 0].copy()

    result["Churn Rate"] = (
        result["Churned Customers"] / result["Total Customers"] * 100
    )

    return result


def make_bar_chart(data, category, title, horizontal=True):
    if data.empty:
        st.info("No data available for this chart with the selected filters.")
        return

    if horizontal:
        fig = px.bar(
            data,
            x="Churn Rate",
            y=category,
            orientation="h",
            text=data["Churn Rate"].map(lambda x: f"{x:.2f}%"),
            hover_data=[
                "Total Customers",
                "Churned Customers"
            ],
            title=title
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            data,
            x=category,
            y="Churn Rate",
            text=data["Churn Rate"].map(lambda x: f"{x:.2f}%"),
            hover_data=[
                "Total Customers",
                "Churned Customers"
            ],
            title=title
        )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="Churn Rate (%)" if not horizontal else "",
        xaxis_title="Churn Rate (%)" if horizontal else "",
        margin=dict(l=10, r=25, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)



st.subheader("Churn Analysis")

left, right = st.columns(2)

with left:
    contract_data = churn_by("Contract")
    make_bar_chart(
        contract_data,
        "Contract",
        "Churn Rate by Contract Type"
    )

with right:
    tenure_data = churn_by("Tenure Group")

    if not tenure_data.empty:
        tenure_order = [
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49+ Months"
        ]
        tenure_data["Tenure Group"] = pd.Categorical(
            tenure_data["Tenure Group"],
            categories=tenure_order,
            ordered=True
        )
        tenure_data = tenure_data.sort_values("Tenure Group")

    make_bar_chart(
        tenure_data,
        "Tenure Group",
        "Churn Rate by Tenure Group",
        horizontal=False
    )

left, right = st.columns(2)

with left:
    payment_data = churn_by("PaymentMethod")
    make_bar_chart(
        payment_data,
        "PaymentMethod",
        "Churn Rate by Payment Method"
    )

with right:
    internet_data = churn_by("InternetService")
    make_bar_chart(
        internet_data,
        "InternetService",
        "Churn Rate by Internet Service"
    )


st.subheader("💡 Key Business Insights")

if filtered.empty:
    st.warning("Select at least one option in each filter to view insights.")
else:
    st.write(
        f"**Overall:** {churn_rate:.2f}% of the currently filtered "
        f"customers have churned."
    )

    for column, label in [
        ("Contract", "contract type"),
        ("Tenure Group", "tenure group"),
        ("PaymentMethod", "payment method"),
        ("InternetService", "internet service")
    ]:
        summary = churn_by(column)

        if not summary.empty:
            highest = summary.loc[summary["Churn Rate"].idxmax()]

            st.write(
                f"- **{label.title()}:** "
                f"{highest[column]} has the highest observed churn rate "
                f"({highest['Churn Rate']:.2f}%) in the current selection."
            )

    st.info(
        "These are observed associations, not proof of causation. "
        "Investigate customer feedback, pricing, and service quality "
        "before choosing retention strategies."
    )


with st.expander("Explore filtered customer data"):
    st.dataframe(filtered, use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_customer_churn.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Customer Churn Analysis | Built with Python, Pandas, Streamlit and Plotly")
