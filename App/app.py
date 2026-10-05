import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UAC Care Load Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL PASTEL COLOR PALETTE
# ============================================================

NAVY = "#29243C"
DARK_PURPLE = "#514568"
PURPLE = "#786A91"

LAVENDER = "#A89CC8"
LIGHT_LAVENDER = "#DDD6E9"

ROSE = "#D79AB4"
LIGHT_ROSE = "#F0D4DF"

SAGE = "#9FBEAA"
LIGHT_SAGE = "#DCE9DF"

PEACH = "#E7B39C"
CREAM = "#F7F5F9"
WHITE = "#FFFFFF"

TEXT = "#302C3B"
MUTED = "#6D6877"
BORDER = "#E2DDE8"


# ============================================================
# GLOBAL CSS
#
# IMPORTANT:
# There is NO visible HTML content in the dashboard.
# This CSS is only used for styling Streamlit's native widgets.
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background-color: #F7F5F9;
    color: #302C3B;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #29243C !important;
}

p, label, span {
    color: #302C3B;
}

section[data-testid="stSidebar"] {
    background-color: #29243C;
}

section[data-testid="stSidebar"] * {
    color: #F8F6FA !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #F8F6FA !important;
}

section[data-testid="stSidebar"] .stSelectbox label {
    color: #F8F6FA !important;
}

section[data-testid="stSidebar"] .stDateInput label {
    color: #F8F6FA !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2DDE8;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(41, 36, 60, 0.07);
}

div[data-testid="stMetricLabel"] {
    color: #6D6877 !important;
}

div[data-testid="stMetricValue"] {
    color: #29243C !important;
    font-weight: 800;
}

div[data-testid="stMetricDelta"] {
    color: #786A91 !important;
}

.stButton button {
    background-color: #514568;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

.stButton button:hover {
    background-color: #29243C;
}

.stDownloadButton button {
    background-color: #DDD6E9;
    color: #29243C !important;
    border: 1px solid #A89CC8;
    border-radius: 10px;
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #E2DDE8;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: #E2DDE8;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "Data/HHS_Unaccompanied_Alien_Children_Program.csv"

try:
    df = pd.read_csv(DATA_PATH)

except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure your project structure is:\n\n"
        "UAC_Care_Load_Analytics/\n"
        "├── Data/\n"
        "│   └── UAC_dataset.csv\n"
        "└── App/\n"
        "    └── app.py"
    )
    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Children in HHS Care"] = (
    df["Children in HHS Care"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Children in HHS Care"] = pd.to_numeric(
    df["Children in HHS Care"],
    errors="coerce"
)

df = df.dropna(
    how="all"
).reset_index(drop=True)

df = (
    df.dropna(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Total_System_Load"] = (
    df["Children in CBP custody"]
    + df["Children in HHS Care"]
)

df["Net_Daily_Intake"] = (
    df["Children transferred out of CBP custody"]
    - df["Children discharged from HHS Care"]
)

df["Daily_Load_Change"] = (
    df["Total_System_Load"].diff()
)

df["Care_Load_Growth_Rate"] = (
    df["Total_System_Load"]
    .pct_change()
    * 100
)

df["Positive_Net_Intake"] = (
    df["Net_Daily_Intake"] > 0
)

df["Positive_Intake_Streak"] = (
    df["Positive_Net_Intake"]
    .astype(int)
    .groupby(
        (~df["Positive_Net_Intake"]).cumsum()
    )
    .cumsum()
)

df["Backlog_Flag"] = (
    df["Positive_Intake_Streak"] >= 7
)

df["CBP_Transfer_Anomaly"] = (
    df["Children transferred out of CBP custody"]
    > df["Children in CBP custody"]
)

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Year_Month"] = (
    df["Date"]
    .dt.to_period("M")
    .astype(str)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 📊")

    st.title("UAC Analytics")

    st.caption(
        "Care Load Intelligence Dashboard"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select dashboard section",
        [
            "Overview",
            "System Load",
            "Care Flow",
            "Backlog & Pressure",
            "Data Quality",
            "Forecast"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Analysis Period")

    minimum_date = df["Date"].min().date()
    maximum_date = df["Date"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(
            minimum_date,
            maximum_date
        ),
        min_value=minimum_date,
        max_value=maximum_date
    )

    st.divider()

    st.caption(
        f"Dataset observations: {len(df):,}"
    )

    st.caption(
        f"Period: {minimum_date} → {maximum_date}"
    )

    st.divider()

    st.caption(
        "UAC System Capacity & Care Load Analytics"
    )


# ============================================================
# DATE FILTER
# ============================================================

if (
    isinstance(selected_dates, tuple)
    and len(selected_dates) == 2
):

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = pd.Timestamp(
        selected_dates[1]
    )

    filtered_df = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

else:

    filtered_df = df.copy()


# ============================================================
# HEADER
# ============================================================

st.title(
    "UAC System Capacity & Care Load Analytics"
)

st.write(
    "A data-driven analytical dashboard for monitoring "
    "system load, care flow, capacity pressure and "
    "forecasting for Unaccompanied Children."
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📌 System Overview")

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    average_system_load = (
        filtered_df["Total_System_Load"]
        .mean()
    )

    average_net_intake = (
        filtered_df["Net_Daily_Intake"]
        .mean()
    )

    load_volatility = (
        filtered_df["Daily_Load_Change"]
        .std()
    )

    backlog_rate = (
        filtered_df["Backlog_Flag"]
        .mean()
        * 100
    )

    total_transfers = (
        filtered_df[
            "Children transferred out of CBP custody"
        ].sum()
    )

    total_discharges = (
        filtered_df[
            "Children discharged from HHS Care"
        ].sum()
    )

    if total_transfers > 0:

        discharge_offset_ratio = (
            total_discharges
            / total_transfers
            * 100
        )

    else:

        discharge_offset_ratio = 0


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "👥 Average System Load",
        f"{average_system_load:,.0f}"
    )

    col2.metric(
        "📥 Average Net Intake",
        f"{average_net_intake:,.1f}"
    )

    col3.metric(
        "📊 Load Volatility",
        f"{load_volatility:,.1f}"
    )

    col4.metric(
        "🚨 Backlog Rate",
        f"{backlog_rate:.1f}%"
    )

    col5.metric(
        "📤 Discharge Offset",
        f"{discharge_offset_ratio:.1f}%"
    )


    st.write("")


    # --------------------------------------------------------
    # KEY AREAS
    # --------------------------------------------------------

    st.subheader("Key Analytical Areas")

    card1, card2, card3 = st.columns(3)

    with card1:

        st.info(
            "🏥 **HHS Care**\n\n"
            "The HHS care population represents the "
            "major downstream component of observed "
            "system load and is an important "
            "capacity-planning indicator."
        )

    with card2:

        st.info(
            "🚔 **CBP Custody**\n\n"
            "CBP custody represents an upstream stage "
            "of the care pipeline before children are "
            "transferred into HHS care."
        )

    with card3:

        st.info(
            "🔄 **Care Flow**\n\n"
            "Transfers increase downstream pressure "
            "while HHS discharges reduce active "
            "care-system load."
        )


    # --------------------------------------------------------
    # TOTAL SYSTEM LOAD
    # --------------------------------------------------------

    st.subheader("📈 Total System Load")

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["Total_System_Load"],
        color=PURPLE,
        linewidth=2.5,
        label="Total System Load"
    )

    rolling_average = (
        filtered_df["Total_System_Load"]
        .rolling(
            window=7,
            min_periods=3
        )
        .mean()
    )

    ax.plot(
        filtered_df["Date"],
        rolling_average,
        color=ROSE,
        linewidth=2,
        linestyle="--",
        label="7-observation Average"
    )

    ax.set_title(
        "Total Children Under System Care",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel(
        "Date",
        color=TEXT
    )

    ax.set_ylabel(
        "Children",
        color=TEXT
    )

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # YEARLY LOAD
    # --------------------------------------------------------

    st.subheader("📅 Average System Load by Year")

    yearly_load = (
        filtered_df
        .groupby("Year")["Total_System_Load"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.bar(
        yearly_load["Year"].astype(str),
        yearly_load["Total_System_Load"],
        color=LAVENDER,
        width=0.55
    )

    ax.set_title(
        "Average System Load by Year",
        fontsize=16,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Average Children"
    )

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# SYSTEM LOAD
# ============================================================

elif page == "System Load":

    st.header("📈 System Load Analysis")

    # --------------------------------------------------------
    # CBP VS HHS
    # --------------------------------------------------------

    st.subheader("CBP Custody vs HHS Care")

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["Children in CBP custody"],
        color=LAVENDER,
        linewidth=2.3,
        label="CBP Custody"
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["Children in HHS Care"],
        color=PURPLE,
        linewidth=2.7,
        label="HHS Care"
    )

    ax.set_title(
        "CBP Custody vs HHS Care",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Children")

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # TWO ANALYSES
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "System Load Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.hist(
            filtered_df[
                "Total_System_Load"
            ].dropna(),
            bins=25,
            color=LIGHT_ROSE,
            edgecolor=WHITE
        )

        ax.set_xlabel(
            "Total System Load"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.18
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with right:

        st.subheader(
            "System Load Variability"
        )

        values = (
            filtered_df[
                "Total_System_Load"
            ]
            .dropna()
            .astype(float)
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        box = ax.boxplot(
            values,
            patch_artist=True
        )

        for patch in box["boxes"]:

            patch.set_facecolor(
                LIGHT_LAVENDER
            )

        for median in box["medians"]:

            median.set_color(
                NAVY
            )

            median.set_linewidth(2)

        ax.set_ylabel(
            "Total System Load"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.18
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# CARE FLOW
# ============================================================

elif page == "Care Flow":

    st.header("🔄 Care Flow Analysis")

    # --------------------------------------------------------
    # TRANSFERS VS DISCHARGES
    # --------------------------------------------------------

    st.subheader(
        "Transfers vs HHS Discharges"
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df[
            "Children transferred out of CBP custody"
        ],
        color=LAVENDER,
        linewidth=2.3,
        label="Transfers to HHS"
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df[
            "Children discharged from HHS Care"
        ],
        color=SAGE,
        linewidth=2.3,
        label="HHS Discharges"
    )

    ax.set_title(
        "Transfers vs HHS Discharges",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Children")

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # NET INTAKE
    # --------------------------------------------------------

    st.subheader(
        "📥 Net Daily Intake Pressure"
    )

    net_values = (
        filtered_df[
            "Net_Daily_Intake"
        ]
        .fillna(0)
        .to_numpy()
    )

    net_dates = (
        filtered_df["Date"]
        .to_numpy()
    )

    bar_colors = np.where(
        net_values >= 0,
        ROSE,
        SAGE
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.bar(
        net_dates,
        net_values,
        color=bar_colors,
        width=1
    )

    ax.axhline(
        0,
        color=NAVY,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Net Daily Intake Pressure",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Transfers − Discharges"
    )

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # MONTHLY NET INTAKE
    # --------------------------------------------------------

    st.subheader(
        "📅 Monthly Average Net Intake"
    )

    monthly = (
        filtered_df
        .groupby("Year_Month")[
            "Net_Daily_Intake"
        ]
        .mean()
        .reset_index()
    )

    monthly_values = (
        monthly["Net_Daily_Intake"]
        .to_numpy()
    )

    monthly_colors = np.where(
        monthly_values >= 0,
        ROSE,
        SAGE
    )

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    ax.bar(
        monthly["Year_Month"],
        monthly_values,
        color=monthly_colors
    )

    ax.axhline(
        0,
        color=NAVY,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Monthly Average Net Intake",
        fontsize=16,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Month")

    ax.set_ylabel(
        "Average Net Intake"
    )

    ax.tick_params(
        colors=MUTED,
        axis="x",
        rotation=45
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# BACKLOG & PRESSURE
# ============================================================

elif page == "Backlog & Pressure":

    st.header(
        "🚨 Backlog & Capacity Pressure"
    )

    positive_days = (
        filtered_df[
            "Positive_Net_Intake"
        ]
        .sum()
    )

    maximum_streak = (
        filtered_df[
            "Positive_Intake_Streak"
        ]
        .max()
    )

    backlog_days = (
        filtered_df[
            "Backlog_Flag"
        ]
        .sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Positive Intake Observations",
        f"{positive_days:,.0f}"
    )

    col2.metric(
        "Longest Positive Streak",
        f"{maximum_streak:,.0f}"
    )

    col3.metric(
        "Backlog-Flagged Observations",
        f"{backlog_days:,.0f}"
    )


    st.subheader(
        "Positive Net Intake Streak"
    )

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df[
            "Positive_Intake_Streak"
        ],
        color=ROSE,
        linewidth=2.5
    )

    ax.axhline(
        7,
        color=PEACH,
        linestyle="--",
        linewidth=2,
        label="7-observation threshold"
    )

    ax.set_title(
        "Consecutive Positive Net Intake",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Consecutive Positive Observations"
    )

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    if maximum_streak >= 7:

        st.warning(
            "Sustained intake pressure detected. "
            "At least one sequence of seven or more "
            "consecutive positive-net-intake observations "
            "was identified. This is an analytical indicator, "
            "not an official government threshold."
        )

    else:

        st.success(
            "No seven-observation positive-intake streak "
            "was detected in the selected period."
        )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.header(
        "⚠️ Data Quality & Anomaly Monitoring"
    )

    anomaly_count = (
        filtered_df[
            "CBP_Transfer_Anomaly"
        ]
        .sum()
    )

    date_difference = (
        filtered_df["Date"]
        .diff()
        .dt.days
    )

    reporting_gaps = (
        date_difference > 1
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CBP Transfer Anomalies",
        f"{anomaly_count:,.0f}"
    )

    col2.metric(
        "Reporting Gaps",
        f"{reporting_gaps:,.0f}"
    )

    col3.metric(
        "Observed Records",
        f"{len(filtered_df):,.0f}"
    )


    st.info(
        "A transfer is a flow while CBP custody is a "
        "point-in-time count. Therefore, a transfer value "
        "greater than same-day custody does not automatically "
        "prove that the record is incorrect. Such observations "
        "are retained and flagged for review."
    )


    anomaly_df = filtered_df[
        filtered_df[
            "CBP_Transfer_Anomaly"
        ]
    ].copy()


    if len(anomaly_df) > 0:

        st.subheader(
            "Flagged CBP Transfer Anomalies"
        )

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.scatter(
            anomaly_df["Date"],
            anomaly_df[
                "Children transferred out of CBP custody"
            ],
            color=ROSE,
            s=45
        )

        ax.set_title(
            "Flagged CBP Transfer Anomalies",
            fontsize=17,
            fontweight="bold",
            color=NAVY
        )

        ax.set_xlabel("Date")

        ax.set_ylabel(
            "Children Transferred"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.18
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        st.subheader(
            "🔎 Flagged Records"
        )

        display_columns = [
            "Date",
            "Children in CBP custody",
            "Children transferred out of CBP custody"
        ]

        st.dataframe(
            anomaly_df[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No CBP transfer anomalies were found "
            "in the selected period."
        )


# ============================================================
# FORECAST
# ============================================================

elif page == "Forecast":

    st.header(
        "🔮 System Load Forecast"
    )

    st.info(
        "Forecasting uses the 720 observed reporting records. "
        "Because the source dataset contains missing calendar "
        "dates, the forecast represents future reporting "
        "observations rather than guaranteed consecutive "
        "calendar days."
    )


    # --------------------------------------------------------
    # FORECAST DATA
    # --------------------------------------------------------

    forecast_df = df[
        [
            "Date",
            "Children in CBP custody",
            "Children in HHS Care"
        ]
    ].copy()

    forecast_df["Total_System_Load"] = (
        forecast_df[
            "Children in CBP custody"
        ]
        + forecast_df[
            "Children in HHS Care"
        ]
    )

    forecast_df = (
        forecast_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    series = (
        forecast_df[
            "Total_System_Load"
        ]
        .astype(float)
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    split_index = int(
        len(series) * 0.80
    )

    train_series = (
        series.iloc[:split_index]
    )

    test_series = (
        series.iloc[split_index:]
    )


    # --------------------------------------------------------
    # NAIVE MODEL
    # --------------------------------------------------------

    naive_predictions = np.repeat(
        train_series.iloc[-1],
        len(test_series)
    )


    # --------------------------------------------------------
    # EXPONENTIAL SMOOTHING
    # --------------------------------------------------------

    es_model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )

    es_fit = es_model.fit(
        optimized=True
    )

    es_predictions = np.asarray(
        es_fit.forecast(
            len(test_series)
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # ARIMA
    # --------------------------------------------------------

    arima_model = ARIMA(
        train_series,
        order=(1, 1, 1)
    )

    arima_fit = arima_model.fit()

    arima_predictions = np.asarray(
        arima_fit.forecast(
            steps=len(test_series)
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    actual = (
        test_series
        .astype(float)
        .to_numpy()
    )

    naive_mae = mean_absolute_error(
        actual,
        naive_predictions
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            actual,
            naive_predictions
        )
    )

    es_mae = mean_absolute_error(
        actual,
        es_predictions
    )

    es_rmse = np.sqrt(
        mean_squared_error(
            actual,
            es_predictions
        )
    )

    arima_mae = mean_absolute_error(
        actual,
        arima_predictions
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            actual,
            arima_predictions
        )
    )


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "Naive Baseline",
                "Exponential Smoothing",
                "ARIMA"
            ],
            "MAE": [
                naive_mae,
                es_mae,
                arima_mae
            ],
            "RMSE": [
                naive_rmse,
                es_rmse,
                arima_rmse
            ]
        }
    )

    comparison["MAE"] = (
        comparison["MAE"]
        .round(2)
    )

    comparison["RMSE"] = (
        comparison["RMSE"]
        .round(2)
    )


    best_model = (
        comparison
        .loc[
            comparison["RMSE"].idxmin(),
            "Model"
        ]
    )


    st.subheader(
        "🏆 Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Naive RMSE",
        f"{naive_rmse:,.0f}"
    )

    c2.metric(
        "Exponential Smoothing RMSE",
        f"{es_rmse:,.0f}"
    )

    c3.metric(
        "ARIMA RMSE",
        f"{arima_rmse:,.0f}"
    )


    st.success(
        f"Selected model based on lowest RMSE: "
        f"**{best_model}**"
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # FORECAST FUTURE 30 OBSERVATIONS
    # --------------------------------------------------------

    if best_model == "Naive Baseline":

        final_predictions = np.repeat(
            series.iloc[-1],
            30
        )

    elif best_model == "Exponential Smoothing":

        final_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        )

        final_fit = final_model.fit(
            optimized=True
        )

        final_predictions = np.asarray(
            final_fit.forecast(30),
            dtype=float
        )

    else:

        final_model = ARIMA(
            series,
            order=(1, 1, 1)
        )

        final_fit = final_model.fit()

        final_predictions = np.asarray(
            final_fit.forecast(30),
            dtype=float
        )


    # --------------------------------------------------------
    # FORECAST GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Future System Load Projection"
    )

    recent = forecast_df.tail(120)

    observed_values = (
        recent["Total_System_Load"]
        .astype(float)
        .to_numpy()
    )

    observed_positions = np.arange(
        len(observed_values)
    )

    future_positions = np.arange(
        len(observed_values) - 1,
        len(observed_values) + 30
    )

    forecast_line = np.concatenate(
        [
            [observed_values[-1]],
            final_predictions
        ]
    )


    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        observed_positions,
        observed_values,
        color=LAVENDER,
        linewidth=2.7,
        label="Observed System Load"
    )

    ax.plot(
        future_positions,
        forecast_line,
        color=ROSE,
        linewidth=2.7,
        linestyle="--",
        marker="o",
        markersize=3.5,
        label="Forecast"
    )

    ax.axvline(
        len(observed_values) - 1,
        color=PEACH,
        linestyle=":",
        linewidth=2,
        label="Forecast Start"
    )

    ax.set_title(
        "Future Total System Load Projection",
        fontsize=17,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel(
        "Observation Sequence"
    )

    ax.set_ylabel(
        "Total System Load"
    )

    ax.tick_params(
        colors=MUTED
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # FORECAST SUMMARY
    # --------------------------------------------------------

    forecast_average = (
        final_predictions.mean()
    )

    forecast_minimum = (
        final_predictions.min()
    )

    forecast_maximum = (
        final_predictions.max()
    )

    last_observed = (
        series.iloc[-1]
    )

    projected_change = (
        (
            final_predictions[-1]
            - last_observed
        )
        / last_observed
        * 100
    )


    st.subheader(
        "🔮 Forecast Summary"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Forecast",
        f"{forecast_average:,.0f}"
    )

    c2.metric(
        "Minimum Forecast",
        f"{forecast_minimum:,.0f}"
    )

    c3.metric(
        "Maximum Forecast",
        f"{forecast_maximum:,.0f}"
    )

    c4.metric(
        "Projected Change",
        f"{projected_change:.1f}%"
    )


    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Forecast Values"
    )

    forecast_table = pd.DataFrame(
        {
            "Forecast Observation": range(
                1,
                31
            ),
            "Projected System Load": np.round(
                final_predictions
            ).astype(int)
        }
    )

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📥 Export"
)

csv_data = (
    filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.sidebar.download_button(
    label="Download Analysis Data",
    data=csv_data,
    file_name="UAC_analysis_data.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAC System Capacity & Care Load Analytics | "
    "Analytical dashboard for research and capacity-planning support."
)

st.caption(
    "Forecasts and indicators are analytical estimates "
    "and should not be interpreted as official government forecasts."
)