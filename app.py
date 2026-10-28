"""Streamlit front end for the Customer Service Chatbot."""

import json
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.chatbot import get_reply


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PATHS AND THEME
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

MODEL_METRICS_PATH = APP_DIR / "models" / "evaluation_metrics.json"
ROOT_METRICS_PATH = APP_DIR / "evaluation_metrics.json"

METRICS_PATH = (
    MODEL_METRICS_PATH
    if MODEL_METRICS_PATH.exists()
    else ROOT_METRICS_PATH
)

PRIMARY = "#14685E"
PRIMARY_DARK = "#102A43"
SUCCESS = "#13665C"
WARNING = "#E95946"
MUTED = "#244150"


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

HOME = "🏠 Home"
CHAT = "💬 Chat"
INTENTS = "🎯 Intent Analysis"
DATA = "🗂️ Dataset Insights"
PERF = "📈 Model Performance"
ABOUT = "ℹ️ About"
CONTACT = "✉️ Contact"

WELCOME = "Hi! How can I help you today?"


# --------------------------------------------------
# QUICK CHAT ACTIONS
# --------------------------------------------------

QUICK_ACTIONS = [
    ("📦 Track My Order", "Where is my order?"),
    ("❌ Cancel Order", "I want to cancel my order"),
    ("↩️ Refund & Returns", "How do I get a refund?"),
    ("💳 Payment Issues", "My payment failed"),
    ("🚚 Delivery Time", "How long does delivery take?"),
    ("🎧 Contact Support", "I want to talk to a human agent"),
]


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

.stApp {
    background: #F7FCF8;
    color: #102A43;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #E4FBF4;
    border-right: 1px solid #BFEDE1;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] * {
    color: #102A43;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #647C89;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: .65rem .8rem;
    border-radius: 10px;
    margin-bottom: 5px;
    width: 100%;
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #D2F7EC;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: #35566A;
    font-size: .94rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #C8F4E8;
    border: 1px solid #74D8C2;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #087F70;
    font-weight: 700;
}

/* Branding */
.brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-weight: 750;
    color: #102A43;
    font-size: 1.02rem;
    line-height: 1.25;
    margin-bottom: 1.15rem;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #18BCA4, #0F9F8F);
    font-size: 1.25rem;
}

.brand small {
    display: block;
    font-weight: 450;
    color: #647C89;
    font-size: .76rem;
    margin-top: .18rem;
}

/* Headings */
h1, h2, h3 {
    color: #102A43;
    letter-spacing: -0.025em;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 650;
}

/* Hero header */
.hero {
    background: linear-gradient(
        115deg,
        #D9FAF1 0%,
        #F2FCF6 58%,
        #FFF5E8 100%
    );
    border: 1px solid #BFEDE1;
    border-radius: 18px;
    padding: 1.45rem 1.65rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 5px 18px rgba(18, 73, 74, .055);
}

.hero h1 {
    margin: 0;
    font-size: clamp(1.45rem, 2.2vw, 1.95rem);
    color: #102A43;
}

.hero p {
    margin: .42rem 0 0;
    color: #4D6A7D;
    font-size: .96rem;
}

/* Dashboard cards */
.kpi {
    background: linear-gradient(145deg, #FFFFFF, #F0FBF6);
    border: 1px solid #D3EEE6;
    border-radius: 15px;
    padding: 1rem 1.15rem;
    box-shadow: 0 3px 12px rgba(18, 73, 74, .04);
    min-height: 118px;
}

.kpi-label {
    color: #647C89;
    font-size: .83rem;
    font-weight: 600;
}

.kpi-value {
    color: #102A43;
    font-size: 1.65rem;
    font-weight: 750;
    margin-top: .28rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.kpi-sub {
    color: #81969B;
    font-size: .76rem;
    margin-top: .28rem;
}

/* Content panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D5EDE5;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(18, 73, 74, .035);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #F5C8BE;
    color: #087F70;
    background: #F0FCF8;
    font-weight: 600;
    padding: .58rem .85rem;
    transition: all .15s ease;
}

.stButton > button:hover {
    border-color: #18BCA4;
    background: #D9FAF1;
    color: #075E55;
}

.stButton > button:focus {
    box-shadow: 0 0 0 .2rem rgba(24, 188, 164, .18);
    border-color: #0F9F8F;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #F2FCF8;
    border: 1px solid #D6EEE7;
    border-radius: 12px;
    padding: .8rem 1rem;
}

/* Chat */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid #DCEFE8;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #D5EDE5;
    border-radius: 10px;
    overflow: hidden;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea {
    border-radius: 10px;
    border-color: #C4E9DF;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #0F9F8F;
    box-shadow: 0 0 0 .15rem rgba(15, 159, 143, .12);
}

/* Responsive layout */
@media (max-width: 768px) {
    .block-container {
        padding-top: .7rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 1.1rem;
    }

    .kpi {
        padding: .8rem;
        min-height: 105px;
    }

    .kpi-value {
        font-size: 1.35rem;
    }
}
</style>
"""

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def now() -> str:
    return datetime.now().strftime("%I:%M %p")


def pct(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "—"


def num(value) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return "—"


def format_intent(name: str) -> str:
    return str(name).replace("_", " ").title()


def class_frame(metrics: dict) -> pd.DataFrame:
    report = metrics.get("classification_report", {})

    rows = [
        {
            "intent": name,
            "label": format_intent(name),
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "support": int(values["support"]),
        }
        for name, values in report.items()
        if isinstance(values, dict)
        and name not in ("macro avg", "weighted avg")
    ]

    return pd.DataFrame(rows)


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME,
                "time": now(),
            }
        ]

    if "predictions" not in st.session_state:
        st.session_state.predictions = []


def ask(prompt: str) -> None:
    """Send a message through the existing chatbot and store the result."""

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "time": now(),
        }
    )

    try:
        reply, intent, confidence = get_reply(prompt)

    except FileNotFoundError as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ {error}",
                "time": now(),
            }
        )
        return

    except Exception as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ Something went wrong: {error}",
                "time": now(),
            }
        )
        return

    confidence = float(confidence)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "time": now(),
            "intent": intent,
            "confidence": confidence,
        }
    )

    st.session_state.predictions.append(
        {
            "time": now(),
            "query": prompt,
            "intent": intent,
            "confidence": confidence,
        }
    )


def start_chat(prompt: str) -> None:
    st.session_state["nav"] = CHAT
    ask(prompt)


def clear_chat() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME,
            "time": now(),
        }
    ]


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items) -> None:
    columns = st.columns(len(items))

    for column, (label, value, subtitle) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def style_chart(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED,
            titleColor="#35566A",
            gridColor="#E3ECE8",
            domainColor="#D0DFDA",
        )
        .configure_legend(
            labelColor=MUTED,
            titleColor="#35566A",
        )
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

def home_page(metrics: dict) -> None:
    page_header(
        "🤖 Customer Support Chatbot",
        "Get instant help with orders, payments, delivery and more.",
    )

    kpi_row(
        [
            (
                "Supported intents",
                num(metrics.get("intent_classes")),
                "Customer service categories",
            ),
            (
                "Training examples",
                num(metrics.get("usable_rows")),
                f"{num(metrics.get('dropped_rows'))} rows removed during cleaning",
            ),
            (
                "Test accuracy",
                pct(metrics.get("test_accuracy")),
                "Held-out test set",
            ),
            (
                "Queries this session",
                str(len(st.session_state.predictions)),
                "Queries submitted in this session",
            ),
        ]
    )

    st.write("")

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("✨ Common queries")
            st.caption("Choose a topic to start a conversation.")

            columns = st.columns(2)

            for index, (label, prompt) in enumerate(QUICK_ACTIONS):
                with columns[index % 2]:
                    st.button(
                        label,
                        key=f"home_quick_{index}",
                        on_click=start_chat,
                        args=(prompt,),
                        use_container_width=True,
                    )

    with right:
        with st.container(border=True):
            st.subheader("🧠 Model overview")
            st.markdown("**Pipeline**")
            st.write("TF-IDF text features → Logistic Regression")
            st.divider()

            col1, col2 = st.columns(2)
            col1.metric("Intent categories", num(metrics.get("intent_classes")))
            report = metrics.get("classification_report", {})
            macro = report.get("macro avg", {})
            col2.metric("Test samples", num(macro.get("support")))

            st.caption(
                "The chatbot predicts an intent and returns its mapped response. "
                "It is not connected to live order or payment systems."
            )


# --------------------------------------------------
# CHAT PAGE
# --------------------------------------------------

def chat_page(metrics: dict) -> None:
    page_header(
        "💬 Chat with Support",
        "Ask about orders, delivery, refunds, payments and your account.",
    )

    with st.container(border=True):
        st.subheader("Quick questions")
        chips = st.columns(3)

        for index, (label, prompt) in enumerate(QUICK_ACTIONS):
            with chips[index % 3]:
                st.button(
                    label,
                    key=f"chat_quick_{index}",
                    on_click=ask,
                    args=(prompt,),
                    use_container_width=True,
                )

    with st.container(height=480, border=True):
        for message in st.session_state.messages:
            avatar = (
                "🤖"
                if message["role"] == "assistant"
                else "🧑"
            )

            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message.get("intent") is not None:
                    with st.expander("🔎 Model details"):
                        st.write(
                            f"Predicted intent: `{message['intent']}`"
                        )
                        st.write(
                            f"Confidence: `{message['confidence']:.1%}`"
                        )

                st.caption(message.get("time", ""))

    prompt = st.chat_input(
        "Type your customer support question..."
    )

    if prompt:
        ask(prompt)
        st.rerun()


# --------------------------------------------------
# INTENT ANALYSIS PAGE
# --------------------------------------------------

def intent_page(metrics: dict) -> None:
    page_header(
        "🎯 Intent Analysis",
        "Explore intent predictions and confidence scores from this chat session.",
    )

    log = st.session_state.predictions

    last = log[-1] if log else None

    average_confidence = (
        sum(item["confidence"] for item in log) / len(log)
        if log
        else None
    )

    kpi_row(
        [
            (
                "Last intent",
                format_intent(last["intent"]) if last else "—",
                "Most recent prediction",
            ),
            (
                "Last confidence",
                pct(last["confidence"]) if last else "—",
                "Latest model confidence",
            ),
            (
                "Average confidence",
                pct(average_confidence),
                "Across this session",
            ),
            (
                "Predictions",
                str(len(log)),
                "This session",
            ),
        ]
    )

    st.write("")

    if not log:
        st.info(
            "No predictions yet. Ask a question on the Chat page "
            "to see live intent analysis."
        )
        return

    dataframe = pd.DataFrame(log)
    dataframe["query_no"] = range(1, len(dataframe) + 1)

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("📋 Recent predictions")

            table = dataframe.iloc[::-1][
                ["time", "query", "intent", "confidence"]
            ]

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "time": "Time",
                    "query": "Query",
                    "intent": "Predicted intent",
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=1,
                        format="%.2f",
                    ),
                },
            )

    with right:
        with st.container(border=True):
            st.subheader("📈 Confidence over session")

            chart = (
                alt.Chart(dataframe)
                .mark_line(
                    point=True,
                    color=PRIMARY,
                    strokeWidth=3,
                )
                .encode(
                    x=alt.X(
                        "query_no:O",
                        title="Query number",
                    ),
                    y=alt.Y(
                        "confidence:Q",
                        title="Confidence",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    tooltip=[
                        alt.Tooltip("query:N", title="Query"),
                        alt.Tooltip("intent:N", title="Intent"),
                        alt.Tooltip(
                            "confidence:Q",
                            title="Confidence",
                            format=".1%",
                        ),
                    ],
                )
                .properties(height=300)
            )

            st.altair_chart(
                style_chart(chart),
                use_container_width=True,
            )


# --------------------------------------------------
# DATASET INSIGHTS PAGE
# --------------------------------------------------

def dataset_page(metrics: dict) -> None:
    page_header(
        "🗂️ Dataset Insights",
        "Explore the training data and its distribution across customer intents.",
    )

    source_rows = metrics.get("source_rows")
    dropped_rows = metrics.get("dropped_rows")

    dropped_share = (
        f"{dropped_rows / source_rows:.1%} of source"
        if source_rows and dropped_rows is not None
        else ""
    )

    report = metrics.get("classification_report", {})
    test_size = report.get("macro avg", {}).get("support")

    test_share = (
        f"≈{test_size / metrics['usable_rows']:.0%} of usable rows"
        if test_size and metrics.get("usable_rows")
        else ""
    )

    kpi_row(
        [
            (
                "Source rows",
                num(source_rows),
                "Raw dataset",
            ),
            (
                "Usable rows",
                num(metrics.get("usable_rows")),
                "After cleaning",
            ),
            (
                "Dropped rows",
                num(dropped_rows),
                dropped_share,
            ),
            (
                "Intent classes",
                num(metrics.get("intent_classes")),
                "Customer categories",
            ),
        ]
    )

    st.write("")

    dataframe = class_frame(metrics)

    if dataframe.empty:
        st.info(
            "Intent distribution is unavailable because the metrics "
            "file has no class report."
        )
        return

    left, right = st.columns([3, 1], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("📊 Intent distribution")
            st.caption(
                "Number of samples per intent in the held-out test set."
            )

            chart = (
                alt.Chart(dataframe)
                .mark_bar(
                    cornerRadiusEnd=5,
                    color=PRIMARY,
                )
                .encode(
                    x=alt.X(
                        "support:Q",
                        title="Test samples",
                    ),
                    y=alt.Y(
                        "label:N",
                        sort="-x",
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="Intent"),
                        alt.Tooltip("support:Q", title="Samples"),
                    ],
                )
                .properties(
                    height=max(320, 24 * len(dataframe))
                )
            )

            st.altair_chart(
                style_chart(chart),
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            st.subheader("📌 Split summary")

            st.metric(
                "Test set size",
                num(test_size),
                test_share or None,
            )

            st.metric(
                "Dropped during cleaning",
                num(dropped_rows),
                dropped_share or None,
            )

            st.caption(
                "Dropped rows were excluded during training. "
                "The displayed evaluation metrics use the held-out test set."
            )


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

def performance_page(metrics: dict) -> None:
    page_header(
        "📈 Model Performance",
        "Review precision, recall and F1 scores on the held-out test set.",
    )

    report = metrics.get("classification_report", {})
    macro = report.get("macro avg", {})

    kpi_row(
        [
            (
                "Accuracy",
                pct(metrics.get("test_accuracy")),
                "Overall test accuracy",
            ),
            (
                "Macro precision",
                pct(macro.get("precision")),
                "Average across intents",
            ),
            (
                "Macro recall",
                pct(macro.get("recall")),
                "Average across intents",
            ),
            (
                "Macro F1",
                pct(macro.get("f1-score")),
                "Balanced F1 score",
            ),
        ]
    )

    st.write("")

    dataframe = class_frame(metrics)

    if dataframe.empty:
        st.info(
            "No per-intent metrics found in evaluation_metrics.json."
        )
        return

    with st.container(border=True):
        st.subheader("📊 F1 score by intent")
        st.caption(
            "Intents are sorted from lowest to highest F1 score. "
            "The axis shows the full 0–1 score range."
        )

        chart = (
            alt.Chart(dataframe)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X(
                    "f1:Q",
                    title="F1 score",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "label:N",
                    sort=alt.EncodingSortField(
                        field="f1",
                        order="ascending",
                    ),
                    title=None,
                ),
                color=alt.condition(
                    alt.datum.f1 < 0.98,
                    alt.value(WARNING),
                    alt.value(SUCCESS),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Intent"),
                    alt.Tooltip(
                        "f1:Q",
                        title="F1 score",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=max(320, 24 * len(dataframe))
            )
        )

        st.altair_chart(
            style_chart(chart),
            use_container_width=True,
        )

        st.caption(
            "Amber bars indicate intents below 0.98 F1. "
            "Green bars indicate intents at or above 0.98."
        )

    with st.container(border=True):
        st.subheader("📋 Detailed metrics")

        table = dataframe.sort_values("f1")[
            ["label", "precision", "recall", "f1", "support"]
        ]

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "label": "Intent",
                "precision": st.column_config.ProgressColumn(
                    "Precision",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
                "recall": st.column_config.ProgressColumn(
                    "Recall",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
                "f1": st.column_config.ProgressColumn(
                    "F1 score",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
                "support": st.column_config.NumberColumn(
                    "Test samples"
                ),
            },
        )





# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

def about_page(metrics: dict) -> None:
    page_header(
        "ℹ️ About This Project",
        "Explore the dataset, NLP pipeline, machine learning "
        "approach, and evaluation of the Customer Support Chatbot.",
    )

    # Dataset
    st.subheader("📊 Dataset")

    st.write(
        "The project uses a customer support dataset containing "
        "text-based queries and their corresponding intent labels. "
        "These examples help the model learn to identify the purpose "
        "of a customer message."
    )

    st.write(
        f"The dataset contains {num(metrics.get('usable_rows'))} "
        f"usable examples across "
        f"{num(metrics.get('intent_classes'))} supported intents."
    )

    st.divider()

    # Machine Learning and NLP
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Machine Learning Approach")

        st.write(
            "The chatbot uses a trained intent classification model "
            "to predict the category that best matches a customer's "
            "message. The predicted intent is used to select a "
            "corresponding predefined response."
        )

        st.markdown(
            """
            **Workflow**
            - Receive a customer query
            - Process the input text
            - Extract numerical text features
            - Predict the intent
            - Return the mapped response
            """
        )

    with col2:
        st.subheader("⚙️ NLP and Feature Extraction")

        st.write(
            "Natural Language Processing enables the application "
            "to work with text-based customer queries. TF-IDF "
            "feature extraction represents text numerically so "
            "the trained classifier can process it."
        )

        st.info(
            "TF-IDF and the trained classifier work together "
            "to convert customer text into an intent prediction."
        )

    st.divider()

    # Model Evaluation
    st.subheader("📈 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Supported Intents",
            num(metrics.get("intent_classes")),
        )

    with col2:
        st.metric(
            "Test Accuracy",
            pct(metrics.get("test_accuracy")),
        )

    st.write(
        "The model's accuracy is measured on a held-out test set. "
        "This metric indicates how often the model predicted the "
        "correct intent for examples in that test set."
    )

    st.caption(
        "Test accuracy alone does not establish real-world "
        "performance. Results can vary on new customer queries."
    )

    st.divider()

    # Technologies
    st.subheader("🛠️ Technologies Used")

    st.markdown(
        """
        - **Python:** Core programming language
        - **Pandas:** Dataset handling and analysis
        - **Scikit-learn:** Machine learning and text feature extraction
        - **TF-IDF:** Numerical representation of text
        - **Streamlit:** Interactive web application and dashboard
        """
    )

    st.divider()

    # Features
    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - Customer query classification
            - Predefined response mapping
            - Intent analysis dashboard
            """
        )

    with col2:
        st.markdown(
            """
            - Prediction confidence details
            - Model performance metrics
            - Interactive user interface
            """
        )

    st.divider()

    # Limitations
    st.subheader("⚠️ Scope and Limitations")

    st.write(
        "The chatbot responds using the intents and responses "
        "supported by its trained pipeline. It does not independently "
        "verify customer information or perform transactions."
    )

    st.info(
        "This is a demonstration application. It is not connected "
        "to live order, payment, refund, or customer account systems."
    )

    st.divider()

    # Objective
    st.subheader("🎯 Project Objective")

    st.write(
        "The objective is to demonstrate the practical application "
        "of Natural Language Processing, text feature extraction, "
        "machine learning-based intent classification, model "
        "evaluation, and interactive application development "
        "for a customer support use case."
    )


# --------------------------------------------------
# CONTACT PAGE
# --------------------------------------------------

def contact_page(metrics: dict = None) -> None:
    page_header(
        "✉️ Contact",
        "Developer profile, project details, and professional links.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👩‍💻 Developer")
        st.write("**Name:** Ayesha Nikhat")
        st.write("**Project:** Customer Support Chatbot")
        st.write("**Domain:** Machine Learning and NLP")

    with col2:
        st.subheader("🔗 Professional Links")
        st.markdown(
            "🔗 [GitHub](https://github.com/ayeshanikhat12)"
        )
        st.markdown(
            "💼 [LinkedIn](https://linkedin.com/in/ayesha-nikhat-12a4aa305)"
        )

    st.divider()

    st.subheader("🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - **Language:** Python
            - **Interface:** Streamlit
            - **Text Features:** TF-IDF
            """
        )

    with col2:
        st.markdown(
            """
            - **Application:** Intent Classification
            - **Model Files:** Pickle
            - **Domain:** Natural Language Processing
            """
        )

    st.divider()

    st.subheader("🎯 Project Purpose")

    st.write(
        "This project demonstrates the application of NLP and "
        "machine learning to classify customer queries and return "
        "mapped responses through an interactive web dashboard."
    )

# --------------------------------------------------
# NAVIGATION MAP
# --------------------------------------------------

NAV_PAGES = {
    HOME: home_page,
    CHAT: chat_page,
    INTENTS: intent_page,
    DATA: dataset_page,
    PERF: performance_page,
    ABOUT: about_page,
    CONTACT : contact_page,

}



# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    init_state()

    metrics = load_metrics() or {}

    # Banner image on all pages
    banner_path = APP_DIR / "banner.png"

    if banner_path.exists():
        st.image(
            str(banner_path),
            use_container_width=True,
        )

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <span class="brand-mark">🤖</span>
                <div>
                    Customer Support
                    <small>AI Chatbot Dashboard</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("WORKSPACE")

        choice = st.radio(
            "Navigation",
            list(NAV_PAGES.keys()),
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()

        st.button(
            "🧹 Clear conversation",
            on_click=clear_chat,
            use_container_width=True,
        )

        st.divider()

        st.caption(
            f"Supported intents: {num(metrics.get('intent_classes'))}"
        )

        st.caption(
            f"Test accuracy: {pct(metrics.get('test_accuracy'))}"
        )

        st.caption("NLP · TF-IDF · Classification")

    # Metrics warning
    if not METRICS_PATH.exists():
        st.warning(
            "Evaluation metrics were not found in the project root "
            "or models folder. Performance figures will show as dashes; "
            "chat predictions are unchanged."
        )

    # Render selected page
    NAV_PAGES[choice](metrics)


if __name__ == "__main__":
    main()
