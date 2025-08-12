# import os
# os.environ["DYNAMIC_SAM2_PATH"] = "/Users/aditi/Documents/GitHub/TokenSHAP/dynamic_sam2"
# from dotenv import load_dotenv
# load_dotenv()

# import sys
# import pandas as pd

# # Adds the root project directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# from token_shap.base import OpenAIEmbeddings, OpenAIModel, LocalModel, HuggingFaceEmbeddings, OllamaModel, TfidfTextVectorizer
# from token_shap.token_shap import StringSplitter, TokenSHAP
# api_k = os.getenv("OPENAI_API_KEY")

# import streamlit as st
# st.title("GPT-4o Token Probability Viewer")

# # Prompt input
# user_prompt = st.text_area("Enter your prompt:", height=150)
# run = st.button("Analyze")

# # 2) Placeholder lives immediately below the button
# progress_slot = st.empty()
# combo_slot = st.empty() #single placeholder
# model = OpenAIModel(
#     model_name='gpt-4o',
#     api_key=api_k
# )
# openai_embedding = OpenAIEmbeddings(
#     model="text-embedding-3-small", 
#     api_key=api_k
# )
# splitter = StringSplitter()
# token_shap_openai = TokenSHAP(model=model, splitter=splitter, vectorizer=openai_embedding)
# # Run button
# if run:
#     if user_prompt.strip() == "":
#         st.warning("Please enter a prompt first.")
#     elif len(user_prompt.strip()) < 10:
#         raise ValueError("Prompt too short for TokenSHAP analysis. Please enter at least 2 tokens.")

#     else:
#         # 3) Create the bar + status inside the placeholder
#         with progress_slot.container():
#             bar = st.progress(0)
#             status = st.empty()
#         combo_box = combo_slot.empty()
#         chart_slot = st.empty()
#         chart = chart_slot.line_chart(pd.DataFrame())
#         # 4) Callback that updates those same widgets
#         def progress_cb(done: int, total: int, combination: str):
#             bar.progress(done / total)
#             status.text(f"Processing {done}/{total} combinations...")
#             if combination is not None:
#                 combo_box.code(combination)

#         # 5) Run your analysis and pass the callback
#         try:
#             df_openai = token_shap_openai.analyze(
#                 user_prompt,
#                 sampling_ratio=1.0,
#                 print_highlight_text=True,
#                 progress_cb=progress_cb,
#             )
#         finally:
#             bar.progress(1.0)
#             status.text("Done!")
#             combo_box.empty()

#         # (optional) remove the bar after finishing:
#         # progress_slot.empty()

#         # 6) Show results
#         st.write("Your prompt was:\n\n", user_prompt)
#         st.dataframe(df_openai)
#         st.pyplot(token_shap_openai.plot_colored_text())

import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from token_shap.base import (
    OpenAIEmbeddings, OpenAIModel,
    LocalModel, HuggingFaceEmbeddings, OllamaModel
)
from token_shap.token_shap import StringSplitter, TokenSHAP
os.environ["DYNAMIC_SAM2_PATH"] = "/Users/aditi/Documents/GitHub/TokenSHAP/dynamic_sam2"
# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Model TokenSHAP Viewer", layout="wide")
st.title("Multi-Model Token Probability & Attribution Viewer")

# =========================
# Sidebar: Model selection
# =========================
st.sidebar.header("Settings")

model_options = [
    "OpenAI - GPT-4o",
    "OpenAI - GPT-4o-mini",
    "OpenAI - GPT-4-turbo",
    "OpenAI - GPT-3.5-turbo",
    "Ollama - llama3",
    "Ollama - mistral",
    "Local HuggingFace - Mistral-7B",
    "Local HuggingFace - LLaMA-2-7B"
]

selected_models = st.sidebar.multiselect(
    "Select one or more models to run:",
    model_options,
    default=["OpenAI - GPT-4o"]
)

embedding_choice = st.sidebar.selectbox(
    "Embeddings Model:",
    [
        "text-embedding-3-small",  # OpenAI
        "text-embedding-3-large",  # OpenAI
        "all-MiniLM-L6-v2"         # HuggingFace
    ]
)

# Prompt input
user_prompt = st.text_area("Enter your prompt:", height=150)
run = st.button("Analyze")

# =========================
# Helper: model init
# =========================
def init_model(model_name: str):
    """Return (model, embedding) pair based on selection."""
    api_k_openai = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")
    api_k_hf = os.getenv("HUGGINGFACE_API_KEY") or st.sidebar.text_input("HuggingFace API Key", type="password")

    if model_name.startswith("OpenAI"):
        m = model_name.split(" - ")[1].lower()
        return (
            OpenAIModel(model_name=m, api_key=api_k_openai),
            OpenAIEmbeddings(model=embedding_choice, api_key=api_k_openai)
        )

    elif model_name.startswith("Ollama"):
        m = model_name.split(" - ")[1]
        return (
            OllamaModel(model_name=m, api_url="http://localhost:11434"),
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    elif model_name.startswith("Local HuggingFace"):
        if "Mistral" in model_name:
            m = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            m = "meta-llama/Llama-2-7b-chat-hf"
        return (
            LocalModel(model_name=m, api_key=api_k_hf),
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
import numpy as np

def compare_attributions(values: dict):
    # align keys
    keys = sorted(set(phi_a) | set(phi_b))
    a = np.array([phi_a.get(k_, 0.0) for k_ in keys])
    b = np.array([phi_b.get(k_, 0.0) for k_ in keys])

    # top-k overlap
    top_a = set([keys[i] for i in a.argsort()[-k:]])
    top_b = set([keys[i] for i in b.argsort()[-k:]])
    overlap = len(top_a & top_b) / max(1, len(top_a | top_b))

    # rank correlation (Spearman)
    rank_a = a.argsort().argsort()
    rank_b = b.argsort().argsort()
    rho = np.corrcoef(rank_a, rank_b)[0,1]

    # signed difference
    delta = {k_: float(a_i - b_i) for k_, a_i, b_i in zip(keys, a, b)}
    return {"overlap@k": overlap, "spearman": float(rho), "delta": delta}


# =========================
# Run analysis
# =========================
if run:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        shapley_values_models = {}
        for selected in selected_models:
            st.subheader(f"Results for {selected}")
            model, embeddings = init_model(selected)

            splitter = StringSplitter()
            token_shap_instance = TokenSHAP(model=model, splitter=splitter, vectorizer=embeddings)

            # Progress bar + placeholders
            progress_slot = st.empty()
            with progress_slot.container():
                bar = st.progress(0)
                combo_box = st.empty()
                status = st.empty()

            # Progress callback
            def progress_cb(done: int, total: int, combination: str):
                bar.progress(done / total)
                status.text(f"Processing {done}/{total} combinations...")
                combo_box.code(combination)

            # Run analysis
            try:
                df_results = token_shap_instance.analyze(
                    user_prompt,
                    sampling_ratio=1.0,
                    progress_cb=progress_cb
                )
                shapley_values_models[selected] = token_shap_instance.shapley_values
            finally:
                bar.progress(1.0)
                status.text("Done!")
                combo_box.empty()

            # Show results
            st.write("Shapley Value Dataframe")
            st.dataframe(df_results)

            st.write("Token Importance Visualization")
            st.pyplot(token_shap_instance.plot_colored_text())
        if len(selected_models) < 2:
            st.info("Pick at least 2 models to compare.")
        else:
            ref = st.selectbox("Reference model", selected_models, index=0)

    # 2) Build tokens×models matrix, align tokens, fill missing with 0
            df_phi = pd.DataFrame({m: shapley_values_models[m] for m in selected_models}).fillna(0.0)

    # (optional) normalize per-model so columns sum to 1 (helps comparability)
            normalize = st.checkbox("Normalize attributions per model (L1)", value=True)
            if normalize:
                df_phi = df_phi.apply(lambda c: c / (np.abs(c).sum() or 1.0), axis=0)

    # 3) Compute Δ vs reference for all models
            delta_df = df_phi.subtract(df_phi[ref], axis=0)  # tokens × models

                # 4) Focus on top-K tokens by max |Δ| across models
            k = st.slider("Top-K tokens by |Δ|", min_value=2, max_value=min(50, len(delta_df)), value=min(15, len(delta_df)))
            top_tokens = delta_df.abs().max(axis=1).sort_values(ascending=False).head(k).index
            delta_top = delta_df.loc[top_tokens, [m for m in selected_models if m != ref]]

            st.caption(f"Δφ = φ(model) − φ({ref})  (positive ⇒ token mattered more than in {ref})")

                # 5) Plot grouped bars with Altair
            plot_data = delta_top.reset_index().melt(id_vars="index", var_name="model", value_name="delta")
            plot_data = plot_data.rename(columns={"index": "token"})

            chart = (
                alt.Chart(plot_data)
                .mark_bar()
                .encode(
                    x=alt.X("token:N", sort=None, title="Token"),
                    y=alt.Y("delta:Q", title="Δφ vs reference"),
                    color="model:N",
                    column=alt.Column("model:N", title=None)  # small multiples (one column per model)
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

                # (nice extra) table of numbers
            with st.expander("See numeric Δφ values"):
                st.dataframe(delta_top)
            st.markdown("---")
