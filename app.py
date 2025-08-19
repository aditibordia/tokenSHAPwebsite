import os
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
        if embedding_choice.startswith("text-embedding"):
            return (
                OpenAIModel(model_name=m, api_key=api_k_openai),
                OpenAIEmbeddings(model=embedding_choice, api_key=api_k_openai)
            )
        else:
            st.error("Incompatible model type with selected embedding. Please select a compatible pair of model and embedding type.")
            st.stop()

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
def phi_similarity_percent(
    df: pd.DataFrame,
    token_id_col: str = "token",   # better: a stable id like "start_char"
    value_col: str = "phi",        # or "delta" if you use Δφ vs reference
    group_col: str | None = None,  # e.g., "prompt_id" to average across prompts
    min_overlap: int = 3,          # min shared tokens to compute similarity
) -> pd.DataFrame:
    """
    Returns a model x model matrix of token-importance similarity in percent.
    Expects df with columns: [model, token_id_col, value_col, (optional) group_col]
    """
    def corr_matrix(sub: pd.DataFrame):
    # if tokens are the index:
        pt = sub.reindex(columns=models)          # ensure consistent column order
        pt = pt.apply(pd.to_numeric, errors="coerce")  # coerce any stray strings
        pt = pt.dropna(how="all")                      # drop tokens missing for all models
        corr = pt.corr(method="pearson", min_periods=min_overlap)

        n = pt.notna().T @ pt.notna()                  # pairwise token overlaps
        return corr, n

    if group_col is None:
        corr, _ = corr_matrix(df)
        sim = ((corr.clip(-1, 1) + 1) / 2) * 100.0
        return sim.round(1)

    # Aggregate across groups (prompts) via Fisher z, weighted by overlaps
    models = sorted(df.columns.unique())
    z_sum = pd.DataFrame(0.0, index=models, columns=models)
    w_sum = pd.DataFrame(0.0, index=models, columns=models)

    corr, n = corr_matrix(df)   # df has many tokens x models
    # sim = ((corr.clip(-1,1) + 1) / 2) * 100
    # st.write(sim)

    # z_avg = z_sum / np.maximum(w_sum, 1)   # avoid div by zero
    # r = np.tanh(z_avg)                     # back to correlation
    sim = corr.abs() * 100
    return pd.DataFrame(sim, index=models, columns=models).round(1)

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
                    xOffset=alt.XOffset("model:N"),                 # ← group by model within each token
                    y=alt.Y("delta:Q", title="Δφ vs reference"),
                    color=alt.Color("model:N", title="Model"),
                    tooltip=["model:N", "token:N", alt.Tooltip("delta:Q", format=".3f")]
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

                # (nice extra) table of numbers
            with st.expander("See numeric Δφ values"):
                st.dataframe(delta_top)
            # df_phi columns: prompt_id, model, token (or start_char), phi
            sim_matrix = phi_similarity_percent(df_phi, token_id_col="start_char", value_col="phi", group_col="prompt_id")
            sim_df = sim_matrix.reset_index().melt(id_vars="index", var_name="model_b", value_name="similarity")
            sim_df = sim_df.rename(columns={"index":"model_a"})
            heat = (alt.Chart(sim_df)
            .mark_rect()
            .encode(
                x=alt.X("model_a:O", title=None),
                y=alt.Y("model_b:O", title=None),
                color=alt.Color("similarity:Q", scale=alt.Scale(domain=[0,100]), title="Similarity %"),
                tooltip=["model_a","model_b", alt.Tooltip("similarity:Q", format=".1f")]
            )
            .properties(height=280)
            )
            st.altair_chart(heat, use_container_width=True)
            d = set()
            for a, b, c in zip(sim_df['model_a'], sim_df['model_b'], sim_df['similarity']):
                if a != b and (b, a) not in d:
                    d.add((a, b))
                    st.write(f"{a} and {b} are **{c}%** similar")
            st.markdown("---")

st.write("Learn more about TokenSHAP here: [TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation](https://arxiv.org/html/2407.10114v1)")
