"""
Page 3: View Evaluation Results
"""

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

st.title("📊 Evaluation Results")
st.markdown("Analyze evaluation outcomes and explore detailed scoring information.")

# Check if results are available
if st.session_state.eval_results is None:
    st.warning("⚠️ No evaluation results available. Please run an evaluation first.")
    st.stop()

results = st.session_state.eval_results

# Summary Statistics
st.header("1. Summary Statistics")

# Check if we have summary stats
if not results.summary_stats:
    st.info("No summary statistics available. This may happen when using custom eval packs with non-standard data formats.")
else:
    # Create columns for each scorer
    scorer_cols = st.columns(len(results.summary_stats))

    for idx, (scorer_name, stats) in enumerate(results.summary_stats.items()):
        with scorer_cols[idx]:
            # Format scorer name for display
            display_name = scorer_name.replace("_", " ").title()

            st.markdown(f"### {display_name}")

            # Main metric
            accuracy = stats.get("accuracy", 0)
            st.metric(
                "Accuracy",
                f"{accuracy:.1%}",
                delta=None,
                help="Percentage of items that passed this scorer",
            )

            # Additional stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Passed", stats.get("passed", 0))
            with col2:
                st.metric("Failed", stats.get("failed", 0))

            # Score distribution for fuzzy match and LLM judge
            if (
                scorer_name in ["fuzzy_match", "llm_judge"]
                and "score_distribution" in stats
            ):
                st.markdown("**Score Distribution**")
                score_dist = stats["score_distribution"]
                st.bar_chart(score_dist)

            if stats.get("errors", 0) > 0:
                st.error(f"⚠️ {stats['errors']} items failed to score")

    # Add this new block after the for loop to show a total error summary:
    total_errors = sum(s.get("errors", 0) for s in results.summary_stats.values())
    if total_errors > 0:
        st.warning(
            f"⚠️ A total of {total_errors} scoring errors occurred across all scorers. Check the detailed results below for individual error messages."
        )
# Detailed Results Table
st.header("2. Detailed Results")

# Check if we have any items to display
if not results.items:
    st.warning("No evaluation items to display. This may happen if the ingestion didn't produce any valid items.")
    st.stop()

# Convert results to DataFrame for display
def results_to_dataframe(results) -> pd.DataFrame:
    data = []
    for item in results.items:
        # FIX: Make the string operations "None-safe" to prevent TypeError.
        # Use 'or ""' as a fallback for None values before slicing or checking length.
        row = {
            "ID": item.id or f"Item {results.items.index(item) + 1}",
            "Input": (item.input or "")[:100] + "..." if item.input and len(item.input) > 100 else (item.input or ""),
            "Output": (item.output or "")[:100] + "..." if item.output and len(item.output) > 100 else (item.output or ""),
            "Expected": (item.expected_output or "")[:100] + "..." if item.expected_output and len(item.expected_output) > 100 else (item.expected_output or ""),
        }

        # Add scores for each scorer
        for score in item.scores:
            row[f"{score.scorer_name}_score"] = score.score
            row[f"{score.scorer_name}_passed"] = "✅" if score.passed else "❌"

        data.append(row)

    return pd.DataFrame(data)


df_results = results_to_dataframe(results)

# Display options
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    show_full_text = st.checkbox("Show full text", value=False)

with col2:
    filter_failures = st.checkbox("Show failures only", value=False)

with col3:
    # Filter by scorer failures
    if filter_failures:
        scorer_filter = st.multiselect(
            "Filter by scorer failures:",
            [s for s in results.summary_stats.keys()],
            default=[],
        )

# Apply filters
display_df = df_results.copy()

if filter_failures and scorer_filter:
    # Filter to show only items that failed selected scorers
    mask = pd.Series([False] * len(display_df))
    for scorer in scorer_filter:
        mask |= display_df[f"{scorer}_passed"] == "❌"
    display_df = display_df[mask]

# Show the table
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400,
)

# Detailed Item View
st.header("3. Detailed Item Analysis")

# Select an item to view details
item_ids = [item.id or f"Item {idx + 1}" for idx, item in enumerate(results.items)]
selected_item_id = st.selectbox("Select an item to view details:", item_ids)

# Find the selected item
selected_idx = item_ids.index(selected_item_id)
selected_item = results.items[selected_idx]

# Display item details
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input")
    st.text_area(
        "Input",
        value=selected_item.input,
        height=150,
        disabled=True,
        key="detail_input",
        label_visibility="collapsed",
    )

    st.markdown("### Expected Output")
    st.text_area(
        "Expected Output",
        value=selected_item.expected_output,
        height=150,
        disabled=True,
        key="detail_expected",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("### Actual Output")
    st.text_area(
        "Actual Output",
        value=selected_item.output,
        height=150,
        disabled=True,
        key="detail_output",
        label_visibility="collapsed",
    )

    st.markdown("### Metadata")
    if selected_item.metadata:
        st.json(selected_item.metadata)
    else:
        st.text("No metadata available")

# Scorer Results for Selected Item
st.markdown("### Scoring Details")

for score in selected_item.scores:
    with st.expander(
        f"{score.scorer_name.replace('_', ' ').title()} - {'✅ Passed' if score.passed else '❌ Failed'}"
    ):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric("Score", f"{score.score:.3f}")
            st.metric("Passed", "Yes" if score.passed else "No")

        with col2:
            if score.reasoning:
                st.markdown("**Reasoning:**")
                st.write(score.reasoning)

            if score.details:
                st.markdown("**Additional Details:**")
                if isinstance(score.details, dict):
                    st.json(score.details)
                else:
                    st.write(score.details)

# Add trace timeline for OTel traces
if "otel_trace" in selected_item.metadata:
    st.markdown("### Trace Timeline")
    trace_data = selected_item.metadata.get("otel_trace", {})
    spans_to_display = []

    # Handle the 'manual_traces.json' format which has a "steps" key
    if "steps" in trace_data and isinstance(trace_data.get("steps"), list):
        spans_to_display = trace_data["steps"]
        # Standardize the title field for consistent display
        for span in spans_to_display:
            if 'name' not in span:
                span['name'] = span.get('stage', 'Unnamed Step')
    
    # Handle the standard OTLP/JSON format with "resource_spans"
    elif "resource_spans" in trace_data and isinstance(trace_data.get("resource_spans"), list):
        for rs in trace_data.get("resource_spans", []):
            for ss in rs.get("scope_spans", []):
                spans_to_display.extend(ss.get("spans", []))

    if not spans_to_display:
        st.warning("Could not find any spans or steps to display in the trace.")
    else:
        # Display all found spans/steps
        for i, span in enumerate(spans_to_display):
            # Use the span's name for the title, falling back to stage or a default
            span_title = span.get("name", span.get("stage", f"Span {i+1}"))
            with st.expander(f"**{i+1}.** {span_title}", expanded=(i == 0)):
                st.json(span)

# Export Results Preview
st.header("4. Results Summary")

st.info("💡 Go to the **Downloads** page to export full results in various formats.")

# Show configuration used
with st.expander("📋 Evaluation Configuration"):
    st.json(results.config)

# Show run metadata
with st.expander("📊 Run Metadata"):
    metadata = {
        "Total Items": len(results.items),
        "Evaluation Mode": results.metadata.get("mode", "Unknown"),
        "Run Timestamp": results.metadata.get("timestamp", "Unknown"),
        "Scorers Used": list(results.summary_stats.keys()),
    }
    st.json(metadata)