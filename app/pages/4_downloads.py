"""
Page 4: Download Center
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import io
from typing import Dict, Any

from core.reporting import (
    results_to_csv,
    results_to_json,
    generate_summary_report,
)

st.title("⬇️ Download Center")
st.markdown("Export evaluation results and related artifacts.")

# Check if results are available
if st.session_state.eval_results is None:
    st.warning("⚠️ No evaluation results available. Please run an evaluation first.")
    st.stop()

results = st.session_state.eval_results

# File naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"eval_results_{timestamp}"

# Download Options
st.header("1. Evaluation Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📄 CSV Format")
    st.markdown("Flat table format, ideal for Excel or data analysis tools.")
    
    csv_data = results_to_csv(results)
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name=f"{base_filename}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    st.markdown("### 📋 JSON Format")
    st.markdown("Structured format with full details and metadata.")
    
    json_data = results_to_json(results)
    st.download_button(
        label="Download Results JSON",
        data=json_data,
        file_name=f"{base_filename}.json",
        mime="application/json",
        use_container_width=True,
    )

with col3:
    st.markdown("### 📊 Summary Report")
    st.markdown("Human-readable summary with key insights.")
    
    summary_report = generate_summary_report(results)
    st.download_button(
        label="Download Summary Report",
        data=summary_report,
        file_name=f"{base_filename}_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

# Additional Exports
st.header("2. Additional Exports")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔧 Configuration Export")
    st.markdown("Export the configuration used for this evaluation run.")
    
    config_export = {
        "evaluation_config": results.config,
        "model_configs": st.session_state.model_configs,
        "selected_scorers": st.session_state.selected_scorers,
        "timestamp": timestamp,
    }
    
    st.download_button(
        label="Download Configuration",
        data=json.dumps(config_export, indent=2),
        file_name=f"{base_filename}_config.json",
        mime="application/json",
        use_container_width=True,
    )

with col2:
    st.markdown("### 📈 Detailed Scores")
    st.markdown("Export individual scores for each item and scorer.")
    
    # Create detailed scores DataFrame
    scores_data = []
    for item in results.items:
        for score in item.scores:
            scores_data.append({
                "item_id": item.id or f"Item_{results.items.index(item) + 1}",
                "scorer": score.scorer_name,
                "score": score.score,
                "passed": score.passed,
                "reasoning": score.reasoning,
            })
    
    scores_df = pd.DataFrame(scores_data)
    scores_csv = scores_df.to_csv(index=False)
    
    st.download_button(
        label="Download Detailed Scores",
        data=scores_csv,
        file_name=f"{base_filename}_detailed_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Future Placeholders
st.header("3. Coming Soon")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Logs")
    st.markdown("*Detailed execution logs will be available in a future update.*")
    st.button("Download Logs", disabled=True, use_container_width=True)

with col2:
    st.markdown("### 🔍 Traces")
    st.markdown("*OpenTelemetry traces will be available in a future update.*")
    st.button("Download Traces", disabled=True, use_container_width=True)

# Preview Section
st.header("4. Export Preview")

preview_type = st.selectbox(
    "Select export type to preview:",
    ["CSV Results", "JSON Results", "Summary Report", "Configuration"],
)

with st.expander("Preview", expanded=True):
    if preview_type == "CSV Results":
        # Show first few rows of CSV
        csv_preview = results_to_csv(results).split('\n')[:10]
        st.text('\n'.join(csv_preview) + "\n...")
    
    elif preview_type == "JSON Results":
        # Show truncated JSON
        json_obj = json.loads(results_to_json(results))
        json_obj["items"] = json_obj["items"][:2]  # Show only first 2 items
        st.json(json_obj)
    
    elif preview_type == "Summary Report":
        # Show first part of summary
        summary_lines = generate_summary_report(results).split('\n')[:30]
        st.markdown('\n'.join(summary_lines) + "\n\n*... (truncated)*")
    
    else:  # Configuration
        st.json(config_export)

# Usage Tips
st.header("5. Export Tips")

st.info("""
**💡 Tips for using exported data:**

- **CSV Format**: Best for importing into Excel, Google Sheets, or data analysis tools like pandas
- **JSON Format**: Ideal for programmatic processing or archiving complete evaluation runs
- **Summary Report**: Great for sharing results with stakeholders or including in documentation
- **Configuration Export**: Useful for reproducing evaluation runs or debugging issues

**📊 For advanced analysis**, consider using the JSON export with a Jupyter notebook to create custom visualizations and deeper insights.
""")

# Footer
st.markdown("---")
st.markdown(
    f"*Results generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*"
)
