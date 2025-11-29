import io
import json
import pickle
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sdv.evaluation.multi_table import evaluate_quality, get_column_plot, run_diagnostic
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from sdv.multi_table import HMASynthesizer

CREDIT_UNION_SAMPLE = {
    "members": "data/members.csv",
    "accounts": "data/accounts.csv",
    "transactions": "data/transactions.csv",
}


def load_sample_credit_union() -> Dict[str, pd.DataFrame]:
    tables = {}
    for name, path in CREDIT_UNION_SAMPLE.items():
        tables[name] = pd.read_csv(path)
    return tables


def _load_table_from_upload(upload) -> pd.DataFrame:
    if upload.name.endswith(".parquet"):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)


def detect_metadata(tables: Dict[str, pd.DataFrame]) -> MultiTableMetadata:
    metadata = MultiTableMetadata()
    for table_name, df in tables.items():
        table_meta = SingleTableMetadata()
        table_meta.detect_from_dataframe(df)
        # MultiTableMetadata.add_table accepts the name and table metadata as positional
        metadata.add_table(table_name, table_meta)

    # Simple heuristic to build relationships: match `<name>_id` references to table primary keys
    for child_name, child_df in tables.items():
        for column in child_df.columns:
            if column.endswith("_id") and column != "transaction_id":
                parent_name = column[:-3] + "s" if column.endswith("id") else column.replace("_id", "s")
                if parent_name in tables and column in tables[parent_name].columns:
                    try:
                        metadata.add_relationship(
                            parent_table_name=parent_name,
                            child_table_name=child_name,
                            parent_primary_key=column,
                            child_foreign_key=column,
                        )
                    except ValueError:
                        # Relationship may already exist or invalid; continue safely
                        pass
    return metadata


def describe_fields(metadata: MultiTableMetadata) -> Dict[str, pd.DataFrame]:
    descriptions = {}
    for table_name in metadata.tables:
        fields = metadata.tables[table_name]["fields"]
        rows = [
            {
                "field": field,
                "sdtype": details.get("sdtype"),
                "type": details.get("type"),
                "format": details.get("format"),
            }
            for field, details in fields.items()
        ]
        descriptions[table_name] = pd.DataFrame(rows)
    return descriptions


def apply_constraints(tables: Dict[str, pd.DataFrame], constraints: List[dict]) -> Dict[str, pd.DataFrame]:
    constrained = {name: df.copy() for name, df in tables.items()}
    for rule in constraints:
        table_name = rule.get("table")
        if table_name not in constrained:
            continue
        df = constrained[table_name]
        if rule.get("type") == "between":
            column = rule.get("column")
            min_val = rule.get("min")
            max_val = rule.get("max")
            if column in df.columns:
                df = df[df[column].between(min_val, max_val)]
        elif rule.get("type") == "unique":
            columns = rule.get("columns", [])
            if columns:
                df = df.drop_duplicates(subset=columns)
        constrained[table_name] = df
    return constrained


def render_constraint_builder():
    st.markdown("### Constraint builder")
    if "constraints" not in st.session_state:
        st.session_state.constraints = []

    with st.expander("Add constraint", expanded=False):
        table_name = st.text_input("Table name", value="members")
        constraint_type = st.selectbox("Constraint type", ["between", "unique"])
        new_rule: Optional[dict] = None
        if constraint_type == "between":
            column = st.text_input("Column")
            min_val = st.number_input("Min value", value=0.0)
            max_val = st.number_input("Max value", value=10000.0)
            if st.button("Add between constraint"):
                new_rule = {"table": table_name, "type": "between", "column": column, "min": min_val, "max": max_val}
        else:
            columns = st.text_input("Columns (comma separated)")
            if st.button("Add unique constraint"):
                new_rule = {"table": table_name, "type": "unique", "columns": [c.strip() for c in columns.split(",") if c.strip()]}

        if new_rule:
            st.session_state.constraints.append(new_rule)
            st.success("Constraint added")

    if st.session_state.constraints:
        st.json(st.session_state.constraints)
        st.download_button(
            "Download constraints JSON",
            data=json.dumps(st.session_state.constraints, indent=2),
            file_name="constraints.json",
            mime="application/json",
        )

    upload = st.file_uploader("Upload constraints JSON", type=["json"], key="constraint_upload")
    if upload:
        try:
            st.session_state.constraints = json.load(upload)
            st.success("Constraints loaded")
        except json.JSONDecodeError:
            st.error("Invalid JSON file")


def render_step_1_load():
    st.header("Step 1: Load credit union data")
    st.write("Upload CSV or Parquet files for each table or use the bundled sample dataset.")

    if st.button("Load sample credit union dataset"):
        st.session_state.tables = load_sample_credit_union()
        st.success("Loaded sample data")

    uploads = st.file_uploader(
        "Upload tables (multiple files allowed)",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        key="table_uploads",
    )
    if uploads:
        tables = {}
        for upload in uploads:
            tables[upload.name.rsplit(".", 1)[0]] = _load_table_from_upload(upload)
        st.session_state.tables = tables
        st.success(f"Loaded {len(tables)} table(s)")

    if "tables" in st.session_state:
        show_tables = st.checkbox("Show table preview", value=True)
        if show_tables:
            for name, df in st.session_state.tables.items():
                st.subheader(f"Table: {name}")
                st.dataframe(df.head())


def render_step_2_metadata():
    st.header("Step 2: Detect or upload metadata")
    if "tables" not in st.session_state:
        st.info("Load data first.")
        return

    if st.button("Detect metadata from tables"):
        st.session_state.metadata = detect_metadata(st.session_state.tables)
        st.success("Metadata detected")

    upload = st.file_uploader("Upload MultiTable metadata JSON", type=["json"], key="metadata_upload")
    if upload:
        st.session_state.metadata = MultiTableMetadata.load_from_json(upload)
        st.success("Metadata loaded from file")

    if "metadata" in st.session_state:
        st.subheader("Field summary")
        descriptions = describe_fields(st.session_state.metadata)
        for name, df in descriptions.items():
            st.markdown(f"**{name}**")
            st.dataframe(df)

        st.download_button(
            "Download metadata JSON",
            data=st.session_state.metadata.to_json(),
            file_name="metadata.json",
            mime="application/json",
        )


def render_step_3_constraints():
    st.header("Step 3: Configure constraints")
    if "tables" not in st.session_state:
        st.info("Load data first.")
        return

    render_constraint_builder()

    if st.button("Apply constraints to data"):
        constraints = st.session_state.get("constraints", [])
        st.session_state.tables = apply_constraints(st.session_state.tables, constraints)
        st.success("Constraints applied to working dataset")


def render_step_4_train():
    st.header("Step 4: Train or upload synthesizer")
    if "metadata" not in st.session_state:
        st.info("Detect metadata first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train HMASynthesizer"):
            synthesizer = HMASynthesizer(st.session_state.metadata)
            working_tables = st.session_state.get("tables")
            synthesizer.fit(working_tables)
            st.session_state.synthesizer = synthesizer
            st.success("Synthesizer trained")

    with col2:
        model_upload = st.file_uploader("Upload synthesizer .pkl", type=["pkl"], key="model_upload")
        if model_upload:
            st.session_state.synthesizer = pickle.load(model_upload)
            st.success("Synthesizer loaded")

    if "synthesizer" in st.session_state:
        st.download_button(
            "Download trained synthesizer",
            data=pickle.dumps(st.session_state.synthesizer),
            file_name="synthesizer.pkl",
            mime="application/octet-stream",
        )


def render_step_5_generate():
    st.header("Step 5: Generate synthetic data")
    if "synthesizer" not in st.session_state:
        st.info("Train or load a synthesizer first.")
        return

    scale = st.slider("Sample scale (relative to original size)", 0.1, 3.0, 1.0, step=0.1)
    if st.button("Generate synthetic data"):
        synthetic = st.session_state.synthesizer.sample(scale=scale)
        st.session_state.synthetic = synthetic
        st.success("Synthetic data generated")

    if "synthetic" in st.session_state:
        for name, df in st.session_state.synthetic.items():
            st.subheader(f"Synthetic table: {name}")
            st.dataframe(df.head())
            st.download_button(
                f"Download {name} synthetic CSV",
                data=df.to_csv(index=False),
                file_name=f"{name}_synthetic.csv",
                mime="text/csv",
            )


def render_step_6_evaluate():
    st.header("Step 6: Evaluate synthetic vs original")
    if "metadata" not in st.session_state:
        st.info("Detect metadata first.")
        return

    st.write("Upload real and synthetic data or use the session data.")
    real_uploads = st.file_uploader(
        "Upload real tables", type=["csv", "parquet"], accept_multiple_files=True, key="real_eval_upload"
    )
    synthetic_uploads = st.file_uploader(
        "Upload synthetic tables", type=["csv", "parquet"], accept_multiple_files=True, key="synthetic_eval_upload"
    )

    real_data = st.session_state.get("tables")
    synthetic_data = st.session_state.get("synthetic")

    if real_uploads:
        real_data = {u.name.rsplit(".", 1)[0]: _load_table_from_upload(u) for u in real_uploads}
    if synthetic_uploads:
        synthetic_data = {u.name.rsplit(".", 1)[0]: _load_table_from_upload(u) for u in synthetic_uploads}

    if st.button("Run evaluation"):
        if not real_data or not synthetic_data:
            st.error("Both real and synthetic data are required")
        else:
            diagnostic_report = run_diagnostic(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=st.session_state.metadata,
            )
            quality_report = evaluate_quality(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=st.session_state.metadata,
            )
            st.session_state.eval_diagnostic = diagnostic_report
            st.session_state.eval_quality = quality_report
            st.success("Evaluation complete")

    if "eval_quality" in st.session_state:
        st.subheader("Quality score")
        st.metric("Overall quality", f"{st.session_state.eval_quality.get_score():.3f}")
        details = st.session_state.eval_quality.get_details()
        st.subheader("Quality details")
        st.dataframe(details)

    if "eval_diagnostic" in st.session_state:
        diag_results = st.session_state.eval_diagnostic.get_results()
        st.subheader("Diagnostic results")
        st.dataframe(diag_results)

    if real_data and synthetic_data:
        st.subheader("Column distribution comparison")
        with st.expander("Select column plot"):
            table_names = list(real_data.keys())
            table_choice = st.selectbox("Table", table_names)
            if table_choice:
                column_choice = st.selectbox("Column", list(real_data[table_choice].columns))
                if column_choice:
                    fig = get_column_plot(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        metadata=st.session_state.metadata,
                        table_name=table_choice,
                        column_name=column_choice,
                    )
                    st.plotly_chart(fig)


def main():
    st.set_page_config(page_title="SDV Credit Union Demo", layout="wide")
    st.title("Credit Union Synthetic Data Demo")
    st.write("Step through the workflow to profile, model, generate, and evaluate multi-table credit union data using SDV.")

    steps = {
        "1. Load data": render_step_1_load,
        "2. Metadata": render_step_2_metadata,
        "3. Constraints": render_step_3_constraints,
        "4. Train model": render_step_4_train,
        "5. Generate data": render_step_5_generate,
        "6. Evaluate": render_step_6_evaluate,
    }
    choice = st.radio("Navigate steps", list(steps.keys()))
    steps[choice]()


if __name__ == "__main__":
    main()
