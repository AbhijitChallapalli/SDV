import json
import pickle
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sdv.evaluation.multi_table import evaluate_quality, get_column_plot, run_diagnostic
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from sdv.metadata.errors import InvalidMetadataError

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
    """Auto-detect MultiTableMetadata from a dict of DataFrames and
    heuristically add relationships based on *_id columns.
    """
    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(data=tables)

    # Simple heuristic to build relationships: match `<name>_id` references to table primary keys
    for child_name, child_df in tables.items():
        for column in child_df.columns:
            if column.endswith("_id") and column != "transaction_id":
                # crude pluralization heuristic
                if column.endswith("id"):
                    parent_name = column[:-2] + "s"
                else:
                    parent_name = column.replace("_id", "s")

                # Skip self-relationships like members -> members
                if parent_name == child_name:
                    continue

                if parent_name in tables and column in tables[parent_name].columns:
                    try:
                        metadata.add_relationship(
                            parent_table_name=parent_name,
                            child_table_name=child_name,
                            parent_primary_key=column,
                            child_foreign_key=column,
                        )
                    except (ValueError, KeyError, InvalidMetadataError):
                        # Relationship may already exist or be invalid; continue safely
                        pass

    return metadata


def _table_meta_to_fields_dict(table_meta) -> Dict:
    """Convert a SingleTableMetadata (or dict-like) to a fields dict."""
    try:
        meta_dict = table_meta.to_dict()
    except AttributeError:
        meta_dict = table_meta

    fields = (
        meta_dict.get("fields")
        or meta_dict.get("columns")
        or meta_dict.get("columns_metadata")
        or {}
    )
    return fields


def describe_fields(metadata: MultiTableMetadata) -> Dict[str, pd.DataFrame]:
    """Build a dataframe per table describing fields/sdtypes."""
    descriptions: Dict[str, pd.DataFrame] = {}
    for table_name, table_meta in metadata.tables.items():
        fields = _table_meta_to_fields_dict(table_meta)
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


def get_visualizable_columns(
    metadata: MultiTableMetadata,
    table_name: str,
    all_columns: List[str],
) -> List[str]:
    """Return columns that have a visualizable sdtype (exclude 'id', etc.)."""
    table_meta = metadata.tables.get(table_name)
    if table_meta is None:
        return all_columns

    fields = _table_meta_to_fields_dict(table_meta)
    visualizable = []
    for col in all_columns:
        info = fields.get(col, {})
        sdtype = info.get("sdtype")
        # SDV complains about 'id' sdtype; treat others as okay
        if sdtype == "id":
            continue
        visualizable.append(col)

    return visualizable or all_columns


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


def render_constraint_builder(tables: Optional[Dict[str, pd.DataFrame]] = None):
    st.markdown("### Constraint builder")
    if "constraints" not in st.session_state:
        st.session_state.constraints = []

    table_options = sorted(list(tables.keys())) if tables else []

    with st.expander("Add constraint", expanded=False):
        if table_options:
            table_name = st.selectbox("Table name", table_options)
        else:
            table_name = st.text_input("Table name", value="members")

        constraint_type = st.selectbox("Constraint type", ["between", "unique"])
        new_rule: Optional[dict] = None

        if constraint_type == "between":
            column = st.text_input("Column")
            min_val = st.number_input("Min value", value=0.0)
            max_val = st.number_input("Max value", value=10000.0)
            if st.button("Add between constraint"):
                new_rule = {
                    "table": table_name,
                    "type": "between",
                    "column": column,
                    "min": min_val,
                    "max": max_val,
                }
        else:
            columns = st.text_input("Columns (comma separated)")
            if st.button("Add unique constraint"):
                new_rule = {
                    "table": table_name,
                    "type": "unique",
                    "columns": [c.strip() for c in columns.split(",") if c.strip()],
                }

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
        # Try using SDV's JSON loader directly
        try:
            st.session_state.metadata = MultiTableMetadata.load_from_json(upload)
            st.success("Metadata loaded from file")
        except (TypeError, AttributeError):
            upload.seek(0)
            meta_dict = json.load(upload)
            st.session_state.metadata = MultiTableMetadata.load_from_dict(meta_dict)
            st.success("Metadata loaded from JSON dict")

    if "metadata" in st.session_state:
        st.subheader("Field summary")
        descriptions = describe_fields(st.session_state.metadata)
        for name, df in descriptions.items():
            st.markdown(f"**{name}**")
            st.dataframe(df)

        # Download metadata as JSON
        try:
            meta_dict = st.session_state.metadata.to_dict()
        except AttributeError:
            meta_dict = st.session_state.metadata.__dict__

        st.download_button(
            "Download metadata JSON",
            data=json.dumps(meta_dict, indent=2),
            file_name="metadata.json",
            mime="application/json",
        )

        # ---- Editable metadata JSON ----
        with st.expander("Edit metadata JSON"):
            meta_json_str = json.dumps(meta_dict, indent=2)
            edited_json_str = st.text_area(
                "Metadata JSON (edit and click Save to apply)",
                value=meta_json_str,
                height=300,
                key="metadata_json_editor",
            )
            if st.button("Save edited metadata"):
                try:
                    new_meta_dict = json.loads(edited_json_str)
                    st.session_state.metadata = MultiTableMetadata.load_from_dict(new_meta_dict)
                    st.success("Metadata updated from JSON.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Could not load metadata from JSON: {e}")


def render_step_3_constraints():
    st.header("Step 3: Configure constraints")
    if "tables" not in st.session_state:
        st.info("Load data first.")
        return

    render_constraint_builder(st.session_state.tables)

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


def build_eval_metadata_dict(
    metadata: MultiTableMetadata,
    real_data: Dict[str, pd.DataFrame],
    synthetic_data: Dict[str, pd.DataFrame],
) -> Dict:
    """Filter metadata to only tables/relationships present in BOTH real and synthetic data."""
    try:
        meta_dict = metadata.to_dict()
    except AttributeError:
        meta_dict = metadata

    real_keys = set(real_data.keys())
    synth_keys = set(synthetic_data.keys())
    common_tables = real_keys & synth_keys

    # Filter tables
    tables_dict = meta_dict.get("tables", {})
    meta_dict["tables"] = {k: v for k, v in tables_dict.items() if k in common_tables}

    # Filter relationships
    rels = meta_dict.get("relationships", [])
    filtered_rels = []
    for rel in rels:
        p = rel.get("parent_table_name")
        c = rel.get("child_table_name")
        if p in common_tables and c in common_tables:
            filtered_rels.append(rel)
    meta_dict["relationships"] = filtered_rels

    return meta_dict


def _normalize_table_key(filename_base: str) -> str:
    """Normalize a filename base into a table key.

    Strips common suffixes like '_synthetic', '_synth', '_real' so that
    files like 'members_synthetic.csv' map back to 'members'.
    """
    name = filename_base
    for suffix in ["_synthetic", "_synt", "_synth", "_real", "_orig"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name

def render_step_6_evaluate():
    st.header("Step 6: Evaluate synthetic vs original")
    if "metadata" not in st.session_state:
        st.info("Detect metadata first.")
        return

    st.write("Upload real and synthetic data or use the session data.")
    real_uploads = st.file_uploader(
        "Upload real tables",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        key="real_eval_upload",
    )
    synthetic_uploads = st.file_uploader(
        "Upload synthetic tables",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        key="synthetic_eval_upload",
    )

    # Default to in-session data (from earlier steps)
    real_data = st.session_state.get("tables")
    synthetic_data = st.session_state.get("synthetic")

    # If user uploads, override with uploaded data
    if real_uploads:
        real_data = {
            _normalize_table_key(u.name.rsplit(".", 1)[0]): _load_table_from_upload(u)
            for u in real_uploads
        }

    if synthetic_uploads:
        synthetic_data = {
            _normalize_table_key(u.name.rsplit(".", 1)[0]): _load_table_from_upload(u)
            for u in synthetic_uploads
        }

    # Debug info: show what the app thinks the table names are
    with st.expander("Debug: table names seen for evaluation", expanded=False):
        if real_data is not None:
            st.write("Real data tables:", list(real_data.keys()))
        else:
            st.write("Real data tables: None")

        if synthetic_data is not None:
            st.write("Synthetic data tables:", list(synthetic_data.keys()))
        else:
            st.write("Synthetic data tables: None")

        try:
            meta_dict_dbg = st.session_state.metadata.to_dict()
        except AttributeError:
            meta_dict_dbg = st.session_state.metadata
        st.write("Metadata tables:", list(meta_dict_dbg.get("tables", {}).keys()))

    if st.button("Run evaluation"):
        if not real_data or not synthetic_data:
            st.error("Both real and synthetic data are required")
        else:
            # Align metadata and data to avoid KeyError on missing tables
            eval_meta_dict = build_eval_metadata_dict(
                st.session_state.metadata, real_data, synthetic_data
            )
            eval_tables = eval_meta_dict.get("tables", {})

            if not eval_tables:
                st.error(
                    "No common tables between metadata and data for evaluation.\n\n"
                    "Check the debug panel above and make sure:\n"
                    "• Filenames (without extensions / suffixes) match metadata table names, or\n"
                    "• You adjust the metadata 'tables' keys to match your filenames."
                )
            else:
                eval_metadata = MultiTableMetadata.load_from_dict(eval_meta_dict)
                common_keys = set(eval_tables.keys())
                real_eval = {k: real_data[k] for k in common_keys if k in real_data}
                synthetic_eval = {
                    k: synthetic_data[k] for k in common_keys if k in synthetic_data
                }

                diagnostic_report = run_diagnostic(
                    real_data=real_eval,
                    synthetic_data=synthetic_eval,
                    metadata=eval_metadata,
                )
                quality_report = evaluate_quality(
                    real_data=real_eval,
                    synthetic_data=synthetic_eval,
                    metadata=eval_metadata,
                )
                st.session_state.eval_diagnostic = diagnostic_report
                st.session_state.eval_quality = quality_report
                st.success("Evaluation complete")

    # ---- Quality report display ----
    if "eval_quality" in st.session_state:
        report = st.session_state.eval_quality

        st.subheader("Quality score")
        try:
            st.metric("Overall quality", f"{report.get_score():.3f}")
        except Exception:
            pass

        st.subheader("Quality details")


        try:
            raw_props = report.get_properties()
        except Exception as e:
            st.warning(f"Could not retrieve quality properties: {e}")
        else:
            # Normalize whatever get_properties() returns into a list of property names
            if isinstance(raw_props, (list, tuple, set)):
                prop_options = [str(p) for p in raw_props]
            elif isinstance(raw_props, dict):
                prop_options = [str(k) for k in raw_props.keys()]
            elif isinstance(raw_props, pd.DataFrame):
                if "Property" in raw_props.columns:
                    prop_options = raw_props["Property"].astype(str).unique().tolist()
                else:
                    prop_options = raw_props.index.astype(str).tolist()
            elif isinstance(raw_props, pd.Series):
                prop_options = raw_props.astype(str).tolist()
            else:
                try:
                    prop_options = [str(p) for p in list(raw_props)]
                except TypeError:
                    prop_options = [str(raw_props)]

            if len(prop_options) == 0:
                st.info("No quality detail properties available.")
            else:
                prop_choice = st.selectbox("Select quality property", prop_options)
                try:
                    details = report.get_details(property_name=prop_choice)
                except TypeError:
                    details = report.get_details(prop_choice)
                st.dataframe(details)

    # ---- Diagnostic report display ----
    if "eval_diagnostic" in st.session_state:
        diag = st.session_state.eval_diagnostic

        st.subheader("Diagnostic score")
        try:
            st.metric("Overall diagnostic", f"{diag.get_score():.3f}")
        except Exception:
            pass

        st.subheader("Diagnostic details")


        try:
            raw_props = diag.get_properties()
        except Exception as e:
            st.warning(f"Could not retrieve diagnostic properties: {e}")
        else:
            if isinstance(raw_props, (list, tuple, set)):
                prop_options = [str(p) for p in raw_props]
            elif isinstance(raw_props, dict):
                prop_options = [str(k) for k in raw_props.keys()]
            elif isinstance(raw_props, pd.DataFrame):
                if "Property" in raw_props.columns:
                    prop_options = raw_props["Property"].astype(str).unique().tolist()
                else:
                    prop_options = raw_props.index.astype(str).tolist()
            elif isinstance(raw_props, pd.Series):
                prop_options = raw_props.astype(str).tolist()
            else:
                try:
                    prop_options = [str(p) for p in list(raw_props)]
                except TypeError:
                    prop_options = [str(raw_props)]

            if len(prop_options) == 0:
                st.info("No diagnostic detail properties available.")
            else:
                prop_choice = st.selectbox("Select diagnostic property", prop_options)
                try:
                    details = diag.get_details(property_name=prop_choice)
                except TypeError:
                    details = diag.get_details(prop_choice)
                st.dataframe(details)

    # ---- Column distribution comparison ----
    if real_data and synthetic_data:
        st.subheader("Column distribution comparison")
        with st.expander("Select column plot"):
            table_names = list(real_data.keys())
            table_choice = st.selectbox("Table", table_names)
            if table_choice:
                all_cols = list(real_data[table_choice].columns)
                if "metadata" in st.session_state:
                    visualizable_cols = get_visualizable_columns(
                        st.session_state.metadata, table_choice, all_cols
                    )
                else:
                    visualizable_cols = all_cols

                if not visualizable_cols:
                    st.info(
                        "No visualizable columns found for this table "
                        "(they may all be ID-like or unsupported)."
                    )
                else:
                    column_choice = st.selectbox("Column", visualizable_cols)
                    if column_choice:
                        try:
                            fig = get_column_plot(
                                real_data=real_data,
                                synthetic_data=synthetic_data,
                                metadata=st.session_state.metadata,
                                table_name=table_choice,
                                column_name=column_choice,
                            )
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.warning(
                                f"Could not visualize this column (try another): "
                                f"{type(e).__name__}: {e}"
                            )




def main():
    st.set_page_config(page_title="SDV Credit Union Demo", layout="wide")
    st.title("Credit Union Synthetic Data Demo")
    st.write(
        "Step through the workflow to profile, model, generate, "
        "and evaluate multi-table credit union data using SDV."
    )

    # Ordered workflow
    steps = {
        "1. Load data": render_step_1_load,
        "2. Metadata": render_step_2_metadata,
        "3. Constraints": render_step_3_constraints,
        "4. Train model": render_step_4_train,
        "5. Generate data": render_step_5_generate,
        "6. Evaluate": render_step_6_evaluate,
    }
    step_names = list(steps.keys())

    # Initialize step index
    if "current_step_index" not in st.session_state:
        st.session_state.current_step_index = 0

    # Sidebar stepper
    with st.sidebar:
        st.header("Workflow")
        selected_step_name = st.radio(
            "Steps",
            step_names,
            index=st.session_state.current_step_index,
        )

    # Sync the index from sidebar selection
    st.session_state.current_step_index = step_names.index(selected_step_name)

    # Render selected step
    steps[selected_step_name]()

    # Next button at the bottom of the current step
    current_idx = st.session_state.current_step_index
    col_prev, col_next = st.columns([1, 1])

    with col_prev:
        # Placeholder for a "Previous" button if you want later
        pass

    with col_next:
        if current_idx < len(step_names) - 1:
            if st.button("Next ▶"):
                st.session_state.current_step_index = current_idx + 1
                st.rerun()
        else:
            st.info("You’ve reached the final step of the workflow.")


if __name__ == "__main__":
    main()
