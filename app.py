import plotly.express as px
import streamlit as st
import numpy as np
from models import ModelParams, Model, ModelSol
from solvers import ModelSolver

st.set_page_config(layout="wide")

@st.cache_resource
def solve_benchmark_model():
    data_file_name = "data_2017(gvc_consistent).npz"
    params = ModelParams.load_from_npz(data_file_name)
    params.gamma = np.swapaxes(params.gamma, 1, 2)
    model = Model(params)
    solver = ModelSolver(model)
    solver.solve()
    return model.sol, params


# Counterfactual model solver (cached)
@st.cache_resource
def solve_counterfactual_model(params, importers, exporters, sectors, tariff_rate):
    from copy import deepcopy
    country_names = list(params.country_list)
    sector_names = list(params.sector_list)
    tilde_tau_1 = params.tilde_tau.copy()
    for i in [country_names.index(x) for x in importers]:
        for j in [country_names.index(x) for x in exporters]:
            for s in [sector_names.index(x) for x in sectors]:
                if i != j:
                    tilde_tau_1[i, j, s] = params.tilde_tau[i, j, s] + (tariff_rate / 100.0)
    cf_params = deepcopy(params)
    cf_params.tilde_tau = tilde_tau_1
    cf_model = Model(cf_params)
    cf_solver = ModelSolver(cf_model)
    cf_solver.solve()
    return cf_model.sol


# Load and solve the model (cached)
sol, params = solve_benchmark_model()

country_names = list(params.country_list)
sector_names = list(params.sector_list)

# For UI: priority order for selection
priority_countries = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "ITA", "CAN", "KOR", "IND",
    "ESP", "NLD", "BEL", "SWE", "RUS", "BRA", "MEX", "AUS"
]
country_names_sorted = [c for c in priority_countries if c in country_names] + [c for c in country_names if c not in priority_countries]

st.title("Model Output Explorer")



# Remove global figure size adjustment block here

from copy import deepcopy

st.header("Model Selection")
# Remove model_view radio; set directly to "Compare Two Models"
model_view = "Compare Two Models"

def get_model_solution(model_type, params, which):
    if model_type == "Benchmark":
        return sol
    else:
        # Only use suffixes for Compare mode, i.e., when which is 1 or 2
        suffix = "_1" if which == 1 else "_2"
        cf_importer_key = f"cf_importer_multiselect{suffix}"
        cf_exporter_key = f"cf_exporter_multiselect{suffix}"
        cf_sector_key = f"cf_sector_multiselect{suffix}"
        # Use descriptive button/label text for compare mode
        if which == 1:
            importer_all_label = "Counterfactual 1: Select ALL Importers"
            importer_none_label = "Counterfactual 1: Remove ALL Importers"
            importer_multiselect_label = "Counterfactual 1: Select Importer(s)"
            exporter_all_label = "Counterfactual 1: Select ALL Exporters"
            exporter_none_label = "Counterfactual 1: Remove ALL Exporters"
            exporter_multiselect_label = "Counterfactual 1: Select Exporter(s)"
            sector_all_label = "Counterfactual 1: Select ALL Sectors"
            sector_none_label = "Counterfactual 1: Remove ALL Sectors"
            sector_multiselect_label = "Counterfactual 1: Select Sector(s)"
            tariff_label = "Counterfactual 1: Tariff Rate (%)"
            warning_label = "Please select at least one importer, one exporter, and one sector for Counterfactual 1."
        else:
            importer_all_label = "Counterfactual 2: Select ALL Importers"
            importer_none_label = "Counterfactual 2: Remove ALL Importers"
            importer_multiselect_label = "Counterfactual 2: Select Importer(s)"
            exporter_all_label = "Counterfactual 2: Select ALL Exporters"
            exporter_none_label = "Counterfactual 2: Remove ALL Exporters"
            exporter_multiselect_label = "Counterfactual 2: Select Exporter(s)"
            sector_all_label = "Counterfactual 2: Select ALL Sectors"
            sector_none_label = "Counterfactual 2: Remove ALL Sectors"
            sector_multiselect_label = "Counterfactual 2: Select Sector(s)"
            tariff_label = "Counterfactual 2: Tariff Rate (%)"
            warning_label = "Please select at least one importer, one exporter, and one sector for Counterfactual 2."
        cols = st.columns(2)
        with cols[0]:
            if st.button(importer_all_label, key=f"cf_select_all_importers{suffix}"):
                st.session_state[cf_importer_key] = country_names_sorted
        with cols[1]:
            if st.button(importer_none_label, key=f"cf_remove_all_importers{suffix}"):
                st.session_state[cf_importer_key] = []
        cf_importers = st.multiselect(importer_multiselect_label, country_names_sorted, default=[], key=cf_importer_key)
        cols = st.columns(2)
        with cols[0]:
            if st.button(exporter_all_label, key=f"cf_select_all_exporters{suffix}"):
                st.session_state[cf_exporter_key] = country_names_sorted
        with cols[1]:
            if st.button(exporter_none_label, key=f"cf_remove_all_exporters{suffix}"):
                st.session_state[cf_exporter_key] = []
        cf_exporters = st.multiselect(exporter_multiselect_label, country_names_sorted, default=[], key=cf_exporter_key)
        cols = st.columns(2)
        with cols[0]:
            if st.button(sector_all_label, key=f"cf_select_all_sectors{suffix}"):
                st.session_state[cf_sector_key] = sector_names
        with cols[1]:
            if st.button(sector_none_label, key=f"cf_remove_all_sectors{suffix}"):
                st.session_state[cf_sector_key] = []
        cf_sectors = st.multiselect(sector_multiselect_label, sector_names, default=sector_names, key=cf_sector_key)
        tariff_rate = st.slider(tariff_label, min_value=0, max_value=100, value=20, step=1, key=f"tariff_rate{suffix}")
        if cf_importers and cf_exporters and cf_sectors:
            # Ensure selected countries are in model order
            cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
            cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
            return solve_counterfactual_model(params, cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_rate)
        else:
            st.warning(warning_label)
            return None

sol_to_show = None
value = None
if model_view == "Compare Two Models":
    # Ensure selector lists are always defined
    selected_importers = []
    selected_exporters = []
    selected_sectors = []
    selected_countries = []
    model1_type = st.radio("Model 1", ["Benchmark", "Counterfactual"], horizontal=True, key="model1_type")
    model2_type = st.radio("Model 2", ["Benchmark", "Counterfactual"], horizontal=True, key="model2_type")
    sol1 = get_model_solution(model1_type, params, which=1)
    sol2 = get_model_solution(model2_type, params, which=2)
    # Header for variable/visualization section
    st.header("Variables and Visualization")
    variable = None
    if sol1 is not None and sol2 is not None:
        sol1_dict = sol1.__dict__
        sol2_dict = sol2.__dict__
        variable_keys = list(set(sol1_dict.keys()) & set(sol2_dict.keys()))
        variable = st.selectbox("Choose an output variable", variable_keys)
        val1 = sol1_dict[variable]
        val2 = sol2_dict[variable]
        if variable.endswith("_hat"):
            value = 100 * (val2 - val1) / (np.abs(val1) + 1e-8)
        else:
            value = val2
        sol_to_show = None  # triggers the correct display logic below
    else:
        value = None
elif model_view == "Counterfactual":
    st.markdown("### Counterfactual Tariff Scenario")

    # Counterfactual construction section (no suffixes in keys/labels/buttons)
    cf_importer_key = "cf_importer_multiselect"
    cols = st.columns(2)
    with cols[0]:
        if st.button("Select ALL Importers", key="cf_select_all_importers"):
            st.session_state[cf_importer_key] = country_names_sorted
    with cols[1]:
        if st.button("Remove ALL Importers", key="cf_remove_all_importers"):
            st.session_state[cf_importer_key] = []
    cf_importers = st.multiselect("Select Importer(s)", country_names_sorted, default=[], key=cf_importer_key)

    cf_exporter_key = "cf_exporter_multiselect"
    cols = st.columns(2)
    with cols[0]:
        if st.button("Select ALL Exporters", key="cf_select_all_exporters"):
            st.session_state[cf_exporter_key] = country_names_sorted
    with cols[1]:
        if st.button("Remove ALL Exporters", key="cf_remove_all_exporters"):
            st.session_state[cf_exporter_key] = []
    cf_exporters = st.multiselect("Select Exporter(s)", country_names_sorted, default=[], key=cf_exporter_key)

    cf_sector_key = "cf_sector_multiselect"
    cols = st.columns(2)
    with cols[0]:
        if st.button("Select ALL Sectors", key="cf_select_all_sectors"):
            st.session_state[cf_sector_key] = sector_names
    with cols[1]:
        if st.button("Remove ALL Sectors", key="cf_remove_all_sectors"):
            st.session_state[cf_sector_key] = []
    cf_sectors = st.multiselect("Select Sector(s)", sector_names, default=sector_names, key=cf_sector_key)

    tariff_rate = st.slider("Tariff Rate (%)", min_value=0, max_value=100, value=20, step=1)

    if cf_importers and cf_exporters and cf_sectors:
        cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
        cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
        sol_to_show = solve_counterfactual_model(params, cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_rate)
    else:
        st.warning("Please select at least one importer, one exporter, and one sector for the counterfactual.")
        sol_to_show = None
    value = None
else:
    sol_to_show = sol
    value = None  # The normal visualization will assign 'value' later as usual.

if model_view == "Compare Two Models":
    # The header and selectbox have already been moved above.
    if value is not None:
        # Show variable explanation if available
        variable_descriptions = {
            "w_hat": r"""$\hat{w}$: shape $(N,)$, index ($i$)<br>Percentage change in nominal wage in country $i$.""",
            "c_hat": r"""$\hat{c}$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in the unit cost of input bundles for producing output in country $i$, sector $s$.""",
            "Pf_hat": r"""$\hat{P}_f$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of final goods in country $i$, sector $s$.""",
            "Pm_hat": r"""$\hat{P}_m$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of intermediate goods in country $i$, sector $s$.""",
            "pif_hat": r"""$\hat{\pi}_f$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used for final consumption.""",
            "pim_hat": r"""$\hat{\pi}_m$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used as intermediate inputs.""",
            "Xf_prime": r"""$X_f'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used for final consumption, under model 2.""",
            "Xm_prime": r"""$X_m'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used as intermediate inputs, under model 2.""",
            "X_prime": r"""$X'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods in sector $s$ under model 2, i.e., the sum of $X_f'$ and $X_m'$.""",
            "Xf_prod_prime": r"""$X_{f,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used for final consumption, under model 2.""",
            "Xm_prod_prime": r"""$X_{m,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used as intermediate inputs, under model 2.""",
            "X_prod_prime": r"""$X_{\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Total production by country $n$ of goods in sector $s$ under model 2, i.e., the sum of $X_{f,\text{prod}}'$ and $X_{m,\text{prod}}'$.""",
            "p_index": r"""$p_{\text{index}}$: shape $(N,)$, index ($i$)<br>Change in the aggregate price index in country $i$.""",
            "real_w": r"""real_w: shape $(N,)$, index ($i$)<br>Percentage change in real wage in country $i$ (i.e., nominal wage deflated by the price index).""",
            "D_prime": r"""$D'$: shape $(N,)$, index ($i$)<br>Trade deficit or surplus in country $i$ under model 2 (the counterfactual scenario)."""
        }
        if variable in variable_descriptions:
            st.markdown(variable_descriptions[variable], unsafe_allow_html=True)
        else:
            st.write(f"Variable shape: {np.shape(value)} (showing % change from Model 1 to Model 2)")

        if isinstance(value, np.ndarray) and value.ndim == 2:
            # 2D: Only show country-sector selectors
            country_key = "country_multiselect"
            cols = st.columns(2)
            with cols[0]:
                if st.button("Select ALL Countries", key="select_all_countries"):
                    st.session_state[country_key] = country_names_sorted
            with cols[1]:
                if st.button("Remove ALL Countries", key="remove_all_countries"):
                    st.session_state[country_key] = []
            selected_countries = st.multiselect("Countries", country_names_sorted, default=[], key=country_key)

            sector_key = "sector_multiselect"
            cols = st.columns(2)
            with cols[0]:
                if st.button("Select ALL Sectors", key="select_all_sectors"):
                    st.session_state[sector_key] = sector_names
            with cols[1]:
                if st.button("Remove ALL Sectors", key="remove_all_sectors"):
                    st.session_state[sector_key] = []
            selected_sectors = st.multiselect("Sectors", sector_names, default=sector_names, key=sector_key)

            if selected_countries and selected_sectors:
                st.markdown("### Figure Size Adjustment")
                st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
                st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
                fig_width = st.session_state["fig_width"]
                fig_height = st.session_state["fig_height"]
                selected_countries_in_model_order = [c for c in country_names if c in selected_countries]
                y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
                for country in selected_countries_in_model_order:
                    c_idx = country_names.index(country)
                    bars = []
                    labels = []
                    for sector in selected_sectors:
                        s_idx = sector_names.index(sector)
                        val = value[c_idx, s_idx]
                        bars.append(val)
                        labels.append(sector)
                    unique_key = f"{country}_2d"
                    fig = px.bar(
                        x=labels,
                        y=bars,
                        labels={'x': "Sector", 'y': y_label},
                        title=f"{country}: Selected Sectors",
                        height=fig_height,
                        width=fig_width
                    )
                    fig.update_traces(hovertemplate=f'Sector: %{{x}}<br>{y_label}: %{{y:.2f}}')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        xaxis_title_font=dict(size=20, color='black'),
                        yaxis_title_font=dict(size=20, color='black')
                    )
                    st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("No countries or sectors selected.")

        elif isinstance(value, np.ndarray) and value.ndim == 3:
            # 3D: Only show importer/exporter/sector selectors
            importer_key = "importer_multiselect"
            cols = st.columns(2)
            with cols[0]:
                if st.button("Select ALL Importer Countries", key="select_all_importers"):
                    st.session_state[importer_key] = country_names_sorted
            with cols[1]:
                if st.button("Remove ALL Importer Countries", key="remove_all_importers"):
                    st.session_state[importer_key] = []
            selected_importers = st.multiselect("Importer Countries", country_names_sorted, default=[], key=importer_key)

            exporter_key = "exporter_multiselect"
            cols = st.columns(2)
            with cols[0]:
                if st.button("Select ALL Exporter Countries", key="select_all_exporters"):
                    st.session_state[exporter_key] = country_names_sorted
            with cols[1]:
                if st.button("Remove ALL Exporter Countries", key="remove_all_exporters"):
                    st.session_state[exporter_key] = []
            selected_exporters = st.multiselect("Exporter Countries", country_names_sorted, default=[], key=exporter_key)

            sector_3d_key = "sector_multiselect_3d"
            cols = st.columns(2)
            with cols[0]:
                if st.button("Select ALL Sectors (3D)", key="select_all_sectors_3d"):
                    st.session_state[sector_3d_key] = sector_names
            with cols[1]:
                if st.button("Remove ALL Sectors (3D)", key="remove_all_sectors_3d"):
                    st.session_state[sector_3d_key] = []
            selected_sectors = st.multiselect("Sectors", sector_names, default=sector_names, key=sector_3d_key)

            if selected_importers and selected_exporters and selected_sectors:
                st.markdown("### Figure Size Adjustment")
                st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
                st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
                fig_width = st.session_state["fig_width"]
                fig_height = st.session_state["fig_height"]
                selected_importers_in_model_order = [c for c in country_names if c in selected_importers]
                selected_exporters_in_model_order = [c for c in country_names if c in selected_exporters]
                y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
                for importer in selected_importers_in_model_order:
                    for exporter in selected_exporters_in_model_order:
                        i_idx = country_names.index(importer)
                        e_idx = country_names.index(exporter)
                        bars = []
                        labels = []
                        for sector in selected_sectors:
                            s_idx = sector_names.index(sector)
                            bars.append(value[i_idx, e_idx, s_idx])
                            labels.append(sector)
                        unique_key = f"{importer}_{exporter}_3d"
                        fig = px.bar(
                            x=labels,
                            y=bars,
                            labels={'x': "Sector", 'y': y_label},
                            title=f"{importer} (Importer) — {exporter} (Exporter): Selected Sectors",
                            height=fig_height,
                            width=fig_width
                        )
                        fig.update_traces(hovertemplate=f'Sector: %{{x}}<br>{y_label}: %{{y:.2f}}')
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            xaxis_title_font=dict(size=20, color='black'),
                            yaxis_title_font=dict(size=20, color='black')
                        )
                        st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("No importers, exporters, or sectors selected for 3D variable.")

        elif isinstance(value, np.ndarray) and value.ndim == 1:
            # Shape (N,) or (S,): multi-select with "Select ALL" buttons for countries or sectors
            if value.shape[0] == len(country_names):
                names = country_names_sorted
                label = "Countries"
                key_prefix = "country"
                default_list = []
            else:
                names = sector_names
                label = "Sectors"
                key_prefix = "sector"
                default_list = names

            one_d_key = f"{key_prefix}_multiselect"
            cols = st.columns(2)
            with cols[0]:
                if st.button(f"Select ALL {label}", key=f"select_all_{key_prefix}"):
                    if label == "Countries":
                        st.session_state[one_d_key] = country_names_sorted
                    else:
                        st.session_state[one_d_key] = names
            with cols[1]:
                if st.button(f"Remove ALL {label}", key=f"remove_all_{key_prefix}"):
                    st.session_state[one_d_key] = []
            # For sectors (1D), default to all names; for countries, default to []
            selected_items = st.multiselect(label, names, default=default_list, key=one_d_key)

            if label == "Countries":
                selected_items_in_model_order = [c for c in country_names if c in selected_items]
                bars = []
                labels = []
                for name in selected_items_in_model_order:
                    idx = country_names.index(name)
                    bars.append(value[idx])
                    labels.append(name)
            else:
                bars = []
                labels = []
                for name in selected_items:
                    idx = names.index(name)
                    bars.append(value[idx])
                    labels.append(name)

            # Insert Figure Size Adjustment just before if bars:
            st.markdown("### Figure Size Adjustment")
            st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
            st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
            fig_width = st.session_state["fig_width"]
            fig_height = st.session_state["fig_height"]

            if bars:
                unique_key = f"1d_{label}_{'_'.join(labels)}"
                y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
                fig = px.bar(
                    x=labels,
                    y=bars,
                    labels={'x': label, 'y': y_label},
                    title="Selected Values",
                    height=fig_height,
                    width=fig_width
                )
                fig.update_traces(hovertemplate=f'{label}: %{{x}}<br>{y_label}: %{{y:.2f}}')
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title_font=dict(size=20, color='black'),
                    yaxis_title_font=dict(size=20, color='black')
                )
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.info(f"No {label.lower()} selected.")

        elif isinstance(value, np.ndarray) and value.ndim == 0:
            # Scalar
            st.write(f"Value: **{value.item():.4f}**")

        else:
            # Other cases: show as is
            st.write("Value:")
            st.write(value)
    else:
        st.info("No model solution available to display.")
elif sol_to_show is not None:
    # Header for variable/visualization section
    st.header("Variables and Visualization")
    # Get available outputs from sol_to_show
    sol_dict = sol_to_show.__dict__

    # User: pick a variable to inspect
    variable = st.selectbox("Choose an output variable", list(sol_dict.keys()))

    # Show variable explanation if available
    variable_descriptions = {
        "w_hat": r"""$\hat{w}$: shape $(N,)$, index ($i$)<br>Percentage change in nominal wage in country $i$.""",
        "c_hat": r"""$\hat{c}$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in the unit cost of input bundles for producing output in country $i$, sector $s$.""",
        "Pf_hat": r"""$\hat{P}_f$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of final goods in country $i$, sector $s$.""",
        "Pm_hat": r"""$\hat{P}_m$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of intermediate goods in country $i$, sector $s$.""",
        "pif_hat": r"""$\hat{\pi}_f$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used for final consumption.""",
        "pim_hat": r"""$\hat{\pi}_m$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used as intermediate inputs.""",
        "Xf_prime": r"""$X_f'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used for final consumption, under model 2.""",
        "Xm_prime": r"""$X_m'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used as intermediate inputs, under model 2.""",
        "X_prime": r"""$X'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods in sector $s$ under model 2, i.e., the sum of $X_f'$ and $X_m'$.""",
        "Xf_prod_prime": r"""$X_{f,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used for final consumption, under model 2.""",
        "Xm_prod_prime": r"""$X_{m,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used as intermediate inputs, under model 2.""",
        "X_prod_prime": r"""$X_{\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Total production by country $n$ of goods in sector $s$ under model 2, i.e., the sum of $X_{f,\text{prod}}'$ and $X_{m,\text{prod}}'$.""",
        "p_index": r"""$p_{\text{index}}$: shape $(N,)$, index ($i$)<br>Change in the aggregate price index in country $i$.""",
        "real_w": r"""real_w: shape $(N,)$, index ($i$)<br>Percentage change in real wage in country $i$ (i.e., nominal wage deflated by the price index).""",
        "D_prime": r"""$D'$: shape $(N,)$, index ($i$)<br>Trade deficit or surplus in country $i$ under model 2 (the counterfactual scenario)."""
    }
    if variable in variable_descriptions:
        st.markdown(variable_descriptions[variable], unsafe_allow_html=True)

    value = sol_dict[variable]

    # Display variable shape for clarity
    st.write(f"Variable shape: {np.shape(value)}")



    # Only show selectors if value is array and for the relevant cases
    # Move selectors inside the correct ndim blocks, and show only the relevant selectors
    if isinstance(value, np.ndarray) and value.ndim == 2:
        # 2D: Only country-sector selectors
        country_key = "country_multiselect"
        cols = st.columns(2)
        with cols[0]:
            if st.button("Select ALL Countries", key="select_all_countries"):
                st.session_state[country_key] = country_names_sorted
        with cols[1]:
            if st.button("Remove ALL Countries", key="remove_all_countries"):
                st.session_state[country_key] = []
        selected_countries = st.multiselect("Countries", country_names_sorted, default=[], key=country_key)

        sector_key = "sector_multiselect"
        cols = st.columns(2)
        with cols[0]:
            if st.button("Select ALL Sectors", key="select_all_sectors"):
                st.session_state[sector_key] = sector_names
        with cols[1]:
            if st.button("Remove ALL Sectors", key="remove_all_sectors"):
                st.session_state[sector_key] = []
        selected_sectors = st.multiselect("Sectors", sector_names, default=sector_names, key=sector_key)

        if selected_countries and selected_sectors:
            st.markdown("### Figure Size Adjustment")
            st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
            st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
            fig_width = st.session_state["fig_width"]
            fig_height = st.session_state["fig_height"]
            selected_countries_in_model_order = [c for c in country_names if c in selected_countries]
            y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
            for country in selected_countries_in_model_order:
                c_idx = country_names.index(country)
                bars = []
                labels = []
                for sector in selected_sectors:
                    s_idx = sector_names.index(sector)
                    val = value[c_idx, s_idx]
                    bars.append(val)
                    labels.append(sector)
                fig = px.bar(
                    x=labels,
                    y=bars,
                    labels={'x': "Sector", 'y': y_label},
                    title=f"{country}: Selected Sectors",
                    height=fig_height,
                    width=fig_width
                )
                fig.update_traces(hovertemplate=f'Sector: %{{x}}<br>{y_label}: %{{y:.2f}}')
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title_font=dict(size=20, color='black'),
                    yaxis_title_font=dict(size=20, color='black')
                )
                st.plotly_chart(fig, use_container_width=False)
        else:
            st.info("No countries or sectors selected.")

    elif isinstance(value, np.ndarray) and value.ndim == 3:
        # 3D: Only importer/exporter/sector selectors
        importer_key = "importer_multiselect"
        cols = st.columns(2)
        with cols[0]:
            if st.button("Select ALL Importer Countries", key="select_all_importers"):
                st.session_state[importer_key] = country_names_sorted
        with cols[1]:
            if st.button("Remove ALL Importer Countries", key="remove_all_importers"):
                st.session_state[importer_key] = []
        selected_importers = st.multiselect("Importer Countries", country_names_sorted, default=[], key=importer_key)

        exporter_key = "exporter_multiselect"
        cols = st.columns(2)
        with cols[0]:
            if st.button("Select ALL Exporter Countries", key="select_all_exporters"):
                st.session_state[exporter_key] = country_names_sorted
        with cols[1]:
            if st.button("Remove ALL Exporter Countries", key="remove_all_exporters"):
                st.session_state[exporter_key] = []
        selected_exporters = st.multiselect("Exporter Countries", country_names_sorted, default=[], key=exporter_key)

        sector_3d_key = "sector_multiselect_3d"
        cols = st.columns(2)
        with cols[0]:
            if st.button("Select ALL Sectors (3D)", key="select_all_sectors_3d"):
                st.session_state[sector_3d_key] = sector_names
        with cols[1]:
            if st.button("Remove ALL Sectors (3D)", key="remove_all_sectors_3d"):
                st.session_state[sector_3d_key] = []
        selected_sectors = st.multiselect("Sectors", sector_names, default=sector_names, key=sector_3d_key)

        if selected_importers and selected_exporters and selected_sectors:
            st.markdown("### Figure Size Adjustment")
            st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
            st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
            fig_width = st.session_state["fig_width"]
            fig_height = st.session_state["fig_height"]
            selected_importers_in_model_order = [c for c in country_names if c in selected_importers]
            selected_exporters_in_model_order = [c for c in country_names if c in selected_exporters]
            y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
            for importer in selected_importers_in_model_order:
                for exporter in selected_exporters_in_model_order:
                    i_idx = country_names.index(importer)
                    e_idx = country_names.index(exporter)
                    bars = []
                    labels = []
                    for sector in selected_sectors:
                        s_idx = sector_names.index(sector)
                        bars.append(value[i_idx, e_idx, s_idx])
                        labels.append(sector)
                    fig = px.bar(
                        x=labels,
                        y=bars,
                        labels={'x': "Sector", 'y': y_label},
                        title=f"{importer} (Importer) — {exporter} (Exporter): Selected Sectors",
                        height=fig_height,
                        width=fig_width
                    )
                    fig.update_traces(hovertemplate=f'Sector: %{{x}}<br>{y_label}: %{{y:.2f}}')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        xaxis_title_font=dict(size=20, color='black'),
                        yaxis_title_font=dict(size=20, color='black')
                    )
                    st.plotly_chart(fig, use_container_width=False)
        else:
            st.info("No importers, exporters, or sectors selected for 3D variable.")

    elif isinstance(value, np.ndarray) and value.ndim == 1:
        # Shape (N,) or (S,): multi-select with "Select ALL" buttons for countries or sectors
        if value.shape[0] == len(country_names):
            names = country_names_sorted
            label = "Countries"
            key_prefix = "country"
            default_list = []
        else:
            names = sector_names
            label = "Sectors"
            key_prefix = "sector"
            default_list = names

        one_d_key = f"{key_prefix}_multiselect"
        cols = st.columns(2)
        with cols[0]:
            if st.button(f"Select ALL {label}", key=f"select_all_{key_prefix}"):
                if label == "Countries":
                    st.session_state[one_d_key] = country_names_sorted
                else:
                    st.session_state[one_d_key] = names
        with cols[1]:
            if st.button(f"Remove ALL {label}", key=f"remove_all_{key_prefix}"):
                st.session_state[one_d_key] = []
        # For sectors (1D), default to all names; for countries, default to []
        selected_items = st.multiselect(label, names, default=default_list, key=one_d_key)

        if label == "Countries":
            selected_items_in_model_order = [c for c in country_names if c in selected_items]
            bars = []
            labels = []
            for name in selected_items_in_model_order:
                idx = country_names.index(name)
                bars.append(value[idx])
                labels.append(name)
        else:
            bars = []
            labels = []
            for name in selected_items:
                idx = names.index(name)
                bars.append(value[idx])
                labels.append(name)

        # Insert Figure Size Adjustment just before if bars:
        st.markdown("### Figure Size Adjustment")
        st.session_state["fig_width"] = st.slider("Figure Width", min_value=400, max_value=2000, value=st.session_state.get("fig_width", 1600), step=100)
        st.session_state["fig_height"] = st.slider("Figure Height", min_value=300, max_value=1000, value=st.session_state.get("fig_height", 700), step=50)
        fig_width = st.session_state["fig_width"]
        fig_height = st.session_state["fig_height"]

        if bars:
            y_label = f"{variable} (% Change)" if variable.endswith("_hat") else variable
            fig = px.bar(
                x=labels,
                y=bars,
                labels={'x': label, 'y': y_label},
                title="Selected Values",
                height=fig_height,
                width=fig_width
            )
            fig.update_traces(hovertemplate=f'{label}: %{{x}}<br>{y_label}: %{{y:.2f}}')
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis_title_font=dict(size=20, color='black'),
                yaxis_title_font=dict(size=20, color='black')
            )
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.info(f"No {label.lower()} selected.")

    elif isinstance(value, np.ndarray) and value.ndim == 0:
        # Scalar
        st.write(f"Value: **{value.item():.4f}**")

    else:
        # Other cases: show as is
        st.write("Value:")
        st.write(value)
else:
    st.info("No model solution available to display.")
    