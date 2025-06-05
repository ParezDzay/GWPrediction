import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import pymannkendall as mk  # Assuming this package for Mann-Kendall tests

# ──────────────── CONFIG ────────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Well Map Viewer", "📈 Groundwater Data", "📉 Groundwater Level Trends for Wells", "📊 Groundwater Prediction"]
)

file_path = r"Wells detailed data.csv" 
gw_file_path = r"GW data.csv"
output_path = r"GW data (missing filled).csv"
cleaned_outlier_path = r"GW data (missing filled).csv"

# ──────────────── LOAD WELL DATA ────────────────
if page not in ["📈 Groundwater Data", "📉 Groundwater Level Trends for Wells"]:
    if not os.path.exists(file_path):
        st.error("Well CSV file not found.")
        st.stop()
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
    df["Coordinate X"] = pd.to_numeric(df.get("Coordinate X"), errors="coerce")
    df["Coordinate Y"] = pd.to_numeric(df.get("Coordinate Y"), errors="coerce")
    df["Depth (m)"] = pd.to_numeric(df.get("Depth (m)"), errors="coerce")
    df.rename(columns={"Coordinate X": "lat", "Coordinate Y": "lon"}, inplace=True)
    df = df.dropna(subset=["lat", "lon"])

# ──────────────── PAGE ROUTING ────────────────
if page == "Home":
    st.title("Erbil Central Sub-Basin CSB Groundwater Data Analysis")
    st.markdown("""
        **Features**:
        - Explore 20 well data
        - Visualize well locations on a map
        - Groundwater Analysis for 20 wells in CSB in Erbil City
    """)

elif page == "Well Map Viewer":
    st.title("Well Map Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "🗺️ Map View", "🔍 Filters", "⬆️ Upload CSV"])

    with tab3:
        st.subheader("Filter Options")
        col1, col2 = st.columns(2)
        with col1:
            selected_basin = st.multiselect("Select Basin(s):", df["Basin"].unique(), default=df["Basin"].unique())
        with col2:
            selected_district = st.multiselect("Select Sub-District(s):", df["sub district"].unique(), default=df["sub district"].unique())
        filtered_df = df[df["Basin"].isin(selected_basin) & df["sub district"].isin(selected_district)]
        st.success("Filters applied.")

    with tab1:
        st.subheader("Filtered Well Data")
        st.dataframe(filtered_df)

    with tab2:
        st.subheader("Well Locations on Map")
        fig = px.scatter_mapbox(
            filtered_df,
            lat="lat",
            lon="lon",
            color="Basin",
            hover_name="Well Name",
            hover_data={"Depth (m)": True, "Geological Formation": True, "lat": False, "lon": False},
            zoom=10,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Upload Additional Well Data (CSV)")
        uploaded_file = st.file_uploader("Upload a CSV file with matching column format", type="csv")
        if uploaded_file:
            try:
                new_data = pd.read_csv(uploaded_file, on_bad_lines='skip')
                new_data.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in new_data.columns]
                st.write("Preview of uploaded data:")
                st.dataframe(new_data)

                if st.button("Append Uploaded Data to Dataset"):
                    new_data.to_csv(file_path, mode='a', header=False, index=False)
                    st.success("Data uploaded and appended successfully.")
            except Exception as e:
                st.error(f"Failed to upload: {e}")

elif page == "📈 Groundwater Data":
    st.title("Groundwater Data Over 20 Years")

    if not os.path.exists(output_path):
        st.error("Groundwater CSV file (missing filled) not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    try:
        gw_df["Year"] = gw_df["Year"].astype(int)
        gw_df["Months"] = gw_df["Months"].astype(int)
        gw_df["Date"] = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01", format="%Y-%m-%d")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Data with Missing",
            "✏️ Edit Raw Data",
            "✅ Data without Missing",
            "⚠️ Outlier % per Well",
            "🧹 Clean Data from Outliers"
        ])

        well_cols = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

        # Tab 1: Raw data with missing
        with tab1:
            st.subheader("Raw Groundwater Table (with missing)")
            st.dataframe(gw_df, use_container_width=True)

        # Tab 2: Edit and save raw data
        with tab2:
            st.subheader("Edit and Save Raw Groundwater Data")
            edited_df = st.data_editor(gw_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Edited Data"):
                edited_df.to_csv(gw_file_path, index=False)
                st.success("Groundwater data saved successfully.")

        # Tab 3: Data without missing (filled)
        with tab3:
            st.subheader("Groundwater Table (Missing Data Filled)")
            st.dataframe(gw_df, use_container_width=True)

            st.subheader("📊 Groundwater Trends (Depth Plot)")
            wells = well_cols
            selected_wells = st.multiselect("Select wells to display:", wells, default=wells[:3])

            if selected_wells:
                melted = gw_df.melt(id_vars=["Date"], value_vars=selected_wells, var_name="Well", value_name="GW_Level")
                melted["GW_Level"] = -melted["GW_Level"]
                fig = px.line(
                    melted,
                    x="Date",
                    y="GW_Level",
                    color="Well",
                    title="Monthly Groundwater Depth Over Time",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one well.")

        # Tab 4: Outlier percentage before and after cleaning
        with tab4:
            st.subheader("Outlier Percentage per Well (Before and After Cleaning)")

            def calculate_outlier_percent(df_in):
                outlier_pct = {}
                for well in well_cols:
                    series = df_in[well].dropna()
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    percent = (len(outliers) / len(series)) * 100 if len(series) > 0 else 0
                    outlier_pct[well] = round(percent, 2)
                return outlier_pct

            # Outlier % before cleaning
            before_cleaning = calculate_outlier_percent(gw_df)

            # Prepare cleaned data for after cleaning calculation
            cleaned_df = gw_df.copy()
            for well in well_cols:
                series = cleaned_df[well]
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df.loc[(series < lower_bound) | (series > upper_bound), well] = np.nan
            for well in well_cols:
                cleaned_df[well] = cleaned_df[well].interpolate(method='linear', limit_direction='both')

            after_cleaning = calculate_outlier_percent(cleaned_df)

            # Combine into dataframe
            combined_df = pd.DataFrame({
                "Outlier % Before Cleaning": before_cleaning,
                "Outlier % After Cleaning": after_cleaning
            })

            st.dataframe(combined_df, use_container_width=True)

        # Tab 5: Clean data from outliers and interpolate
        with tab5:
            st.subheader("Clean Data by Removing Outliers and Interpolating")

            st.dataframe(cleaned_df, use_container_width=True)

            if st.button("Save Cleaned Data to CSV"):
                try:
                    cleaned_df.to_csv(cleaned_outlier_path, index=False)
                    st.success(f"Cleaned data saved successfully to:\n{cleaned_outlier_path}")
                except PermissionError:
                    st.error("Permission denied: Please close the file if it is open and try again.")

    except Exception as e:
        st.error(f"Error loading groundwater data: {e}")

elif page == "📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")

    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    gw_df["Date"] = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01", format="%Y-%m-%d")

    wells = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Plots", "📊 MK & Sen’s Slope", "🧪 MMK Analysis", "💡 ITA Analysis"])

    with tab1:
        st.subheader("Groundwater Level Trends")

        color_sequence = px.colors.qualitative.D3  # Good distinct colors

        for i, well in enumerate(wells):
            fig = px.line(
                gw_df,
                x="Date",
                y=well,
                title=f"{well} Groundwater Level Trend",
                markers=True,
                color_discrete_sequence=[color_sequence[i % len(color_sequence)]],
                labels={"Date": "Date", well: "Groundwater Level (m)"}
            )
            fig.update_traces(
                line=dict(width=1, shape='spline', smoothing=1.3),
                marker=dict(size=3, symbol='circle')
            )
            fig.update_layout(
                yaxis_title="Groundwater Level (m)",
                xaxis_title="Years",
                yaxis_autorange='reversed',
                plot_bgcolor='rgba(245,245,245,1)',
                hovermode='x unified',
                margin=dict(l=40, r=20, t=40, b=40),
                font=dict(size=13),
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    def run_mann_kendall(series):
        try:
            result = mk.original_test(series.dropna())
            return result
        except Exception:
            return None

    def run_modified_mk(series):
        try:
            result = mk.hamed_rao_modification_test(series.dropna())
            return result
        except Exception:
            return None

    with tab2:
        st.subheader("Mann-Kendall & Sen’s Slope Analysis")
        mk_results = {}
        for well in wells:
            res = run_mann_kendall(gw_df[well])
            if res:
                mk_results[well] = {
                    "Trend": res.trend,
                    "p-value": round(res.p, 4),
                    "Sen’s slope": round(res.slope, 4)
                }
            else:
                mk_results[well] = {
                    "Trend": "N/A",
                    "p-value": "N/A",
                    "Sen’s slope": "N/A"
                }
        mk_df = pd.DataFrame(mk_results).T
        st.dataframe(mk_df.style.format({"p-value": "{:.4f}", "Sen’s slope": "{:.4f}"}), use_container_width=True)

    with tab3:
        st.subheader("Modified Mann-Kendall Test Results")
        mmk_results = {}
        for well in wells:
            res = run_modified_mk(gw_df[well])
            if res:
                mmk_results[well] = {
                    "Trend": res.trend,
                    "p-value": round(res.p, 4),
                    "Sen’s slope": round(res.slope, 4)
                }
            else:
                mmk_results[well] = {
                    "Trend": "N/A",
                    "p-value": "N/A",
                    "Sen’s slope": "N/A"
                }
        mmk_df = pd.DataFrame(mmk_results).T
        st.dataframe(mmk_df.style.format({"p-value": "{:.4f}", "Sen’s slope": "{:.4f}"}), use_container_width=True)
    with tab4:
         st.subheader("ITA Analysis – Trend Metrics")

    ita_results = []

    for well in wells:
        series = gw_df[well].dropna()
        if len(series) < 2:
            continue  # Skip wells with insufficient data

        x = np.arange(len(series))
        y = series.values

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Calculate Sand and Scrit
        std_dev = np.std(y)
        sand = 0.5 * std_dev
        scrit = 0.95 * std_dev  # Adjust factor as needed

        ita_results.append({
            "Well": well,
            "Slope": round(slope, 4),
            "S": round(sand, 4),
            "Scrit": round(scrit, 4),
            "R²": round(r_squared, 4)
        })

    ita_df = pd.DataFrame(ita_results)
    st.dataframe(ita_df, use_container_width=True)

# ──────────────── GROUNDWATER PREDICTION ────────────────
elif page == "📊 Groundwater Prediction":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA

    st.title("📊 Groundwater Prediction Models")
    tab1, tab2 = st.tabs(["🔮 ANN Prediction", "📉 ARIMA"])

    @st.cache_data
    def load_gw_data():
        if not os.path.exists(cleaned_outlier_path):
            st.error("Cleaned groundwater data CSV not found.")
            return None
        data = pd.read_csv(cleaned_outlier_path)
        data["Date"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Months"].astype(str) + "-01", format="%Y-%m-%d")
        data.set_index("Date", inplace=True)
        return data

    gw_data = load_gw_data()
    if gw_data is None:
        st.stop()

    wells = [col for col in gw_data.columns if col not in ["Year", "Months"]]

    def prepare_ann_data(series, n_steps=12):
        """Prepare supervised data for ANN with lag features"""
        X, y = [], []
        for i in range(len(series) - n_steps):
            X.append(series[i:i+n_steps])
            y.append(series[i+n_steps])
        return np.array(X), np.array(y)

    with tab1:
        st.subheader("Artificial Neural Network (ANN) Prediction")

        selected_well = st.selectbox("Select Well for ANN Prediction", wells)
        n_steps = st.slider("Number of Lag Months (Input window size)", min_value=6, max_value=24, value=12)

        series = gw_data[selected_well].dropna().values.reshape(-1, 1)

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        series_scaled = scaler.fit_transform(series)

        X, y = prepare_ann_data(series_scaled.flatten(), n_steps)

        # Split train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build ANN model
        model = Sequential([
            Dense(50, activation='relu', input_shape=(n_steps,)),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        if st.button("Train ANN Model"):
            with st.spinner("Training ANN model..."):
                model.fit(X_train, y_train, epochs=100, verbose=0)
                st.success("Training completed.")

                # Predict
                y_pred = model.predict(X_test).flatten()
                y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

                mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                st.write(f"Test MSE: {mse:.4f}")

                # Plot results
                plt.figure(figsize=(10,4))
                plt.plot(y_test_rescaled, label='Actual')
                plt.plot(y_pred_rescaled, label='Predicted')
                plt.title(f"ANN Prediction for {selected_well}")
                plt.xlabel("Time Step")
                plt.ylabel("Groundwater Level")
                plt.legend()
                st.pyplot(plt)

    with tab2:
        st.subheader("ARIMA Prediction")

        selected_well_arima = st.selectbox("Select Well for ARIMA Prediction", wells, key="arima_well")
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Difference order (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)

        series_arima = gw_data[selected_well_arima].dropna()

        if st.button("Run ARIMA Model"):
            with st.spinner("Fitting ARIMA model..."):
                try:
                    model_arima = ARIMA(series_arima, order=(p,d,q))
                    model_fit = model_arima.fit()
                    forecast = model_fit.forecast(steps=12)

                    # Plot historical and forecast
                    plt.figure(figsize=(10,4))
                    plt.plot(series_arima.index, series_arima.values, label='Historical')
                    future_index = pd.date_range(series_arima.index[-1] + pd.offsets.MonthBegin(),
                                                 periods=12, freq='MS')
                    plt.plot(future_index, forecast, label='Forecast', linestyle='--')
                    plt.title(f"ARIMA Forecast for {selected_well_arima}")
                    plt.xlabel("Date")
                    plt.ylabel("Groundwater Level")
                    plt.legend()
                    st.pyplot(plt)

                    st.write(forecast)

                except Exception as e:
                    st.error(f"ARIMA model failed: {e}")
