import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import ruptures as rpt
from PIL import Image
from autots import AutoTS
from autots.tools.shaping import infer_frequency
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot.cointegration import phillips_ouliaris
from scipy.stats import pearsonr
from pycaret.time_series import *

# Start of execution time calculation
start = time.time()

# Setting the app layout
st.set_page_config(layout="wide")

# Display FLNT logo
image = Image.open(r"flnt logo.png")
st.sidebar.image(image,
                 width=160)

# Display file uploader (adding space beneath the FLNT logo)
st.sidebar.write("")
st.sidebar.write("")
data1 = st.sidebar.file_uploader("",type=["csv", "xls", "xlsx"])

st.sidebar.write("---")

# Check for errors during upload
try:
    # Read dataset file uploader
    if data1 is not None:
        if data1.name.endswith(".csv"):
            data_df1 = pd.read_csv(data1)
        else:
            data_df1 = pd.read_excel(data1)

    # Impute missing values with mean
    data_df1 = data_df1.fillna(data_df1.mean())

    # For choosing features and targets
    data_df1_types = data_df1.dtypes.to_dict()

    # Choosing features and target for file 1
    targets1 = []
    for key, val in data_df1_types.items():
        if val != object:
            targets1.append(key)

    help_dependent = "Dependent variable is the effect. It is the value that you are trying to forecast"
    help_independent = "Independent variable is the cause. It is the value which may contribute to the forecast"
    chosen_target1 = st.sidebar.selectbox("Choose dependent variable",
                                          targets1,
                                          help=help_dependent)
    features1 = list(data_df1_types.keys())
    features1.remove(chosen_target1)
    chosen_date1 = st.sidebar.selectbox("Choose date column to use",
                                        features1)
    chosen_features1 = st.sidebar.multiselect("Choose independent variable(s) to use",
                                              features1,
                                              help=help_independent)

    st.sidebar.write("---")

    # Create a dataframe based on chosen variables
    new_cols1 = chosen_features1.copy()
    new_cols1.append(chosen_target1)

    data_df1 = data_df1[new_cols1]

    # Preprocess data for experiment setup
    data_df1_series = data_df1.copy()

    # For descriptive stats
    data_df1_cols = data_df1.columns
    data_df1_shape = data_df1.shape

    data_df1_series[chosen_date1] = pd.to_datetime(data_df1_series[chosen_date1],
                                                   dayfirst=True)

    data_df1_series.set_index(data_df1_series[chosen_date1],
                              inplace=True)
    data_df1_series.drop(chosen_date1,
                         axis=1,
                         inplace=True)

    # Get the inferred frequency of the dataset uploaded
    inferred_frequency = infer_frequency(data_df1_series)
    st.sidebar.write(f"Inferred Frequency of Dataset Uploaded: {inferred_frequency}")

    st.sidebar.write("---")

    # Maximum number of lags for Granger-causality Test
    granger_lag = st.sidebar.number_input("Max lag for Granger-causality test",
                                          step=1,
                                          value=5,
                                          help="Maximum number of lag to use for checking causality of two time series")

    # Create tabs for plots and statistics
    plot_tab, stat_tab, forecast_tab, prescriptive_tab = st.tabs(["Plots",
                                                                  "Statistics",
                                                                  "Forecast",
                                                                  "Prescriptive"])

    # Test for stationarity of time series data
    def test_stationarity(timeseries):
        """
        Performs Augmented Dickey-Fuller test to check stationarity of input time series data

        :param timeseries: time series data to test for stationarity
        :return: boolean True(stationary) or False(non-stationary)
        """
        # perform Dickey-Fuller test
        dftest = adfuller(timeseries,
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        if dfoutput['p-value'] > 0.05:
            return False
        else:
            return True

    # Decompose time series data to get components (trend, seasonal, residual)
    def decompose(df, target_col):
        """
        Performs seasonal decomposition to extract trend, seasonal component, and residual from time series data

        :param df: input time series dataframe
        :param target_col: target column of dataframe
        :return: seasonal component, trend component, residual component (in this order)
        """
        dec = seasonal_decompose(df[target_col])
        return dec.seasonal, dec.trend, dec.resid

    # Test for stationarity, seasonalities present, and white noise
    max_seasonality = st.sidebar.number_input("Max seasonality to test",
                                              step=1,
                                              value=60)

    s = setup(data_df1_series,
              target=chosen_target1,
              seasonal_period=[i for i in range(1, max_seasonality + 1)],
              verbose=False)

    seasonality = s.seasonality_present
    all_seasonality = s.all_sp_values
    white_noise = s.white_noise

    seasonal, trend, residual = decompose(data_df1_series, chosen_target1)

    stationarity_data1 = test_stationarity(data_df1_series[chosen_target1])

    # Measures of central tendency
    data_stats = data_df1_series[chosen_target1].describe()
    mean_data1 = round(data_stats['mean'], 2)
    median_data1 = round(data_stats['50%'], 2)
    std_data1 = round(data_stats['std'], 2)

    # For Granger causality test
    residual_filled = residual.fillna(residual.mean())

    with stat_tab:
        st.header("Descriptive Statistics")
        st.write("---")

        # Show descriptive statistics for file 1
        st.metric("No. of Variables",
                  data_df1_shape[1])
        st.metric("No. of Observations",
                  data_df1_shape[0])
        st.metric("Mean",
                  mean_data1)
        st.metric("Median",
                  median_data1)
        st.metric("Standard Deviation",
                  std_data1)
        st.metric("Seasonality",
                  seasonality)
        help_stationary = "This tells whether the dataset has seasonality or trend. " \
                          "A dataset with trend or seasonality is not stationary"
        st.metric("Stationarity",
                  stationarity_data1,
                  help=help_stationary)
        help_white_noise = "The past values of the predictors cannot be used " \
                           "to predict the future values if white noise is present." \
                           " In other words, the time series uploaded is a random walk."
        st.metric("White Noise",
                  white_noise,
                  help=help_white_noise)
        st.write("Seasonalities Present: ",
                  str(all_seasonality))

        st.write("---")

        st.subheader("Granger Causality Test Results")

        # Granger-causality test
        granger_features = list()
        for feature in data_df1_series.columns:
            for i in range(1, granger_lag+1):
                p_val = grangercausalitytests(pd.DataFrame(zip(data_df1_series[feature], residual_filled)),
                                              maxlag=granger_lag,
                                              verbose=False)[i][0]['ssr_ftest'][1]
                if p_val < 0.05:
                    if feature != chosen_target1:
                        granger_features.append(feature)
                        st.write(
                            f"Knowing the values of {feature} is useful in predicting {chosen_target1} at lag {i}: {True}")

    with plot_tab:
        st.subheader(f"Plots for {data1.name}")

        def make_plot(
                name: str,
                x_data: pd.Series or np.array,
                y_data: pd.Series or np.array,
                x_title: str,
                y_title: str):
            """
            Create plotly graph object plots from given parameters

            :param name: name data to be shown
            :param x_data: pandas series or numpy array for horizontal (x-axis)
            :param y_data: pandas series or numpy array for vertical (y-axis)
            :param x_title: x-axis title
            :param y_title: y-axis title
            :return: plotly graph object plot
            """
            fig = go.Figure()
            fig.add_trace(go.Line(name=name,
                                  x=x_data,
                                  y=y_data))
            fig.update_xaxes(gridcolor='grey')
            fig.update_yaxes(gridcolor='grey')
            fig.update_layout(colorway=["#7EE3C9"],
                              xaxis_title=x_title,
                              yaxis_title=y_title,
                              title=f"{name} Plot")

            st.plotly_chart(fig,
                            use_container_width=True)

        # Data 1 plot
        make_plot(data1.name,
                  data_df1_series.index,
                  data_df1_series[chosen_target1],
                  chosen_date1,
                  chosen_target1)

        # Seasonal plot
        make_plot("Seasonal",
                  seasonal.index,
                  seasonal,
                  seasonal.index.name,
                  seasonal.name)

        # Trend plot
        make_plot("Trend",
                  trend.index,
                  trend,
                  trend.index.name,
                  trend.name)

        # Residual plot
        make_plot("Residual",
                  residual.index,
                  residual,
                  residual.index.name,
                  residual.name)

        st.subheader("Cross Correlation Plots")
        st.write("NOTE: The correlation values shown are based on the residuals (stationary version "
                 "of the time series data)."
                 " If the p-value is greater than 0.05, then the correlation value is purely by chance and"
                 " statistically insignificant.")
        st.write("")

        # Cross correlation plots
        for feat in data_df1_series.columns:
            lag_user = st.number_input(f"Cross correlation lag/shift for {feat}",
                                       step=1,
                                       key=feat)

            def make_correlation_plot(
                    df: pd.DataFrame,
                    target: str,
                    feature: str,
                    period: int,
                    date: str,
                    name: str):
                """
                Creates cross correlation and autocorrelation plots for time series data with corresponding lags

                :param df: input data in dataframe
                :param target: target name
                :param feature: feature name
                :param period: lag/shift to use
                :param date: chosen date column
                :param name: for title of plot
                :return: cross correlation and autocorrelation plot depending on the feature name
                """
                fig = go.Figure()
                fig.add_trace(go.Line(name=target,
                                      x=df.index,
                                      y=df[target]))
                fig.add_trace(go.Line(name=f"Shifted {feature}",
                                      x=df.index,
                                      y=df[feature].shift(periods=-1*period)))

                if stationarity_data1:
                    # If data is stationary, compute the correlation coefficient directly
                    corr_user = pearsonr(df[target].fillna(0),
                                         df[feature].shift(periods=-1*period).fillna(0))
                else:
                    # Stationarize time series then calculate correlation
                    residual_target = seasonal_decompose(df[target]).resid
                    residual_feature = seasonal_decompose(df[feature]).resid
                    corr_user = pearsonr(residual_target.fillna(0),
                                         residual_feature.shift(periods=-1*period).fillna(0))
                    
                fig.update_xaxes(gridcolor='grey')
                fig.update_yaxes(gridcolor='grey')
                fig.update_layout(xaxis_title=date,
                                  yaxis_title="Data",
                                  colorway=["#7EE3C9", "#70B0E0"],
                                  title=f"{name}: {round(corr_user[0], 2)} | p-value: {round(corr_user[1], 3)}")

                st.plotly_chart(fig,
                                use_container_width=True)

            if feat == chosen_target1:
                name = "Autocorrelation"
                make_correlation_plot(data_df1_series,
                                      chosen_target1,
                                      feat,
                                      lag_user,
                                      chosen_date1,
                                      name)

            else:
                name = "Data Correlation"
                make_correlation_plot(data_df1_series,
                                      chosen_target1,
                                      feat,
                                      lag_user,
                                      chosen_date1,
                                      name)

        st.subheader("Change Point Plot")

        # Change point plot
        @st.cache(allow_output_mutation=True)
        def change_point_plot(
                data: pd.Series or np.array,
                target: str,
                algorithm: str = "Pelt"):
            """
            Creates a plot of the input data with predicted change points based on the chosen algorithm
            :param data: a pandas series or numpy array
            :param target: target column
            :param algorithm: algorithm to detect change points,
            default="Pelt", options: "Pelt, "Binseg", "Window", "Dynp"
            :return: change point plot with break points
            """
            models_dict = {"Pelt": [rpt.Pelt, "rbf"],
                           "Binseg": [rpt.Binseg, "l2"],
                           "Window": [rpt.Window, "l2"]}

            model = models_dict[algorithm][1]

            if algorithm == "Pelt":
                algo = models_dict[algorithm][0](model=model).fit(data)
                my_bkps = algo.predict(pen=10)
            elif algorithm == "Window":
                algo = models_dict[algorithm][0](width=40, model=model).fit(data)
                my_bkps = algo.predict(n_bkps=10)
            else:
                algo = models_dict[algorithm][0](model=model).fit(data)
                my_bkps = algo.predict(n_bkps=10)

            fig = go.Figure()
            fig.add_trace(go.Line(name="Data",
                                  x=data.index,
                                  y=data[target]))
            for i in my_bkps[0:-1]:
                fig.add_vline(x=data.iloc[i].name, line_width=1, line_dash="dash", line_color="grey")
                
            fig.update_xaxes(gridcolor="#182534")
            fig.update_yaxes(gridcolor="#182534")
            fig.update_layout(colorway=["#7EE3C9"],
                              xaxis_title=data.index.name,
                              yaxis_title=chosen_target1,
                              title=f"Change Point Plot for {data1.name}")

            st.plotly_chart(fig,
                            use_container_width=True)

        algorithm_option = st.sidebar.selectbox("Choose algorithm to use for change point detection",
                                                ("Pelt", "Binseg", "Window"))

        change_point_plot(data_df1_series,
                          chosen_target1,
                          algorithm_option)

    with forecast_tab:
        # Create autoML model for forecasting
        data1_slider = st.sidebar.number_input("Forecast Horizon",
                                               min_value=1,
                                               value=5,
                                               step=1)
        if st.button("Forecast"):
            @st.cache(allow_output_mutation=True)
            def modeling(slider):
                model = AutoTS(
                    forecast_length=slider,
                    frequency='infer',
                    prediction_interval=0.95,
                    ensemble=None,
                    model_list='fast',
                    max_generations=5,
                    num_validations=1,
                    no_negatives=True
                )
                model = model.fit(data_df1_series)
                return model

            model = modeling(data1_slider)
            model_name1 = model.best_model_name
            prediction = model.predict()

            x_data1 = prediction.forecast.index
            y_data1 = prediction.forecast[chosen_target1].values
            y_upper1 = prediction.upper_forecast[chosen_target1].values
            y_lower1 = prediction.lower_forecast[chosen_target1].values

            # Forecast 1 plot
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                name="Data",
                x=data_df1_series.index,
                y=data_df1_series[chosen_target1]
            ))

            fig5.add_trace(go.Scatter(
                name='Prediction',
                x=x_data1,
                y=y_data1,
                # mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))

            fig5.add_trace(go.Scatter(
                name='Upper Bound',
                x=x_data1,
                y=y_upper1,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))

            fig5.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_data1,
                y=y_lower1,
                marker=dict(color="#70B0E0"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

            fig5.update_xaxes(gridcolor='grey')
            fig5.update_yaxes(gridcolor='grey')
            fig5.update_layout(xaxis_title=chosen_date1,
                               yaxis_title=chosen_target1,
                               title=f"{data1.name} Forecast using {model_name1}",
                               hovermode="x",
                               colorway=["#7EE3C9"])

            st.plotly_chart(fig5,
                            use_container_width=True)

    with prescriptive_tab:
        # Summarize results and give recommendations
        if len(granger_features) > 1:
            st.write(f"These variables {set(granger_features)} (at a certain lag) are useful"
                     f" for predicting the future values of {chosen_target1}."
                     f" Refer to the Granger-causality test results to identify the significant lags to use.")

        st.write("---")

        # Phillips-Ouliaris Test for cointegration
        if not stationarity_data1:
            po_test = phillips_ouliaris(data_df1_series[chosen_target1],
                                        data_df1_series[data_df1_series.columns])
        if po_test.pvalue < 0.05:
            st.write(f"These variables {data_df1_series.columns} have"
                     f" no cointegration with {chosen_target1}")
        else:
            st.write(f"The chosen independent variables have"
                     f" cointegration with {chosen_target1}."
                     f" Thus, they have a significant relationship or correlation"
                     f" which will be useful for forecasting.")

except (NameError, IndexError, KeyError) as e:
    pass

print("Done Rendering Application!")

st.write("---")
end = time.time()
execution_time = end - start
st.write(f"Execution Time: {round(execution_time, 2)} seconds")
