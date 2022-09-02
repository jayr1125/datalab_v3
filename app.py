import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from autots import AutoTS
from autots.tools.shaping import infer_frequency
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot.cointegration import phillips_ouliaris
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
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
    primary_seasonality = s.primary_sp_to_use
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
                           "In other words, the time series uploaded is a random walk."
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

        # Data 1 plot
        fig1 = go.Figure()
        fig1.add_trace(go.Line(name=data1.name,
                               x=data_df1_series.index,
                               y=data_df1_series[chosen_target1]))
        fig1.update_xaxes(gridcolor='grey')
        fig1.update_yaxes(gridcolor='grey')
        fig1.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=chosen_date1,
                           yaxis_title=chosen_target1,
                           title=f"{chosen_date1} vs. {chosen_target1}")

        st.plotly_chart(fig1,
                        use_container_width=True)

        # Seasonal plot
        fig2 = go.Figure()
        fig2.add_trace(go.Line(name="Seasonal",
                               x=seasonal.index,
                               y=seasonal))

        fig2.update_xaxes(gridcolor='grey')
        fig2.update_yaxes(gridcolor='grey')
        fig2.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=seasonal.index.name,
                           yaxis_title=seasonal.name,
                           title=f"Seasonal Component of {data1.name}")

        st.plotly_chart(fig2,
                        use_container_width=True)

        # Trend plot
        fig3 = go.Figure()
        fig3.add_trace(go.Line(name="Trend",
                               x=trend.index,
                               y=trend))

        fig3.update_xaxes(gridcolor='grey')
        fig3.update_yaxes(gridcolor='grey')
        fig3.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=trend.index.name,
                           yaxis_title=trend.name,
                           title=f"Trend Component of {data1.name}")

        st.plotly_chart(fig3,
                        use_container_width=True)

        # Residual plot
        fig4 = go.Figure()
        fig4.add_trace(go.Line(name="Residual",
                               x=residual.index,
                               y=residual))

        fig4.update_xaxes(gridcolor='grey')
        fig4.update_yaxes(gridcolor='grey')
        fig4.update_layout(colorway=["#7ee3c9"],
                           xaxis_title=residual.index.name,
                           yaxis_title=residual.name,
                           title=f"Residual Component of {data1.name}")

        st.plotly_chart(fig4,
                        use_container_width=True)

        st.subheader("Cross Correlation Plots")
        st.write("NOTE: The correlation values shown are based on the residuals (stationary version "
                 "of the time series data)."
                 " If the p-value is greater than 0.05, then the correlation value is purely by chance and"
                 " statistically insignificant.")
        st.write("")

        # Cross correlation plots
        if stationarity_data1:
            # If data is stationary, compute the correlation coefficient directly
            corr_user = pearsonr(data_df1_series[chosen_target1].fillna(0),
                                 data_df1_series[feat].shift(periods=-1*lag_user).fillna(0))
        else:
            # Stationarize time series then calculate correlation
            residual_target = seasonal_decompose(data_df1_series[chosen_target1]).resid
            residual_feature = seasonal_decompose(data_df1_series[feat]).resid
            corr_user = pearsonr(residual_target.fillna(0),
                                 residual_feature.shift(periods=-1*lag_user).fillna(0))
            
        # Cross correlation plots
        for feat in data_df1_series.columns:
            lag_user = st.number_input(f"Cross correlation lag/shift for {feat}",
                                       step=1,
                                       key=feat)
            
            if feat == chosen_target1:
                # Manual mode for cross correlation and choosing lag/shift
                fig5 = go.Figure()
                fig5.add_trace(go.Line(name=chosen_target1,
                                       x=data_df1_series.index,
                                       y=data_df1_series[chosen_target1]))
                fig5.add_trace(go.Line(name=f"Shifted {feat}",
                                       x=data_df1_series.index,
                                       y=data_df1_series[feat].shift(periods=-1*lag_user)))
                fig5.update_xaxes(gridcolor='grey')
                fig5.update_yaxes(gridcolor='grey')
                corr_user = data_df1_series[chosen_target1].corr(data_df1_series[feat].shift(periods=-1*lag_user))
                fig5.update_layout(xaxis_title=chosen_date1,
                                   yaxis_title="Data",
                                   colorway=["#7ee3c9", "#70B0E0"],
                                   title=f"Autocorrelation: {round(corr_user[0], 2)} | p-value: {round(corr_user[1], 3)}")

                st.plotly_chart(fig5,
                                use_container_width=True)
                
            else:
                fig5 = go.Figure()
                fig5.add_trace(go.Line(name=chosen_target1,
                                       x=data_df1_series.index,
                                       y=data_df1_series[chosen_target1]))
                fig5.add_trace(go.Line(name=f"Shifted {feat}",
                                       x=data_df1_series.index,
                                       y=data_df1_series[feat].shift(periods=-1*lag_user)))
                fig5.update_xaxes(gridcolor='grey')
                fig5.update_yaxes(gridcolor='grey')
                corr_user = data_df1_series[chosen_target1].corr(data_df1_series[feat].shift(periods=-1*lag_user))
                fig5.update_layout(xaxis_title=chosen_date1,
                                   yaxis_title="Data",
                                   colorway=["#7ee3c9", "#70B0E0"],
                                   title=f"Data Correlation: {round(corr_user[0], 2)} | p-value: {round(corr_user[1], 3)}")

                st.plotly_chart(fig5,
                                use_container_width=True)

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
                               colorway=["#7ee3c9"])

            st.plotly_chart(fig5,
                            use_container_width=True)

    with prescriptive_tab:
        # Summarize results and give recommendations
        st.write(f"The recommended seasonality to use is {primary_seasonality}")

        st.write("---")

        if len(granger_features) > 1:
            st.write(f"These variables {set(granger_features)} (at a certain lag) are useful"
                     f" for predicting the future values of {chosen_target1}."
                     f" Refer to the Granger-causality test results to identify the significant lags to use.")

        st.write("---")

        # Phillips-Ouliaris Test for cointegration
        if stationarity_data1 == False:
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

        #feature_lag = st.sidebar.number_input("Max lag for dependent variable to test for variable importance",
        #                                      step=1,
        #                                      value=5)

        #series = data_df1_series[chosen_target1]
        #differenced = series.diff(feature_lag)
        #differenced = differenced[feature_lag:]
        #series = differenced.copy()

        #dataframe_lag = pd.DataFrame()
        #for i in range(feature_lag, 0, -1):
        #    dataframe_lag['t' + str(i)] = series.shift(i).values[:, 0]
        #st.write(series.shift(5).values[:, 0])
        #dataframe_lag['t'] = series.values[:, 0]
        #dataframe_lag = dataframe_lag[feature_lag+1:]
        #st.dataframe(dataframe_lag)

except (NameError, IndexError, KeyError) as e:
    pass

print("Done Rendering Application!")

st.write("---")
end = time.time()
execution_time = end - start
st.write(f"Execution Time: {round(execution_time, 2)} seconds")
