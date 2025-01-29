# Note - do not run in RAPIDS VENV!
import dask.dataframe as dd # install dask-expr as well!
import pandas_ta as ta
import pandas as pd
import os
import pyarrow as pa
import gc
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
import shutil
import itertools
from sklearn.linear_model import LinearRegression

# Ensure necessary dependencies are installed
try:
    import pyarrow
except ImportError:
    raise ImportError("pyarrow is not installed. Please install it using 'pip install pyarrow'.")

class ForexETL:
    def __init__(self, base_dir, start_year=2019):
        self.base_dir = base_dir
        self.start_year = start_year

    def load_data(self, currency_pair, timeframe):
        file_pattern = f"{currency_pair}_{timeframe}_"
        file_path = self._find_file(file_pattern)
        if not file_path:
            raise FileNotFoundError(f"No file found for pattern {file_pattern}")

        df = pd.read_csv(file_path, sep='\t')
        df.columns = df.columns.str.replace('<', '').str.replace('>', '')

        if 'TIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
            df.drop(columns=['DATE', 'TIME'], inplace=True)
            df.rename(columns={'DATETIME': 'DATE'}, inplace=True)
        else:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%Y.%m.%d')

        df['currency_pair'] = currency_pair
        df['timeframe'] = timeframe

        # Filter out rows where the year is before start year
        df = df[df['DATE'].dt.year >= self.start_year]

        # Ensure DataFrame is sorted by date and set the index
        df = df.sort_values(by='DATE')
        df = df.set_index('DATE')

        return df

    def _find_file(self, pattern):
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.startswith(pattern) and file.endswith('.csv'):
                    return os.path.join(root, file)
        return None

    def add_technical_indicators(self, df):
        ######################################## Candlestick Patterns ########################
        # Outputs 1 if a Doji candlestick pattern is detected, otherwise 0.
        df['cdl_doji'] = ta.cdl_doji(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        # Outputs 1 if an Inside Bar candlestick pattern is detected, otherwise 0.
        df['cdl_inside'] = ta.cdl_inside(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])

        ######################################## Momentum Indicators ########################
        # Relative Strength Index (RSI)
        # Outputs the RSI values for the specified period (14).
        df['RSI_14'] = ta.rsi(df['CLOSE'], length=14)
        # Shorter
        df['RSI_7'] = ta.rsi(df['CLOSE'], length=7)
        df['RSI_10'] = ta.rsi(df['CLOSE'], length=10)
        # Longer
        df['RSI_20'] = ta.rsi(df['CLOSE'], length=20)

        # Moving Average Convergence Divergence (MACD)
        # Outputs the MACD line, signal line, and MACD histogram for the specified parameters (fast=12, slow=26, signal=9).
        # Transform MACD and Signal line to percentages of the closing price
        macd_result = ta.macd(df['CLOSE'], fast=12, slow=26, signal=9)
        df['MACD_12_26_9'] = macd_result['MACD_12_26_9'] / df['CLOSE']
        df['MACD_SIGNAL_12_26_9'] = macd_result['MACDs_12_26_9'] / df['CLOSE']
        # Keep the MACD Histogram as is
        df['MACD_HIST_12_26_9'] = macd_result['MACDh_12_26_9']

        # Shorter period MACD 1
        macd_short_result_1 = ta.macd(df['CLOSE'], fast=6, slow=13, signal=4)
        df['MACD_6_13_4'] = macd_short_result_1['MACD_6_13_4'] / df['CLOSE']
        df['MACD_SIGNAL_6_13_4'] = macd_short_result_1['MACDs_6_13_4'] / df['CLOSE']
        df['MACD_HIST_6_13_4'] = macd_short_result_1['MACDh_6_13_4']

        # Shorter period MACD 2
        macd_short_result_2 = ta.macd(df['CLOSE'], fast=8, slow=18, signal=6)
        df['MACD_8_18_6'] = macd_short_result_2['MACD_8_18_6'] / df['CLOSE']
        df['MACD_SIGNAL_8_18_6'] = macd_short_result_2['MACDs_8_18_6'] / df['CLOSE']
        df['MACD_HIST_8_18_6'] = macd_short_result_2['MACDh_8_18_6']

        # Longer period MACD 1
        macd_long_result_1 = ta.macd(df['CLOSE'], fast=16, slow=34, signal=12)
        df['MACD_16_34_12'] = macd_long_result_1['MACD_16_34_12'] / df['CLOSE']
        df['MACD_SIGNAL_16_34_12'] = macd_long_result_1['MACDs_16_34_12'] / df['CLOSE']
        df['MACD_HIST_16_34_12'] = macd_long_result_1['MACDh_16_34_12']


        # Stochastic Oscillator
        # Outputs the Stochastic K and D values for the specified parameters (fastk=14, slowk=3, slowd=3).
        stoch_result = ta.stoch(df['HIGH'], df['LOW'], df['CLOSE'], k=14, d=3, smooth_k=3)
        df['STOCHk_14_3_3'] = stoch_result['STOCHk_14_3_3']
        df['STOCHd_14_3_3'] = stoch_result['STOCHd_14_3_3']

        # Shorter period Stochastic Oscillator 1
        stoch_short_result_1 = ta.stoch(df['HIGH'], df['LOW'], df['CLOSE'], k=7, d=2, smooth_k=2)
        df['STOCHk_7_2_2'] = stoch_short_result_1['STOCHk_7_2_2']
        df['STOCHd_7_2_2'] = stoch_short_result_1['STOCHd_7_2_2']

        # Shorter period Stochastic Oscillator 2
        stoch_short_result_2 = ta.stoch(df['HIGH'], df['LOW'], df['CLOSE'], k=10, d=3, smooth_k=3)
        df['STOCHk_10_3_3'] = stoch_short_result_2['STOCHk_10_3_3']
        df['STOCHd_10_3_3'] = stoch_short_result_2['STOCHd_10_3_3']

        # Longer period Stochastic Oscillator 1
        stoch_long_result_1 = ta.stoch(df['HIGH'], df['LOW'], df['CLOSE'], k=21, d=4, smooth_k=4)
        df['STOCHk_21_4_4'] = stoch_long_result_1['STOCHk_21_4_4']
        df['STOCHd_21_4_4'] = stoch_long_result_1['STOCHd_21_4_4']

        # Chande Momentum Oscillator (CMO)
        # Outputs the CMO values for the specified period (14).
        df['CMO_14'] = ta.cmo(df['CLOSE'], length=14)

        # Shorter period CMO 1
        df['CMO_7'] = ta.cmo(df['CLOSE'], length=7)

        # Shorter period CMO 2
        df['CMO_10'] = ta.cmo(df['CLOSE'], length=10)

        # Longer period CMO 1
        df['CMO_21'] = ta.cmo(df['CLOSE'], length=21)

        # Commodity Channel Index (CCI)
        # Outputs the CCI values for the specified period (14).
        df['CCI_14'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=14)

        # Shorter period CCI 1
        df['CCI_7'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=7)

        # Shorter period CCI 2
        df['CCI_10'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=10)

        # Longer period CCI 1
        df['CCI_21'] = ta.cci(df['HIGH'], df['LOW'], df['CLOSE'], length=21)

        # Rate of Change (ROC)
        # Outputs the ROC values for the specified period (12).
        df['ROC_12'] = ta.roc(df['CLOSE'], length=12)

        # Shorter period ROC 1
        df['ROC_6'] = ta.roc(df['CLOSE'], length=6)

        # Shorter period ROC 2
        df['ROC_9'] = ta.roc(df['CLOSE'], length=9)

        # Longer period ROC 1
        df['ROC_18'] = ta.roc(df['CLOSE'], length=18)

        # Awesome Oscillator (AO)
        # Outputs the AO values.
        df['AO'] = ta.ao(df['HIGH'], df['LOW'])

        # Percentage Price Oscillator (PPO)
        # Outputs the PPO line, signal line, and PPO histogram for the specified parameters (fast=12, slow=26, signal=9).
        ppo_result = ta.ppo(df['CLOSE'], fast=12, slow=26, signal=9)
        df['PPO_12_26_9'] = ppo_result['PPO_12_26_9']
        df['PPO_SIGNAL_12_26_9'] = ppo_result['PPOs_12_26_9']
        df['PPO_HIST_12_26_9'] = ppo_result['PPOh_12_26_9']

        # Shorter period PPO 1
        ppo_short_result_1 = ta.ppo(df['CLOSE'], fast=6, slow=13, signal=6)
        df['PPO_6_13_6'] = ppo_short_result_1['PPO_6_13_6']
        df['PPO_SIGNAL_6_13_6'] = ppo_short_result_1['PPOs_6_13_6']
        df['PPO_HIST_6_13_6'] = ppo_short_result_1['PPOh_6_13_6']

        # Shorter period PPO 2
        ppo_short_result_2 = ta.ppo(df['CLOSE'], fast=8, slow=17, signal=8)
        df['PPO_8_17_8'] = ppo_short_result_2['PPO_8_17_8']
        df['PPO_SIGNAL_8_17_8'] = ppo_short_result_2['PPOs_8_17_8']
        df['PPO_HIST_8_17_8'] = ppo_short_result_2['PPOh_8_17_8']

        # Longer period PPO 1
        ppo_long_result_1 = ta.ppo(df['CLOSE'], fast=18, slow=39, signal=9)
        df['PPO_18_39_9'] = ppo_long_result_1['PPO_18_39_9']
        df['PPO_SIGNAL_18_39_9'] = ppo_long_result_1['PPOs_18_39_9']
        df['PPO_HIST_18_39_9'] = ppo_long_result_1['PPOh_18_39_9']

        # True Strength Index (TSI)
        # Outputs the TSI line and signal line for the specified parameters (long=25, short=13).
        tsi_result = ta.tsi(df['CLOSE'], fast=25, slow=13, signal=13)
        df['TSI_25_13'] = tsi_result['TSI_25_13_13']
        df['TSI_SIGNAL_25_13'] = tsi_result['TSIs_25_13_13']

        # Shorter period TSI 1
        tsi_short_result_1 = ta.tsi(df['CLOSE'], fast=15, slow=7, signal=7)
        df['TSI_15_7'] = tsi_short_result_1['TSI_15_7_7']
        df['TSI_SIGNAL_15_7'] = tsi_short_result_1['TSIs_15_7_7']

        # Shorter period TSI 2
        tsi_short_result_2 = ta.tsi(df['CLOSE'], fast=20, slow=10, signal=10)
        df['TSI_20_10'] = tsi_short_result_2['TSI_20_10_10']
        df['TSI_SIGNAL_20_10'] = tsi_short_result_2['TSIs_20_10_10']

        # Longer period TSI 1
        tsi_long_result_1 = ta.tsi(df['CLOSE'], fast=30, slow=15, signal=15)
        df['TSI_30_15'] = tsi_long_result_1['TSI_30_15_15']
        df['TSI_SIGNAL_30_15'] = tsi_long_result_1['TSIs_30_15_15']

        # Williams %R (WILLR)
        # Outputs the Williams %R values for the specified period (14).
        df['WILLR_14'] = ta.willr(df['HIGH'], df['LOW'], df['CLOSE'], length=14)

        # Shorter period WILLR 1
        df['WILLR_10'] = ta.willr(df['HIGH'], df['LOW'], df['CLOSE'], length=10)

        # Shorter period WILLR 2
        df['WILLR_12'] = ta.willr(df['HIGH'], df['LOW'], df['CLOSE'], length=12)

        # Longer period WILLR 1
        df['WILLR_20'] = ta.willr(df['HIGH'], df['LOW'], df['CLOSE'], length=20)

        # Ultimate Oscillator (UO)
        # Outputs the UO values for the specified parameters (short=7, medium=14, long=28).
        df['UO_7_14_28'] = ta.uo(df['HIGH'], df['LOW'], df['CLOSE'], fast=7, slow=14, length=28)

        # Shorter period UO 1
        df['UO_5_10_20'] = ta.uo(df['HIGH'], df['LOW'], df['CLOSE'], fast=5, slow=10, length=20)

        # Shorter period UO 2
        df['UO_6_12_24'] = ta.uo(df['HIGH'], df['LOW'], df['CLOSE'], fast=6, slow=12, length=24)

        # Longer period UO 1
        df['UO_10_20_40'] = ta.uo(df['HIGH'], df['LOW'], df['CLOSE'], fast=10, slow=20, length=40)

        ######################################## Overlap Studies ########################

        # Arnaud Legoux Moving Average (ALMA)
        # Outputs a smoothed moving average of the closing prices
        df['ALMA_9_0.85_6'] = ta.alma(df['CLOSE'], length=9, sigma=6, offset=0.85)/ df['CLOSE']

        # Shorter period ALMA 1
        df['ALMA_5_0.85_6'] = ta.alma(df['CLOSE'], length=5, sigma=6, offset=0.85)/ df['CLOSE']

        # Shorter period ALMA 2
        df['ALMA_7_0.85_6'] = ta.alma(df['CLOSE'], length=7, sigma=6, offset=0.85)/ df['CLOSE']

        # Longer period ALMA 1
        df['ALMA_15_0.85_6'] = ta.alma(df['CLOSE'], length=15, sigma=6, offset=0.85)/ df['CLOSE']

        # Double Exponential Moving Average (DEMA)
        # Outputs a smoothed moving average with less lag than the standard EMA
        df['DEMA_10'] = ta.dema(df['CLOSE'], length=10)/ df['CLOSE']

        # Shorter period DEMA 1
        df['DEMA_5'] = ta.dema(df['CLOSE'], length=5)/ df['CLOSE']

        # Shorter period DEMA 2
        df['DEMA_7'] = ta.dema(df['CLOSE'], length=7)/ df['CLOSE']

        # Longer period DEMA 1
        df['DEMA_15'] = ta.dema(df['CLOSE'], length=15)/ df['CLOSE']

        # Exponential Moving Average (EMA)
        # Outputs a smoothed moving average that places a greater weight on recent prices
        df['EMA_12'] = ta.ema(df['CLOSE'], length=12)/ df['CLOSE']

        # Shorter period EMA 1
        df['EMA_5'] = ta.ema(df['CLOSE'], length=5)/ df['CLOSE']

        # Shorter period EMA 2
        df['EMA_8'] = ta.ema(df['CLOSE'], length=8)/ df['CLOSE']

        # Longer period EMA 1
        df['EMA_20'] = ta.ema(df['CLOSE'], length=20)/ df['CLOSE']

        # Hull Moving Average (HMA)
        # Outputs a smoothed moving average that reduces lag while improving smoothness
        df['HMA_15'] = ta.hma(df['CLOSE'], length=15)/ df['CLOSE']

        # Shorter period HMA 1
        df['HMA_5'] = ta.hma(df['CLOSE'], length=5)/ df['CLOSE']

        # Shorter period HMA 2
        df['HMA_10'] = ta.hma(df['CLOSE'], length=10)/ df['CLOSE']

        # Longer period HMA 1
        df['HMA_20'] = ta.hma(df['CLOSE'], length=20)/ df['CLOSE']

        # Kaufman Adaptive Moving Average (KAMA)
        # Outputs a moving average that adjusts its sensitivity based on market volatility
        df['KAMA_10_2_30'] = ta.kama(df['CLOSE'], length=10, fast=2, slow=30)/ df['CLOSE']

        # Shorter period KAMA 1
        df['KAMA_5_2_30'] = ta.kama(df['CLOSE'], length=5, fast=2, slow=30)/ df['CLOSE']

        # Shorter period KAMA 2
        df['KAMA_7_2_30'] = ta.kama(df['CLOSE'], length=7, fast=2, slow=30)/ df['CLOSE']

        # Longer period KAMA 1
        df['KAMA_15_2_30'] = ta.kama(df['CLOSE'], length=15, fast=2, slow=30)/ df['CLOSE']

        # Simple Moving Average (SMA)
        # Outputs the unweighted mean of the previous n closing prices
        df['SMA_10'] = ta.sma(df['CLOSE'], length=10)/ df['CLOSE']

        # Shorter period SMA 1
        df['SMA_5'] = ta.sma(df['CLOSE'], length=5)/ df['CLOSE']

        # Shorter period SMA 2
        df['SMA_7'] = ta.sma(df['CLOSE'], length=7)/ df['CLOSE']

        # Longer period SMA 1
        df['SMA_15'] = ta.sma(df['CLOSE'], length=15)/ df['CLOSE']

        # Triple Exponential Moving Average (TEMA)
        # Outputs a smoothed moving average that aims to reduce lag more than the standard EMA
        df['TEMA_10'] = ta.tema(df['CLOSE'], length=10)/ df['CLOSE']

        # Shorter period TEMA 1
        df['TEMA_5'] = ta.tema(df['CLOSE'], length=5)/ df['CLOSE']

        # Shorter period TEMA 2
        df['TEMA_7'] = ta.tema(df['CLOSE'], length=7)/ df['CLOSE']

        # Longer period TEMA 1
        df['TEMA_15'] = ta.tema(df['CLOSE'], length=15)/ df['CLOSE']

        # Volume Weighted Average Price (VWAP)
        # Outputs the average price weighted by volume
        df['VWAP'] = ta.vwap(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL']) / df['CLOSE']

        # Weighted Moving Average (WMA)
        # Outputs a moving average where more recent prices are given greater weight
        df['WMA_10'] = ta.wma(df['CLOSE'], length=10)/ df['CLOSE']

        # Shorter
        df['WMA_5'] = ta.wma(df['CLOSE'], length=5)/ df['CLOSE']
        df['WMA_7'] = ta.wma(df['CLOSE'], length=7)/ df['CLOSE']
        # Longer
        df['WMA_15'] = ta.wma(df['CLOSE'], length=15)/ df['CLOSE']

        # Ichimoku Cloud Components
        # Outputs multiple components for trend identification:
        # - Conversion Line (Tenkan-sen)
        # - Base Line (Kijun-sen)
        # - Leading Span A (Senkou Span A)
        # - Leading Span B (Senkou Span B)
        # - Lagging Span (Chikou Span)
        ichimoku_result = ta.ichimoku(df['HIGH'], df['LOW'], df['CLOSE'])

        main_ichimoku = ichimoku_result[0]
        senkou_span = ichimoku_result[1]

        df['ICHIMOKU_CONVERSION_9'] = main_ichimoku['ITS_9']/ df['CLOSE']
        df['ICHIMOKU_BASE_26'] = main_ichimoku['IKS_26']/ df['CLOSE']
        df['ICHIMOKU_SPAN_A_9'] = main_ichimoku['ISA_9']/ df['CLOSE']
        df['ICHIMOKU_SPAN_B_26'] = main_ichimoku['ISB_26']/ df['CLOSE']
        df['ICHIMOKU_LAGGING_26'] = main_ichimoku['ICS_26']/ df['CLOSE']

        # Ichimoku Cloud Components - Shorter Period 1
        ichimoku_shorter1 = ta.ichimoku(df['HIGH'], df['LOW'], df['CLOSE'], tenkan=5, kijun=20, senkou=40)
        main_ichimoku_shorter1 = ichimoku_shorter1[0]

        df['ICHIMOKU_CONVERSION_5'] = main_ichimoku_shorter1['ITS_5']/ df['CLOSE']
        df['ICHIMOKU_BASE_20'] = main_ichimoku_shorter1['IKS_20']/ df['CLOSE']
        df['ICHIMOKU_SPAN_A_5'] = main_ichimoku_shorter1['ISA_5']/ df['CLOSE']
        df['ICHIMOKU_SPAN_B_20'] = main_ichimoku_shorter1['ISB_20']/ df['CLOSE']
        df['ICHIMOKU_LAGGING_20'] = main_ichimoku_shorter1['ICS_20']/ df['CLOSE']

        # Ichimoku Cloud Components - Shorter Period 2
        ichimoku_shorter2 = ta.ichimoku(df['HIGH'], df['LOW'], df['CLOSE'], tenkan=7, kijun=22, senkou=44)
        main_ichimoku_shorter2 = ichimoku_shorter2[0]

        df['ICHIMOKU_CONVERSION_7'] = main_ichimoku_shorter2['ITS_7']/ df['CLOSE']
        df['ICHIMOKU_BASE_22'] = main_ichimoku_shorter2['IKS_22']/ df['CLOSE']
        df['ICHIMOKU_SPAN_A_7'] = main_ichimoku_shorter2['ISA_7']/ df['CLOSE']
        df['ICHIMOKU_SPAN_B_22'] = main_ichimoku_shorter2['ISB_22']/ df['CLOSE']
        df['ICHIMOKU_LAGGING_22'] = main_ichimoku_shorter2['ICS_22']/ df['CLOSE']

        # Ichimoku Cloud Components - Longer Period 1
        ichimoku_longer1 = ta.ichimoku(df['HIGH'], df['LOW'], df['CLOSE'], tenkan=12, kijun=30, senkou=60)
        main_ichimoku_longer1 = ichimoku_longer1[0]

        df['ICHIMOKU_CONVERSION_12'] = main_ichimoku_longer1['ITS_12']/ df['CLOSE']
        df['ICHIMOKU_BASE_30'] = main_ichimoku_longer1['IKS_30']/ df['CLOSE']
        df['ICHIMOKU_SPAN_A_12'] = main_ichimoku_longer1['ISA_12']/ df['CLOSE']
        df['ICHIMOKU_SPAN_B_30'] = main_ichimoku_longer1['ISB_30']/ df['CLOSE']
        df['ICHIMOKU_LAGGING_30'] = main_ichimoku_longer1['ICS_30']/ df['CLOSE']

        ######################################## Performance Metrics ########################

        # Log Return
        df['LOGRET_1'] = ta.log_return(df['CLOSE'], length=1)
        df['LOGRET_3'] = ta.log_return(df['CLOSE'], length=3)
        df['LOGRET_5'] = ta.log_return(df['CLOSE'], length=5)
        df['LOGRET_7'] = ta.log_return(df['CLOSE'], length=7)

        # Percent Return
        df['PCTRET_1'] = ta.percent_return(df['CLOSE'], length=1)
        df['PCTRET_3'] = ta.percent_return(df['CLOSE'], length=3)
        df['PCTRET_5'] = ta.percent_return(df['CLOSE'], length=5)
        df['PCTRET_7'] = ta.percent_return(df['CLOSE'], length=7)

        ######################################## Trend Indicators ########################
        # Average Directional Index (ADX)
        # The ADX is used to quantify the strength of a trend.
        # It includes ADX, ADX_POS (positive directional indicator), and ADX_NEG (negative directional indicator) values.
        adx_result = ta.adx(df['HIGH'], df['LOW'], df['CLOSE'], length=14)
        df['ADX_14'] = adx_result['ADX_14']
        df['ADX_POS_14'] = adx_result['DMP_14']
        df['ADX_NEG_14'] = adx_result['DMN_14']

        # ADX - Shorter Period 1
        adx_result_shorter1 = ta.adx(df['HIGH'], df['LOW'], df['CLOSE'], length=10)
        df['ADX_10'] = adx_result_shorter1['ADX_10']
        df['ADX_POS_10'] = adx_result_shorter1['DMP_10']
        df['ADX_NEG_10'] = adx_result_shorter1['DMN_10']

        # ADX - Shorter Period 2
        adx_result_shorter2 = ta.adx(df['HIGH'], df['LOW'], df['CLOSE'], length=7)
        df['ADX_7'] = adx_result_shorter2['ADX_7']
        df['ADX_POS_7'] = adx_result_shorter2['DMP_7']
        df['ADX_NEG_7'] = adx_result_shorter2['DMN_7']

        # ADX - Longer Period 1
        adx_result_longer1 = ta.adx(df['HIGH'], df['LOW'], df['CLOSE'], length=20)
        df['ADX_20'] = adx_result_longer1['ADX_20']
        df['ADX_POS_20'] = adx_result_longer1['DMP_20']
        df['ADX_NEG_20'] = adx_result_longer1['DMN_20']

        # Aroon Indicator
        # The Aroon indicator is used to identify trends and potential reversals by measuring the time between highs and lows.
        # It includes AROON_UP and AROON_DOWN values.
        aroon_result = ta.aroon(df['HIGH'], df['LOW'], length=25)
        df['AROON_UP_25'] = aroon_result['AROONU_25']
        df['AROON_DOWN_25'] = aroon_result['AROOND_25']

        # Aroon Indicator - Shorter Period 1
        aroon_result_shorter1 = ta.aroon(df['HIGH'], df['LOW'], length=20)
        df['AROON_UP_20'] = aroon_result_shorter1['AROONU_20']
        df['AROON_DOWN_20'] = aroon_result_shorter1['AROOND_20']

        # Aroon Indicator - Shorter Period 2
        aroon_result_shorter2 = ta.aroon(df['HIGH'], df['LOW'], length=15)
        df['AROON_UP_15'] = aroon_result_shorter2['AROONU_15']
        df['AROON_DOWN_15'] = aroon_result_shorter2['AROOND_15']

        # Aroon Indicator - Longer Period 1
        aroon_result_longer1 = ta.aroon(df['HIGH'], df['LOW'], length=30)
        df['AROON_UP_30'] = aroon_result_longer1['AROONU_30']
        df['AROON_DOWN_30'] = aroon_result_longer1['AROOND_30']

        # Choppiness Index (CHOP)
        # The Choppiness Index is used to determine if the market is trending or ranging.
        # It includes CHOP value.
        df['CHOP_14'] = ta.chop(df['HIGH'], df['LOW'], df['CLOSE'], length=14)

        # CHOP - Shorter Period 1
        df['CHOP_10'] = ta.chop(df['HIGH'], df['LOW'], df['CLOSE'], length=10)

        # CHOP - Shorter Period 2
        df['CHOP_7'] = ta.chop(df['HIGH'], df['LOW'], df['CLOSE'], length=7)

        # CHOP - Longer Period 1
        df['CHOP_20'] = ta.chop(df['HIGH'], df['LOW'], df['CLOSE'], length=20)

        # Parabolic SAR (PSAR)
        # The PSAR is used to identify potential reversal points in the market.
        # It includes PSAR_LONG, PSAR_SHORT, PSAR_AF (acceleration factor), and PSAR_REVERSAL values.
        psar_result = ta.psar(df['HIGH'], df['LOW'], df['CLOSE'])
        df['PSAR_LONG_0.02_0.2'] = psar_result['PSARl_0.02_0.2']/ df['CLOSE']
        df['PSAR_SHORT_0.02_0.2'] = psar_result['PSARs_0.02_0.2']/ df['CLOSE']
        df['PSAR_AF_0.02_0.2'] = psar_result['PSARaf_0.02_0.2']
        df['PSAR_REVERSAL_0.02_0.2'] = psar_result['PSARr_0.02_0.2']

        # PSAR - Shorter Period 1
        psar_result_shorter1 = ta.psar(df['HIGH'], df['LOW'], df['CLOSE'], af=0.02, max_af=0.1)
        df['PSAR_LONG_0.02_0.1'] = psar_result_shorter1['PSARl_0.02_0.1']/ df['CLOSE']
        df['PSAR_SHORT_0.02_0.1'] = psar_result_shorter1['PSARs_0.02_0.1']/ df['CLOSE']
        df['PSAR_AF_0.02_0.1'] = psar_result_shorter1['PSARaf_0.02_0.1']
        df['PSAR_REVERSAL_0.02_0.1'] = psar_result_shorter1['PSARr_0.02_0.1']

        # PSAR - Shorter Period 2
        psar_result_shorter2 = ta.psar(df['HIGH'], df['LOW'], df['CLOSE'], af=0.01, max_af=0.1)
        df['PSAR_LONG_0.01_0.1'] = psar_result_shorter2['PSARl_0.01_0.1']/ df['CLOSE']
        df['PSAR_SHORT_0.01_0.1'] = psar_result_shorter2['PSARs_0.01_0.1']/ df['CLOSE']
        df['PSAR_AF_0.01_0.1'] = psar_result_shorter2['PSARaf_0.01_0.1']
        df['PSAR_REVERSAL_0.01_0.1'] = psar_result_shorter2['PSARr_0.01_0.1']

        # PSAR - Longer Period 1
        psar_result_longer1 = ta.psar(df['HIGH'], df['LOW'], df['CLOSE'], af=0.02, max_af=0.3)
        df['PSAR_LONG_0.02_0.3'] = psar_result_longer1['PSARl_0.02_0.3']/ df['CLOSE']
        df['PSAR_SHORT_0.02_0.3'] = psar_result_longer1['PSARs_0.02_0.3']/ df['CLOSE']
        df['PSAR_AF_0.02_0.3'] = psar_result_longer1['PSARaf_0.02_0.3']
        df['PSAR_REVERSAL_0.02_0.3'] = psar_result_longer1['PSARr_0.02_0.3']

        # QStick Indicator
        # The QStick indicator measures buying and selling pressure based on the difference between the open and close prices.
        # It includes QSTICK value.
        df['QSTICK_10'] = ta.qstick(df['OPEN'], df['CLOSE'], length=10)

        # QStick Indicator - Shorter Period 1
        df['QSTICK_7'] = ta.qstick(df['OPEN'], df['CLOSE'], length=7)

        # QStick Indicator - Shorter Period 2
        df['QSTICK_5'] = ta.qstick(df['OPEN'], df['CLOSE'], length=5)

        # QStick Indicator - Longer Period 1
        df['QSTICK_15'] = ta.qstick(df['OPEN'], df['CLOSE'], length=15)

        # TTM Trend (TTM_TRND)
        # The TTM Trend identifies the direction of the market by comparing the close price to an average of the highs and lows over a specified period.
        # It includes TTM_TRND value.
        df['TTM_TRND_6'] = ta.ttm_trend(df['HIGH'], df['LOW'], df['CLOSE'], length=6)

        # TTM Trend - Shorter Period 1
        df['TTM_TRND_4'] = ta.ttm_trend(df['HIGH'], df['LOW'], df['CLOSE'], length=4)

        # TTM Trend - Shorter Period 2
        df['TTM_TRND_3'] = ta.ttm_trend(df['HIGH'], df['LOW'], df['CLOSE'], length=3)

        # TTM Trend - Longer Period 1
        df['TTM_TRND_8'] = ta.ttm_trend(df['HIGH'], df['LOW'], df['CLOSE'], length=8)

        # Volatility Hopf (VHF)
        # The VHF measures the strength of a trend by comparing the range of price movements to the sum of individual price movements over a specified period.
        # It includes VHF value.
        df['VHF_28'] = ta.vhf(df['CLOSE'], length=28)

        # VHF - Shorter Period 1
        df['VHF_20'] = ta.vhf(df['CLOSE'], length=20)

        # VHF - Shorter Period 2
        df['VHF_14'] = ta.vhf(df['CLOSE'], length=14)

        # VHF - Longer Period 1
        df['VHF_35'] = ta.vhf(df['CLOSE'], length=35)

        # Vortex Indicator
        # The Vortex Indicator is used to identify the start of a new trend and includes VORTEX_POS (positive trend) and VORTEX_NEG (negative trend) values.
        vortex_result = ta.vortex(df['HIGH'], df['LOW'], df['CLOSE'], length=14)
        df['VORTEX_POS_14'] = vortex_result['VTXP_14']
        df['VORTEX_NEG_14'] = vortex_result['VTXM_14']

        # Vortex Indicator - Shorter Period 1
        vortex_result_shorter1 = ta.vortex(df['HIGH'], df['LOW'], df['CLOSE'], length=10)
        df['VORTEX_POS_10'] = vortex_result_shorter1['VTXP_10']
        df['VORTEX_NEG_10'] = vortex_result_shorter1['VTXM_10']

        # Vortex Indicator - Shorter Period 2
        vortex_result_shorter2 = ta.vortex(df['HIGH'], df['LOW'], df['CLOSE'], length=7)
        df['VORTEX_POS_7'] = vortex_result_shorter2['VTXP_7']
        df['VORTEX_NEG_7'] = vortex_result_shorter2['VTXM_7']

        # Vortex Indicator - Longer Period 1
        vortex_result_longer1 = ta.vortex(df['HIGH'], df['LOW'], df['CLOSE'], length=20)
        df['VORTEX_POS_20'] = vortex_result_longer1['VTXP_20']
        df['VORTEX_NEG_20'] = vortex_result_longer1['VTXM_20']

        # Trend Signals (TSignals)
        # The TSignals indicator provides trend direction signals, including trends, trades, entries, and exits.
        tsignals_result = ta.tsignals(df['CLOSE'])
        df['TS_Trends'] = tsignals_result['TS_Trends']
        df['TS_Trades'] = tsignals_result['TS_Trades']
        df['TS_Entries'] = tsignals_result['TS_Entries']
        df['TS_Exits'] = tsignals_result['TS_Exits']

        # TSignals - Shorter Period
        tsignals_result_shorter2 = ta.tsignals(df['CLOSE'], length=7)
        df['TS_Trends_7'] = tsignals_result_shorter2['TS_Trends']
        df['TS_Trades_7'] = tsignals_result_shorter2['TS_Trades']
        df['TS_Entries_7'] = tsignals_result_shorter2['TS_Entries']
        df['TS_Exits_7'] = tsignals_result_shorter2['TS_Exits']

        ######################################## Volatility Indicators ########################
        # Average True Range (ATR)
        # The ATR measures market volatility by calculating the average of true ranges over a specified period.
        # It includes ATR value.
        df['ATR_14'] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=14, percent=True)

        # ATR - Shorter Period 1
        df['ATR_10'] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=10, percent=True)

        # ATR - Shorter Period 2
        df['ATR_7'] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=7, percent=True)

        # ATR - Longer Period 1
        df['ATR_20'] = ta.atr(df['HIGH'], df['LOW'], df['CLOSE'], length=20, percent=True)

        # Bollinger Bands (BBANDS)
        # Bollinger Bands are volatility bands placed above and below a moving average.
        # %B for length=20, std=2
        bbands_result = ta.bbands(df['CLOSE'], length=20, std=2)
        df['PCT_BBANDS_20_2'] = (df['CLOSE'] - bbands_result['BBL_20_2.0']) / (
                    bbands_result['BBU_20_2.0'] - bbands_result['BBL_20_2.0'])

        # %B for length=15, std=2
        bbands_result_shorter1 = ta.bbands(df['CLOSE'], length=15, std=2)
        df['PCT_BBANDS_15_2'] = (df['CLOSE'] - bbands_result_shorter1['BBL_15_2.0']) / (
                    bbands_result_shorter1['BBU_15_2.0'] - bbands_result_shorter1['BBL_15_2.0'])

        # %B for length=10, std=2
        bbands_result_shorter2 = ta.bbands(df['CLOSE'], length=10, std=2)
        df['PCT_BBANDS_10_2'] = (df['CLOSE'] - bbands_result_shorter2['BBL_10_2.0']) / (
                    bbands_result_shorter2['BBU_10_2.0'] - bbands_result_shorter2['BBL_10_2.0'])

        # %B for length=25, std=2
        bbands_result_longer1 = ta.bbands(df['CLOSE'], length=25, std=2)
        df['PCT_BBANDS_25_2'] = (df['CLOSE'] - bbands_result_longer1['BBL_25_2.0']) / (
                    bbands_result_longer1['BBU_25_2.0'] - bbands_result_longer1['BBL_25_2.0'])

        # Keltner Channels (KC)
        # Keltner Channels are volatility-based envelopes set above and below an exponential moving average.
        # It includes KC_UPPER, KC_MIDDLE, and KC_LOWER values.
        kc_result = ta.kc(df['HIGH'], df['LOW'], df['CLOSE'], length=20)
        df['PCT_KC_20_2'] = (df['CLOSE'] - kc_result['KCLe_20_2']) / (kc_result['KCUe_20_2'] - kc_result['KCLe_20_2'])

        # If you want to include other periods, add them similarly
        # %K for length=15, scalar=2
        kc_result_shorter1 = ta.kc(df['HIGH'], df['LOW'], df['CLOSE'], length=15)
        df['PCT_KC_15_2'] = (df['CLOSE'] - kc_result_shorter1['KCLe_15_2']) / (
                    kc_result_shorter1['KCUe_15_2'] - kc_result_shorter1['KCLe_15_2'])

        # %K for length=10, scalar=2
        kc_result_shorter2 = ta.kc(df['HIGH'], df['LOW'], df['CLOSE'], length=10)
        df['PCT_KC_10_2'] = (df['CLOSE'] - kc_result_shorter2['KCLe_10_2']) / (
                    kc_result_shorter2['KCUe_10_2'] - kc_result_shorter2['KCLe_10_2'])

        # %K for length=25, scalar=2
        kc_result_longer1 = ta.kc(df['HIGH'], df['LOW'], df['CLOSE'], length=25)
        df['PCT_KC_25_2'] = (df['CLOSE'] - kc_result_longer1['KCLe_25_2']) / (
                    kc_result_longer1['KCUe_25_2'] - kc_result_longer1['KCLe_25_2'])

        # Donchian Channels (Donchian)
        # Donchian Channels plot the highest high and the lowest low over a specified period.
        # It includes DONCHIAN_UPPER, DONCHIAN_MIDDLE, and DONCHIAN_LOWER values.
        donchian_result = ta.donchian(df['HIGH'], df['LOW'], lower_length=20, upper_length=20)
        df['PCT_DONCHIAN_20_20'] = (df['CLOSE'] - donchian_result['DCL_20_20']) / (
                    donchian_result['DCU_20_20'] - donchian_result['DCL_20_20'])

        # Shorter Period 1
        donchian_result_shorter1 = ta.donchian(df['HIGH'], df['LOW'], lower_length=15, upper_length=15)
        df['PCT_DONCHIAN_15_15'] = (df['CLOSE'] - donchian_result_shorter1['DCL_15_15']) / (
                    donchian_result_shorter1['DCU_15_15'] - donchian_result_shorter1['DCL_15_15'])

        # Shorter Period 2
        donchian_result_shorter2 = ta.donchian(df['HIGH'], df['LOW'], lower_length=10, upper_length=10)
        df['PCT_DONCHIAN_10_10'] = (df['CLOSE'] - donchian_result_shorter2['DCL_10_10']) / (
                    donchian_result_shorter2['DCU_10_10'] - donchian_result_shorter2['DCL_10_10'])

        # Longer Period 1
        donchian_result_longer1 = ta.donchian(df['HIGH'], df['LOW'], lower_length=25, upper_length=25)
        df['PCT_DONCHIAN_25_25'] = (df['CLOSE'] - donchian_result_longer1['DCL_25_25']) / (
                    donchian_result_longer1['DCU_25_25'] - donchian_result_longer1['DCL_25_25'])

        # Mass Index (MASSI)
        # The Mass Index identifies trend reversals by examining the range between the high and low prices.
        # It includes MASSI value.
        df['MASSI_25'] = ta.massi(df['HIGH'], df['LOW'], length=25)

        # MASSI - Shorter Period 1
        df['MASSI_20'] = ta.massi(df['HIGH'], df['LOW'], length=20)

        # MASSI - Shorter Period 2
        df['MASSI_15'] = ta.massi(df['HIGH'], df['LOW'], length=15)

        # MASSI - Longer Period 1
        df['MASSI_30'] = ta.massi(df['HIGH'], df['LOW'], length=30)

        ######################################## Volume Indicators ########################
        # Accumulation Distribution (AD)
        # The Accumulation/Distribution Line (ADL) is a volume-based indicator designed to measure the cumulative flow of money into and out of a security.
        df['AD'] = ta.ad(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'])

        # AD - Shorter Period 1
        df['AD_10'] = ta.ad(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=10)

        # AD - Shorter Period 2
        df['AD_7'] = ta.ad(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=7)

        # AD - Longer Period 1
        df['AD_20'] = ta.ad(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=20)

        # Chaikin Money Flow (CMF)
        # Chaikin Money Flow (CMF) is a volume-weighted average of accumulation and distribution over a specified period.
        df['CMF_20'] = ta.cmf(df['HIGH'], df['LOW'], df['CLOSE'],
                                       df['TICKVOL'], length=20)

        # CMF - Shorter Period 1
        df['CMF_15'] = ta.cmf(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=15)

        # CMF - Shorter Period 2
        df['CMF_10'] = ta.cmf(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=10)

        # CMF - Longer Period 1
        df['CMF_25'] = ta.cmf(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], length=25)

        # Elder’s Force Index (EFI)
        # Elder’s Force Index (EFI) uses price and volume to assess the power behind a price move.
        df['EFI_13'] = ta.efi(df['CLOSE'], df['TICKVOL'], length=13)

        # EFI - Shorter Period 1
        df['EFI_10'] = ta.efi(df['CLOSE'], df['TICKVOL'], length=10)

        # EFI - Shorter Period 2
        df['EFI_7'] = ta.efi(df['CLOSE'], df['TICKVOL'], length=7)

        # EFI - Longer Period 1
        df['EFI_20'] = ta.efi(df['CLOSE'], df['TICKVOL'], length=20)

        # On-Balance Volume (OBV)
        # On-Balance Volume (OBV) measures buying and selling pressure as a cumulative indicator that adds volume on up days and subtracts volume on down days.
        df['OBV'] = ta.obv(df['CLOSE'], df['TICKVOL'])

        # Klinger Volume Oscillator (KVO)
        # The Klinger Volume Oscillator (KVO) is a volume-based indicator that combines short- and long-term volume moving averages to identify trends.
        kvo_result = ta.kvo(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], fast=34, slow=55, signal=13)
        df['KVO_34_55_13'] = kvo_result['KVO_34_55_13']
        df['KVO_SIGNAL_34_55_13'] = kvo_result['KVOs_34_55_13']

        # Klinger Volume Oscillator - Shorter Period 1
        kvo_result_shorter1 = ta.kvo(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], fast=20, slow=40, signal=13)
        df['KVO_20_40_13'] = kvo_result_shorter1['KVO_20_40_13']
        df['KVO_SIGNAL_20_40_13'] = kvo_result_shorter1['KVOs_20_40_13']

        # Klinger Volume Oscillator - Shorter Period 2
        kvo_result_shorter2 = ta.kvo(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], fast=15, slow=30, signal=13)
        df['KVO_15_30_13'] = kvo_result_shorter2['KVO_15_30_13']
        df['KVO_SIGNAL_15_30_13'] = kvo_result_shorter2['KVOs_15_30_13']

        # Klinger Volume Oscillator - Longer Period 1
        kvo_result_longer1 = ta.kvo(df['HIGH'], df['LOW'], df['CLOSE'], df['TICKVOL'], fast=45, slow=60, signal=13)
        df['KVO_45_60_13'] = kvo_result_longer1['KVO_45_60_13']
        df['KVO_SIGNAL_45_60_13'] = kvo_result_longer1['KVOs_45_60_13']

        # Ratio of Close over Open
        df['RATIO_CLOSE_OVER_OPEN'] = df['CLOSE'] / df['OPEN']
        # Ratio of Close over High
        df['RATIO_CLOSE_OVER_HIGH'] = df['CLOSE'] / df['HIGH']
        # Ratio of Close over Low
        df['RATIO_CLOSE_OVER_LOW'] = df['CLOSE'] / df['LOW']

        # Periods since lowest low in last 5, 10, 30, 60 timesteps
        df['PERIODS_SINCE_LOW_5'] = df['LOW'].rolling(window=5).apply(lambda x: (x.argmax() if len(x) == 5 else None),
                                                                      raw=True)
        df['PERIODS_SINCE_LOW_10'] = df['LOW'].rolling(window=10).apply(
            lambda x: (x.argmax() if len(x) == 10 else None), raw=True)
        df['PERIODS_SINCE_LOW_30'] = df['LOW'].rolling(window=30).apply(
            lambda x: (x.argmax() if len(x) == 30 else None), raw=True)
        df['PERIODS_SINCE_LOW_60'] = df['LOW'].rolling(window=60).apply(
            lambda x: (x.argmax() if len(x) == 60 else None), raw=True)

        # Periods since highest high in last 5, 10, 30, 60 timesteps
        df['PERIODS_SINCE_HIGH_5'] = df['HIGH'].rolling(window=5).apply(lambda x: (x.argmax() if len(x) == 5 else None),
                                                                        raw=True)
        df['PERIODS_SINCE_HIGH_10'] = df['HIGH'].rolling(window=10).apply(
            lambda x: (x.argmax() if len(x) == 10 else None), raw=True)
        df['PERIODS_SINCE_HIGH_30'] = df['HIGH'].rolling(window=30).apply(
            lambda x: (x.argmax() if len(x) == 30 else None), raw=True)
        df['PERIODS_SINCE_HIGH_60'] = df['HIGH'].rolling(window=60).apply(
            lambda x: (x.argmax() if len(x) == 60 else None), raw=True)

        # Ratio of Close to lowest low in last 5, 10, 30, 60 timesteps
        df['RATIO_CLOSE_TO_LOW_5'] = df['CLOSE'] / df['LOW'].rolling(window=5).min()
        df['RATIO_CLOSE_TO_LOW_10'] = df['CLOSE'] / df['LOW'].rolling(window=10).min()
        df['RATIO_CLOSE_TO_LOW_30'] = df['CLOSE'] / df['LOW'].rolling(window=30).min()
        df['RATIO_CLOSE_TO_LOW_60'] = df['CLOSE'] / df['LOW'].rolling(window=60).min()

        # Ratio of Close to second lowest low in last 5, 10, 30, 60 timesteps
        df['RATIO_CLOSE_TO_SECOND_LOW_5'] = df['CLOSE'] / df['LOW'].rolling(window=5).apply(
            lambda x: sorted(x)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_LOW_10'] = df['CLOSE'] / df['LOW'].rolling(window=10).apply(
            lambda x: sorted(x)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_LOW_30'] = df['CLOSE'] / df['LOW'].rolling(window=30).apply(
            lambda x: sorted(x)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_LOW_60'] = df['CLOSE'] / df['LOW'].rolling(window=60).apply(
            lambda x: sorted(x)[1] if len(x) > 1 else np.nan)

        # Ratio of Close to highest high in last 5, 10, 30, 60 timesteps
        df['RATIO_CLOSE_TO_HIGH_5'] = df['CLOSE'] / df['HIGH'].rolling(window=5).max()
        df['RATIO_CLOSE_TO_HIGH_10'] = df['CLOSE'] / df['HIGH'].rolling(window=10).max()
        df['RATIO_CLOSE_TO_HIGH_30'] = df['CLOSE'] / df['HIGH'].rolling(window=30).max()
        df['RATIO_CLOSE_TO_HIGH_60'] = df['CLOSE'] / df['HIGH'].rolling(window=60).max()

        # Ratio of Close to second highest high in last 5, 10, 30, 60 timesteps
        df['RATIO_CLOSE_TO_SECOND_HIGH_5'] = df['CLOSE'] / df['HIGH'].rolling(window=5).apply(
            lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_HIGH_10'] = df['CLOSE'] / df['HIGH'].rolling(window=10).apply(
            lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_HIGH_30'] = df['CLOSE'] / df['HIGH'].rolling(window=30).apply(
            lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else np.nan)
        df['RATIO_CLOSE_TO_SECOND_HIGH_60'] = df['CLOSE'] / df['HIGH'].rolling(window=60).apply(
            lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else np.nan)

        del (
            bbands_result, bbands_result_shorter1, bbands_result_shorter2, bbands_result_longer1,
            kc_result, kc_result_shorter1, kc_result_shorter2, kc_result_longer1,
            donchian_result, donchian_result_shorter1, donchian_result_shorter2, donchian_result_longer1,
            macd_result, macd_short_result_1, macd_short_result_2, macd_long_result_1,
            stoch_result, stoch_short_result_1, stoch_short_result_2, stoch_long_result_1,
            ppo_result, ppo_short_result_1, ppo_short_result_2, ppo_long_result_1,
            tsi_result, tsi_short_result_1, tsi_short_result_2, tsi_long_result_1,
            adx_result, adx_result_shorter1, adx_result_shorter2, adx_result_longer1,
            aroon_result, aroon_result_shorter1, aroon_result_shorter2, aroon_result_longer1,
            psar_result, psar_result_shorter1, psar_result_shorter2, psar_result_longer1,
            ichimoku_result, main_ichimoku, senkou_span, ichimoku_shorter1, main_ichimoku_shorter1,
            ichimoku_shorter2, main_ichimoku_shorter2, ichimoku_longer1, main_ichimoku_longer1,
            vortex_result, vortex_result_shorter1, vortex_result_shorter2, vortex_result_longer1,
            tsignals_result, tsignals_result_shorter2, kvo_result, kvo_result_shorter1, kvo_result_shorter2,
            kvo_result_longer1
        )

        return df

    def process_and_save(self, currency_pairs, timeframes):
        parquet_files = []

        for currency_pair in currency_pairs:
            combined_data = []
            print(f"............................ELT for {currency_pair} ............................")
            for timeframe in timeframes:
                try:
                    print(f"Started ELT for {currency_pair} {timeframe}")
                    df = self.load_data(currency_pair, timeframe)
                    df = self.add_technical_indicators(df)
                    combined_data.append(df)

                    # Free up memory after processing each pair and timeframe
                    del df
                    gc.collect()
                    print(f"ETL Done for {currency_pair} {timeframe}")
                    print('....................................')
                except FileNotFoundError:
                    print(f"File not found for {currency_pair} {timeframe}")

            # Combine all dataframes into a single Pandas DataFrame for the current currency pair
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=False)
                del combined_data

                # Convert to a Dask DataFrame for parallel processing
                ddf_currency_pair = dd.from_pandas(combined_df, npartitions=15)
                del combined_df
                ddf_currency_pair = ddf_currency_pair.set_index('DATE')

                # Save the Dask DataFrame as a Parquet file
                parquet_file_path = os.path.join(self.base_dir, f"{currency_pair}_forex_data.parquet")
                ddf_currency_pair.to_parquet(parquet_file_path, write_index=True, engine='pyarrow', compression='snappy')
                print(f"Data for {currency_pair} saved to {parquet_file_path}")

                # Collect the parquet file paths
                parquet_files.append(parquet_file_path)

                # Free up memory after saving
                del ddf_currency_pair
                gc.collect()

        # Combine all saved Parquet files into a single Dask DataFrame
        ddf_combined = dd.concat([dd.read_parquet(file, engine='pyarrow') for file in parquet_files], axis=0)

        # Save the combined DataFrame as a single Parquet file
        enhanced_parquet_file_path = os.path.join(self.base_dir, "enhanced_combined_forex_data.parquet")
        ddf_combined.to_parquet(enhanced_parquet_file_path, write_index=True, engine='pyarrow', compression='snappy', write_metadata_file=False)

        print(f"Enhanced data saved to {enhanced_parquet_file_path}")

        # Free up memory after saving
        del ddf_combined
        gc.collect()

def delete_parquet_files(base_dir, currency_pairs):
    for currency_pair in currency_pairs:
        parquet_file_path = os.path.join(base_dir, f"{currency_pair}_forex_data.parquet")
        if os.path.exists(parquet_file_path):
            try:
                shutil.rmtree(parquet_file_path)
                print(f"Deleted directory: {parquet_file_path}")
            except OSError as e:
                print(f"Error deleting directory {parquet_file_path}: {e}")
        else:
            print(f"Directory not found: {parquet_file_path}")

    enhanced_parquet_file_path = os.path.join(base_dir, "enhanced_combined_forex_data.parquet")
    if os.path.exists(enhanced_parquet_file_path):
        try:
            shutil.rmtree(enhanced_parquet_file_path)
            print(f"Deleted directory: {enhanced_parquet_file_path}")
        except OSError as e:
            print(f"Error deleting directory {enhanced_parquet_file_path}: {e}")
    else:
        print(f"Directory not found: {enhanced_parquet_file_path}")


# Do the ETL
if __name__ == "__main__":
    base_dir = "../../data/Forex"
    etl = ForexETL(base_dir, start_year=2020)

    currency_pairs = ["EURGBP", "EURUSD", "GBPUSD"]
    timeframes = ["Daily", "H1", "M5", "M10", "M30"]

    etl.process_and_save(currency_pairs, timeframes)
    print('ETL Part 1 Done..................')

    # Read the saved Parquet file directly into a Dask DataFrame
    enhanced_parquet_file_path = os.path.join(base_dir, "enhanced_combined_forex_data.parquet")
    ddf = dd.read_parquet(enhanced_parquet_file_path, engine='pyarrow')
    print('ddf read..................')

    # Convert to a single Pandas DataFrame
    combined_df = ddf.compute()
    print('df combined..................')

    # Save as a single Parquet file
    single_file_parquet_path = os.path.join(base_dir, "single_file_forex_data.parquet")
    combined_df.to_parquet(single_file_parquet_path, engine='pyarrow', compression='snappy')

    print(f"Data saved to {single_file_parquet_path}")

    delete_parquet_files(base_dir, currency_pairs)
    print('Parquet files and folders deletion done.')
