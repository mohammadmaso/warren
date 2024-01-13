import datetime
import pandas as pd
from prophet import Prophet
import pytse_client as tse

class Dataset:
    def __init__(self, ticker):
        self.ticker = ticker
        self.instrument = None

    def build_dataset(self):
        start_date = pd.Timestamp('2010-01-01')
        end_date = pd.Timestamp.now()

        self.instrument = tse.download(symbols=self.ticker, adjust=True)
        try:
            history = self.instrument[self.ticker]
            history = history.reset_index()
            # Ensure the 'date' column is of datetime type
        # Ensure 'date' is datetime and handle NaT/Nan values
            history['date'] = pd.to_datetime(history['date'], errors='coerce')  # Coerce errors to NaT
            history = history.dropna(subset=['date'])  # Drop rows where 'date' is NaT

            # Filter the dataframe for the desired date range
            mask = (history['date'] >= start_date) & (history['date'] <= end_date)
            self.dataset = history.loc[mask]
            self.add_forecast_date()
        except Exception as e:
            print("Exception raised at: `Dataset.build_dataset()`", e)
            return False
        else:
            return True

    def add_forecast_date(self):
        present_date = self.dataset['date'].max()
        day_number = pd.to_datetime(present_date).isoweekday()
        if day_number in [5, 6]:
            self.forecast_date = present_date + datetime.timedelta(days=(8 - day_number))
        else:
            self.forecast_date = present_date + datetime.timedelta(days=1)
        print("Present date:", present_date)
        print("Valid Forecast Date:", self.forecast_date)
        test_row = pd.DataFrame([[self.forecast_date, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0]], columns=self.dataset.columns)
        self.dataset = pd.concat([self.dataset, test_row], ignore_index=True)


class FeatureEngineering(Dataset):
    def create_features(self):
        status = self.build_dataset()
        if status:
            self.create_lag_features()
            self.impute_missing_values()
            self.dataset.drop(columns=["open", "high", "low","adjClose", "value", "volume", "count", "yesterday"], inplace=True)
            print(self.dataset.tail(3))
            return True
        else:
            raise Exception("Dataset creation failed!")

    def create_lag_features(self, periods=12):
        for i in range(1, periods + 1):
            self.dataset[f"Close_lag_{i}"] = self.dataset['close'].shift(i)
            self.dataset[f"Open_lag_{i}"] = self.dataset['open'].shift(i)
            self.dataset[f"High_lag_{i}"] = self.dataset['high'].shift(i)
            self.dataset[f"Low_lag_{i}"] = self.dataset['low'].shift(i)
        return True

    def impute_missing_values(self):
        self.dataset.fillna(0, inplace=True)

        return True


class MasterProphet(FeatureEngineering):
    def __init__(self, ticker):
        super().__init__(ticker)
        self.model = None

    def build_model(self):
        additional_features = [col for col in self.dataset.columns if "lag" in col]
        try:
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
            for name in additional_features:
                self.model.add_regressor(name)
        except Exception as e:
            print("Exception raised at: `MasterProphet.build_model()`", e)
            return False
        else:
            return True

    def train_and_forecast(self):
        self.model.fit(df=self.dataset.iloc[:-1, :].rename(columns={"date": "ds", "close":"y"}))
        return self.model.predict(self.dataset.iloc[-1:][[col for col in self.dataset if col != "close"]].rename(columns={"date": "ds"}))

    def forecast(self):
        self.create_features()
        self.build_model()
        return self.train_and_forecast()