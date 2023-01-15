import pandas as pd
import numpy as np
import keras as kr
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.metrics import recall_score, r2_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

DATA = "2017_2019.csv"
DAYS = 13


class SunlightModel():
    def __init__(self):
        self.df = pd.read_csv(DATA)

    # utils
    def get_day(self, df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:

        return df.loc[(df['Year'] == day.year) &
                      (df['Day'] == day.day) &
                      (df['Month'] == day.month)]

    def get_records_night(self, df: pd.DataFrame, day: pd.Timestamp):
        records_beafore_df = self.get_n_days_beafore(self.df, day, 1)
        records_beafore_df = records_beafore_df.loc[(
            records_beafore_df.Hour > 12) & (records_beafore_df["Clearsky DNI"] != 0)]

        records_df = self.get_day(self.df, day)
        records_df = records_df.loc[(records_df.Hour < 12) & (
            records_df.DNI == 0) & (records_df["Clearsky DNI"] != 0)]

        return (pd.concat([records_beafore_df, records_df]))

    def get_n_days_beafore(self, df: pd.DataFrame, day: pd.Timestamp, n: int):
        day_beafore = day - pd.offsets.Day(n)
        return self.get_day(self.df, day_beafore)

    def get_working_day(self, df: pd.DataFrame, day: pd.Timestamp):
        records = self.get_day(self.df, day)
        return records.loc[(records.DNI != 0) | (records["Clearsky DNI"] != 0)]

    def get_Data_n_records(self, df, num, flatten_x=False):
        X = []
        Y = []
        for i in range(len(df)):
            if i < num+1:
                pass
            else:
                X.append(df.iloc[i-num-1:i-1].values)
                Y.append(df.iloc[i].Y)

        if flatten_x:  # Option to return one long row containing the X data
            # long_x = []
            X = np.array(X)
            first, second, third = X.shape
            return X.reshape(first, second * third), np.array(Y)

        return np.array(X), np.array(Y)

    def prepare_data(self):
        self.df.drop(['Unnamed: 18'], inplace=True, axis=1)

        dates = set()
        for i, data in self.df.iterrows():
            dates.add(pd.to_datetime(
                str(data['Year'])[:-2]+'-'+str(data['Month']
                                               )[:-2] + '-' + str(data['Day'])[:-2],
                format="%Y-%m-%d"
            ))
        _ = pd.Series(list(dates)).sort_values()
        self.get_records_night(self.df, _.iloc[2])

        records_df = pd.DataFrame()
        for idx, date in enumerate(pd.Series(list(dates)).sort_values().values):

            date = pd.to_datetime(date)
            day = self.get_working_day(self.df, date)
            record = pd.DataFrame([{'date': date}])

            record['month'] = date.month
            record['len_day'] = len(day)
            record['temp_mean'] = day.Temperature.mean()
            record['press_mean'] = day.Pressure.mean()
            record['wind_mean'] = day["Wind Speed"].mean()
            record['Dew_Point_mean'] = day['Dew Point'].mean()

            record['temp_max'] = day.Temperature.max()
            record['press_max'] = day.Pressure.max()
            record['wind_max'] = day["Wind Speed"].max()
            record['Dew_Point_max'] = day['Dew Point'].max()

            record['temp_min'] = day.Temperature.min()
            record['press_min'] = day.Pressure.min()
            record['wind_min'] = day["Wind Speed"].min()
            record['Dew_Point_min'] = day['Dew Point'].min()

            night = self.get_records_night(self.df, date)

            if idx == 0:
                record['len_night'] = 0
            else:
                record['len_night'] = len(night)
                record['night_temp_mean'] = night.Temperature.mean()
                record['night_press_mean'] = night.Pressure.mean()
                record['night_wind_mean'] = night["Wind Speed"].mean()

                record['night_temp_max'] = night.Temperature.max()
                record['night_press_max'] = night.Pressure.max()
                record['night_wind_max'] = night["Wind Speed"].max()

                record['night_temp_min'] = night.Temperature.min()
                record['night_press_min'] = night.Pressure.min()
                record['night_wind_min'] = night["Wind Speed"].min()

            record['Y'] = (day["Clearsky DNI"].sum() +
                           day['DNI'].sum()) / record['len_day']
            records_df = pd.concat([records_df, record])

        records_df.drop(["date"], inplace=True, axis=1)
        records_df.dropna(inplace=True)
        X, Y = self.get_Data_n_records(records_df, DAYS, True)
        scaler = skl.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(records_df)

        return X, Y

    def train_model(self, X, Y):
        Y = pd.Series(Y)
        Y = (Y-Y.min())/(Y.max()-Y.min())
        x_train, x_test, y_train, y_test = train_test_split(X, Y)
        # x_train,x_val,y_train,y_val = train_test_split(x_train,y_train)

        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)
        print(r2_score(y_test.values, rf.predict(x_test)))
        return rf

    def generate_random_sample(self, rf, X):
        rnd = random.randrange(DAYS, len(X))
        x_predict = X[rnd-DAYS:rnd]
        return rf.predict(x_predict)


if __name__ == 'main':
    model = SunlightModel()
    X, Y = model.prepare_data()
    rf = model.train(X, Y)

model = SunlightModel()
X, Y = model.prepare_data()
rf = model.train_model(X, Y)
