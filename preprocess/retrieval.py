import pandas as pd
import numpy as np

end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(months=6)

date_rng = pd.date_range(start=start_date, end=end_date, freq='4H')
values = np.random.randn(len(date_rng))

def get_data():
    return pd.DataFrame({'datetime': date_rng, 'value': values})

def prediction():
    base_end = end_date + pd.Timedelta(days=2)
    start_date_pred = base_end
    end_date_pred = base_end + pd.DateOffset(months=1)

    date_rng_pred = pd.date_range(start=start_date_pred, end=end_date_pred, freq='4H')
    values_pred = np.random.randn(len(date_rng_pred))
    return pd.DataFrame({'datetime': date_rng_pred, 'value': values_pred})
