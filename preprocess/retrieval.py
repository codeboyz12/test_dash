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

def recently_mock():
    np.random.seed(42)

    # สร้าง DataFrame: 20 แถว × 10 คอลัมน์
    df_mock = pd.DataFrame(
        np.random.randint(0, 100, size=(20, 10)),  # จำนวนเต็มสุ่ม 0–99
        columns=[f"col_{i}" for i in range(1, 11)] # ตั้งชื่อ col_1 ... col_10
    )

    return df_mock