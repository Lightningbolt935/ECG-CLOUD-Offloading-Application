import pandas as pd
import wfdb
import numpy as np

# Check actual signal units and values
rec = wfdb.rdrecord('./mitdb/100')
print("Units:", rec.units)
print("ADC gain:", rec.adc_gain)
print("Baseline:", rec.baseline)
print("Signal range (mV):", rec.p_signal[:,0].min(), "to", rec.p_signal[:,0].max())
print("Std:", rec.p_signal[:,0].std())

# Check what our ST scores look like for a known normal record
df = pd.read_csv('./task_profiles.csv')
rec100 = df[df['record_id']==100]
print("\nRecord 100 (normal) ST deviation stats:")
print(rec100['st_deviation'].describe())
print("\nRecord 207 (VF) ST deviation stats:")
rec207 = df[df['record_id']==207]
print(rec207['st_deviation'].describe())