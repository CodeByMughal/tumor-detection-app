import pandas as pd
import numpy as np

# Simulate a small dataset of patient scan records
data = {
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'age':        [45, 62, 38, 55, 70],
    'scan_type':  ['Brain MRI', 'Lung CT', 'Brain MRI', 'Hip MRI', 'Lung CT'],
    'diagnosis':  ['Tumor', 'Normal', 'Normal', 'AVN Stage 2', 'Tumor'],
    'confidence': [0.94, 0.87, 0.91, 0.88, 0.96]
}

df = pd.DataFrame(data)

print("=== All patient records ===")
print(df)

print("\n=== Tumor patients only ===")
print(df[df['diagnosis'].str.contains('Tumor')])

print("\n=== AVN patients only ===")
print(df[df['diagnosis'].str.contains('AVN')])

print("\n=== Scan type counts ===")
print(df['scan_type'].value_counts())

print("\n=== Average confidence per diagnosis ===")
print(df.groupby('diagnosis')['confidence'].mean())