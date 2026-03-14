import pandas as pd

patients = pd.read_csv("patients.csv")
doctors = pd.read_csv("doctors.csv")
appointments = pd.read_csv("appointments.csv")
treatments = pd.read_csv("treatments.csv")
billing = pd.read_csv("billing.csv")

print(patients.head())