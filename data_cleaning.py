# ===============================
# 1. Import Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# ===============================
# 2. Load Datasets
# ===============================

patients = pd.read_csv("patients.csv")
doctors = pd.read_csv("doctors.csv")
appointments = pd.read_csv("appointments.csv")
treatments = pd.read_csv("treatments.csv")
billing = pd.read_csv("billing.csv")

print("Datasets Loaded Successfully")


# ===============================
# 3. Basic Data Inspection
# ===============================

print(patients.head())
print(doctors.head())
print(appointments.head())
print(treatments.head())
print(billing.head())

print(patients.info())
print(doctors.info())
print(appointments.info())
print(treatments.info())
print(billing.info())


# ===============================
# 4. Remove Duplicate Rows
# ===============================

patients = patients.drop_duplicates()
doctors = doctors.drop_duplicates()
appointments = appointments.drop_duplicates()
treatments = treatments.drop_duplicates()
billing = billing.drop_duplicates()


# ===============================
# 5. Handle Missing Values
# ===============================

patients['contact_number'] = patients['contact_number'].fillna("Unknown")
patients['email'] = patients['email'].fillna("not_provided")

doctors['phone_number'] = doctors['phone_number'].fillna("Unknown")
doctors['email'] = doctors['email'].fillna("not_provided")

patients['insurance_provider'] = patients['insurance_provider'].fillna("None")
patients['insurance_number'] = patients['insurance_number'].fillna("None")

treatments['description'] = treatments['description'].fillna("No description")


# ===============================
# 6. Convert Date Columns
# ===============================

patients['registration_date'] = pd.to_datetime(patients['registration_date'], errors='coerce')
patients['date_of_birth'] = pd.to_datetime(patients['date_of_birth'], errors='coerce')

appointments['appointment_date'] = pd.to_datetime(appointments['appointment_date'], errors='coerce')

treatments['treatment_date'] = pd.to_datetime(treatments['treatment_date'], errors='coerce')

billing['bill_date'] = pd.to_datetime(billing['bill_date'], errors='coerce')


# ===============================
# 7. Clean Text Columns
# ===============================

patients['first_name'] = patients['first_name'].str.strip()
patients['last_name'] = patients['last_name'].str.strip()

doctors['first_name'] = doctors['first_name'].str.strip()
doctors['last_name'] = doctors['last_name'].str.strip()

patients['gender'] = patients['gender'].str.upper()

patients['email'] = patients['email'].str.lower()
doctors['email'] = doctors['email'].str.lower()


# ===============================
# 8. Validate Numeric Values
# ===============================

treatments = treatments[treatments['cost'] >= 0]
billing = billing[billing['amount'] >= 0]


# ===============================
# 9. Merge All Tables
# ===============================

df = appointments.merge(patients, on="patient_id", how="left")

df = df.merge(doctors, on="doctor_id", how="left")

df = df.merge(treatments, on="appointment_id", how="left")

df = df.merge(billing, on="treatment_id", how="left")


# ===============================
# 10. Feature Engineering
# ===============================

# Patient Age
current_year = pd.Timestamp.now().year
df['age'] = current_year - df['date_of_birth'].dt.year


# Age Groups
df['age_group'] = pd.cut(df['age'],
                         bins=[0,18,35,50,70,100],
                         labels=["Child","Young Adult","Adult","Senior","Elderly"])


# Appointment Month
df['appointment_month'] = df['appointment_date'].dt.month


# Appointment Day
df['appointment_day'] = df['appointment_date'].dt.day_name()


# Registration Year
df['registration_year'] = df['registration_date'].dt.year


# Treatment Cost Category
df['cost_category'] = pd.cut(df['cost'],
                             bins=[0,100,500,1000,5000],
                             labels=["Low","Medium","High","Very High"])


# ===============================
# 11. Exploratory Data Analysis
# ===============================

print("Dataset Shape:", df.shape)
print(df.describe())


# ===============================
# Gender Distribution
# ===============================

sns.countplot(x='gender', data=df)
plt.title("Patient Gender Distribution")
plt.show()


# ===============================
# Age Distribution
# ===============================

plt.hist(df['age'], bins=20)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# ===============================
# Age Groups
# ===============================

sns.countplot(x='age_group', data=df)
plt.title("Age Group Distribution")
plt.show()


# ===============================
# Appointment Status
# ===============================

sns.countplot(x='status', data=df)
plt.title("Appointment Status")
plt.show()


# ===============================
# Doctor Specialization
# ===============================

sns.countplot(y='specialization', data=df)
plt.title("Doctor Specialization Distribution")
plt.show()


# ===============================
# Most Common Treatments
# ===============================

treatments_count = df['treatment_type'].value_counts()

treatments_count.plot(kind='bar')
plt.title("Most Common Treatments")
plt.show()


# ===============================
# Revenue by Payment Method
# ===============================

revenue_payment = df.groupby("payment_method")["amount"].sum()

revenue_payment.plot(kind='bar')
plt.title("Revenue by Payment Method")
plt.show()


# ===============================
# Monthly Appointments
# ===============================

appointments_month = df.groupby("appointment_month").size()

appointments_month.plot(kind='line', marker='o')
plt.title("Monthly Appointments")
plt.show()


# ===============================
# Doctor Workload
# ===============================

doctor_workload = df.groupby("doctor_id").size()

doctor_workload.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Doctors by Appointments")
plt.show()


# ===============================
# Billing Status
# ===============================

sns.countplot(x='payment_status', data=df)
plt.title("Billing Payment Status")
plt.show()


# ===============================
# Correlation Heatmap
# ===============================

numeric_cols = df[['age','cost','amount','years_experience']]

sns.heatmap(numeric_cols.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


# ===============================
# 12. Save Cleaned Dataset
# ===============================

df.to_csv("hospital_final_dataset.csv", index=False)

print("Final dataset saved successfully")
df.to_csv("hospital_final_dataset.csv", index=False)