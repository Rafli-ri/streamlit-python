import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score  # Perbaikan impor accuracy_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Judul aplikasi
st.title("Customer Churn Analysis with Streamlit")

# Import data
df = pd.read_csv('Data Train.csv')

df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge'], axis='columns', inplace=True)

# Konversi kolom 'churn' menjadi biner (0 dan 1)
df['churn_0_1'] = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)

# Pisahkan fitur (X) dan variabel target (y)
X = df.drop(['churn', 'churn_0_1'], axis=1)
y = df['churn_0_1']

# Tentukan fitur kategori untuk one-hot encoding
fitur_kategori = ['state', 'area_code']

# Buat transformer kolom untuk preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', preprocessing.OneHotEncoder(), fitur_kategori)
    ])

# Buat pipeline dengan preprocessing dan regresi logistik
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='saga', max_iter=2000, penalty='l1'))])

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitkan pipeline pada data pelatihan
pipeline.fit(X_train, y_train)

# Prediksi pada data pengujian
y_pred = pipeline.predict(X_test)

# Evaluasi model
akurasi = accuracy_score(y_test, y_pred)
st.write(f"Akurasi: {akurasi}")


# Menampilkan info dataset
st.subheader("Dataset Information")
st.write(df.info())

# Menampilkan dataset
st.subheader("Dataset Preview")
st.write(df.head())

# EDA - Memeriksa Nilai Duplikat
st.subheader("Checking Duplicate Values")
st.write("Number of Duplicate Rows:", df.duplicated().sum())

# EDA - Memeriksa Jumlah Unik Value
st.subheader("Checking Unique Values")
st.write(df.nunique())

# EDA - Mengidentifikasi Korelasi antar Variabel
# st.write("## Heatmap Korelasi")
# fig, ax = plt.subplots(figsize=(14, 9))
# heatmap = sns.heatmap(df.corr(), cmap='Greens', ax=ax)
# st.pyplot(fig)

# EDA - Menghapus Kolom yang Berkorelasi Tinggi
# st.subheader("Removing Highly Correlated Columns")
# df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge'], axis='columns', inplace=True)

# Visualisasi Pie Chart Rasio Churn
st.subheader("Churn Ratio Visualization")
fig, ax = plt.subplots()
ax.axis('equal')
labels = ['Churn', 'No Churn']
churn_counts = df['churn'].value_counts()
ax.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF'])
st.pyplot(fig)

# Preprocessing for Logistic Regression
st.subheader("Logistic Regression Model")
df['churn_0_1'] = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)
X = df.iloc[:, :-2]
y = df.iloc[:, -1]
X = pd.get_dummies(data=X, columns=['state', 'area_code'])
X['international_plan'] = X['international_plan'].apply(lambda x: 1 if x == 'yes' else 0)
X['voice_mail_plan'] = X['voice_mail_plan'].apply(lambda x: 1 if x == 'yes' else 0)

# Standardize input variables
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
y_train = y_train.values.ravel()

# Logistic Regression Model
logistic_model = LogisticRegression(solver='saga', max_iter=2000, penalty='l1')
logistic_model.fit(X_train, y_train)

# Model Accuracy
accuracy = logistic_model.score(X_test, y_test)
st.subheader("Logistic Regression Model Accuracy")
st.write(f"Accuracy: {accuracy:.2%}")

# Visualisasi Barplot Koefisien Terbesar
st.subheader("Top 20 Coefficients Visualization")
temp = pd.DataFrame(list(zip(X.columns, np.absolute(logistic_model.coef_[0]))),
                    columns=['Feature', 'Coefficient']).sort_values('Coefficient', ascending=False).reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=temp.head(20), y='Feature', x='Coefficient', palette='Greens_r')
st.pyplot()

churn_state = df.groupby(['state'])[['churn_0_1']].mean() * 100
churn_state.rename(columns={'churn_0_1':'churn rate (%)'}, inplace=True)
churn_state.reset_index(inplace=True)

# Urutkan dalam urutan menurun berdasarkan rasio churn
churn_state.sort_values(by='churn rate (%)', ascending=False, inplace=True)
churn_state.head()

st.write("## Rasio Churn Mingguan")
fig, ax = plt.subplots(figsize=(16, 6))
sns.barplot(data=churn_state, x='state', y='churn rate (%)', palette='Greens_r', ax=ax)
st.pyplot(fig)

# Rasio Churn berdasarkan Jumlah Pesan Suara
st.write("## Rasio Churn berdasarkan Jumlah Pesan Suara")
st.write("### 1. Perbandingan distribusi number_vmail_messages dan rata-rata menurut churn")
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.boxplot(data=df, x='churn', y='number_vmail_messages', palette='Greens', ax=ax1)
sns.barplot(data=df, x='churn', y='number_vmail_messages', palette='Greens', ax=ax2)
plt.close(2)
plt.close(3)
plt.tight_layout()
st.pyplot(fig)

st.write("### 2. membandingkan ketika voice_mail_plan = 0")
vmail_customer = df[df['voice_mail_plan'] == 'yes']
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.boxplot(data=vmail_customer, x='churn', y='number_vmail_messages', palette='Greens', ax=ax1)
sns.barplot(data=vmail_customer, x='churn', y='number_vmail_messages', palette='Greens', ax=ax2)
plt.close(2)
plt.close(3)
plt.tight_layout()
st.pyplot(fig)


# Visualize distribution of churn and non-churn based on number_vmail_messages using Streamlit
st.write("### 3. Distribusi churn dan non churn berdasarkan number_vmail_messages")
fig1, ax1 = plt.subplots(figsize=(15, 6))
sns.countplot(data=df[df['number_vmail_messages'] != 0], x='number_vmail_messages', hue='churn', palette='Greens_r', ax=ax1)
st.pyplot(fig1)

# Visualize the comparison of churn ratio based on number_vmail_messages using Streamlit
st.write("### 4. Perbandingan rasio churn menurut number_vmail_messages")
fig2, ax2 = plt.subplots(figsize=(15, 6))
sns.barplot(data=df, x='number_vmail_messages', y='churn_0_1', palette='Greens_r', ci=None, ax=ax2)
st.pyplot(fig2)

# Menampilkan informasi tambahan
st.subheader("Additional Information")
st.write("1. High churn rate in states: NJ and CA.")
st.write("2. Low churn rate in states: VA, HI, and AK.")
st.write("3. Customers without international plan have higher churn rate.")
st.write("4. Customers with more than 4 customer service calls have higher churn rate.")

# Menyimpan data untuk analisis lebih lanjut
df.to_csv('processed_data.csv', index=False)