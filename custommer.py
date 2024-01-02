import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score  # Perbaikan impor accuracy_score
import streamlit as st
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Judul aplikasi
st.title("Customer Churn Analysis with Streamlit")

# Menampilkan informasi tambahan
st.subheader("Additional Information")
st.write("1. High churn rate in states: NJ and CA.")
st.write("2. Low churn rate in states: VA, HI, and AK.")
st.write("3. Customers without international plan have higher churn rate.")
st.write("4. Customers with more than 4 customer service calls have higher churn rate.")


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
st.write("### Number of Duplicate Rows:", df.duplicated().sum())

# EDA - Memeriksa Jumlah Unik Value
st.subheader("Checking Unique Values")
st.write(df.nunique())

st.subheader("Describe")
st.write(df.head())

# EDA - Mengidentifikasi Korelasi antar Variabel
# st.write("## Heatmap Korelasi")
# fig, ax = plt.subplots(figsize=(14, 9))
# heatmap = sns.heatmap(df.corr(), cmap='Greens', ax=ax)
# st.pyplot(fig)

# EDA - Menghapus Kolom yang Berkorelasi Tinggi
# st.subheader("Removing Highly Correlated Columns")
# df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge'], axis='columns', inplace=True)

# Visualisasi Pie Chart Rasio Churn
st.subheader("Mengidentifikasi variabel Kunci untuk Memprediksi Churn")
st.write("Menghitung koefisien masing-masing variabel dengan reresi logistik untuk menentukan variabel mana yang penting untuk memprediksi churn")
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

st.write("## Rasio Churn per minggu")
st.write("## Rasio Churn Mingguan")
fig, ax = plt.subplots(figsize=(16, 6))
sns.barplot(data=churn_state, x='state', y='churn rate (%)', palette='Greens_r', ax=ax)
st.pyplot(fig)
st.write("- NJ menjadi state dengan jumlah chunners terbanyak.\n",
        "- VA, HI, dan AK ternyata merupakan state dengan sedikit chunners "
      )

st.subheader("Tingkat Churn berdasarkan Plan")
st.write("voice_mail_plan, international_plan merupakan tingkat churn berdasarkan jenis customer")
st.write("1. Tingkat churn dengan atau tanpa voice_mail_plan")

# st.bar_chart(df.groupby('voice_mail_plan')['churn_0_1'].mean())
fig, ax = plt.subplots()
sns.barplot(data=df, x='voice_mail_plan', y='churn_0_1', palette='Greens', ax=ax)
st.pyplot(fig)

st.write("Tingkat customer churn yang tidak menggunakan voice_mail_plan adalah 16%, sedangkan tingkat customer churn yang menggunakan voice_mail_plan adalah sekitar 7%.\n",
         "Uji-t apakah perbedaan rasio signifikan\n",
         "(Sebenarnya rasio bisa dikatakan konsep yang sama dengan mean, jadi boleh saja menggunakan uji-t seperti perbedaan mean)\n"
         )

# Streamlit app
st.write('### Uji Varian menggunakan Levene')
# Split the data into two groups based on 'voice_mail_plan'
group_yes = df[df['voice_mail_plan'] == 'yes']['churn_0_1']
group_no = df[df['voice_mail_plan'] == 'no']['churn_0_1']
# Perform Levene's test
lev_result = stats.levene(group_yes, group_no)
# Display the result in Streamlit
st.write(f'LeveneResult(F): {lev_result.statistic:.2f}')
st.write(f'p-value: {lev_result.pvalue:.3f}')

st.write('### Menjalankan uji-t sampel independen dengan heterogenitas')
t_result = stats.ttest_ind(group_yes, group_no, equal_var=False)
st.write(f't statistic : {t_result.statistic:.2f}')
st.write(f'p-value  {t_result.pvalue:.3f}')
st.write("p < 0.01 dan terdapat perbedaan yang cukup jauh bahkan jika dilihat secara visual, sehingga dapat dikakatan bahwa customer yang tidak menggunakan voice_mail_plan memiliki churn rate yang lebih tinggi")


st.write("2. Tingkat churn berdasarkan international_plan")
fig, ax = plt.subplots()
sns.barplot(data=df, x='international_plan', y='churn_0_1', palette='Greens', ax=ax)
st.pyplot(fig)

st.write("Tingkat customer churn yang tidak menggunakan international_plan sekitar 11%,\n"
          "sedangkan tingkat customer churn yang menggunakan international_plan sekitar 42%\n",
         )
# Uji-t apakah perbedaan rasio signifiikan
temp1 = df[df['international_plan'] == 'yes']['churn_0_1']
temp2 = df[df['international_plan'] == 'no']['churn_0_1']
 # Uji varian dengan Levene
lev_result = stats.levene(temp1, temp2)
st.write(f't statistic : {lev_result.statistic:.2f}')
st.write(f'p-value  {lev_result.pvalue:.3f}')

# Menjalankan uji-t sampel independen dengan heterogenitas
t_result = stats.ttest_ind(temp1, temp2, equal_var=False)
st.write(f't statistic : {t_result.statistic:.2f}')
st.write(f'p-value  {t_result.pvalue:.3f}')
st.write("- p < 0.01 dan terdapat perbedaan yang besar walaupun dilihat secara visual, sehingga dapat dikatakan bahwa customer yang menggunakan international_plan memiliki churn rate yang lebih tinggi\n"
"- Tingkat customer churn yang menggunakan international_plan sebesar 42%, maka perlu memeriksaa faktor ketidakpuasan keseluruhan dari plan tersebut")




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
st.write("Tampak bahwa number_vmail_messages adalah 0 dalam banyak kasus, karena voice_mail_plan tidak digunakan oleh pelanggan churn")

st.write("### 2. membandingkan ketika voice_mail_plan = 0")
st.write("Jika voice_mail_plan tidak digunakan, vmail_messsages adalah 0")
vmail_customer = df[df['voice_mail_plan'] == 'yes']
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.boxplot(data=vmail_customer, x='churn', y='number_vmail_messages', palette='Greens', ax=ax1)
sns.barplot(data=vmail_customer, x='churn', y='number_vmail_messages', palette='Greens', ax=ax2)
plt.close(2)
plt.close(3)
plt.tight_layout()
st.pyplot(fig)
st.write("Saat membandingkan hanya customer yang menggunakan voice_mail_plan, dapat dipastikan bahwa number_vmail_messages tidak berubah secara signifikan tergantung apakah mereka churn atau tidak churn")


# Visualize distribution of churn and non-churn based on number_vmail_messages using Streamlit
st.write("### 3. Distribusi churn dan non churn berdasarkan number_vmail_messages")
st.write("Visualisasi kecuali jika number_vmail_messages = 0 (jumlah kasus sangat banyak 0)")
fig1, ax1 = plt.subplots(figsize=(15, 6))
sns.countplot(data=df[df['number_vmail_messages'] != 0], x='number_vmail_messages', hue='churn', palette='Greens_r', ax=ax1)
st.pyplot(fig1)

# Visualize the comparison of churn ratio based on number_vmail_messages using Streamlit
st.write("### 4. Perbandingan rasio churn menurut number_vmail_messages")
fig2, ax2 = plt.subplots(figsize=(15, 6))
sns.barplot(data=df, x='number_vmail_messages', y='churn_0_1', palette='Greens_r', ci=None, ax=ax2)
st.pyplot(fig2)
st.write("- number_vmail_messages yang rendah tidak berarti banyak, kecuali jika vnumber_vmail_messages = 0\n",
        "- Tampaknya sulit untuk menentukan apakah customer churn hanya dengan number_vmail_messages"
      )



# Menyimpan data untuk analisis lebih lanjut
df.to_csv('processed_data.csv', index=False)