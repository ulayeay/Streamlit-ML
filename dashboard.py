import streamlit as st
import pandas as pd
from PIL import Image
import os


st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Malaria Dataset")
st.write("""
    Dataset yang digunakan dalam tugas besar ini diperoleh dari repositori Kaggle, dengan nama “cell-images-for-detecting-malaria”. Dataset ini dikembangkan untuk tujuan klasifikasi malaria berdasarkan citra mikroskopis sel darah manusia dan telah banyak digunakan sebagai benchmark dalam penelitian Machine Learning di bidang medis.
    \n 
         Isi DataSet:
""")
DATA_DIR = "cell_images"
@st.cache_data
def load_image_paths():
    classes = ["Parasitized", "Uninfected"]
    data = []
    for label in classes:
        folder = os.path.join(DATA_DIR, label)
        for img_file in os.listdir(folder):
            if img_file.endswith(".png"):
                data.append({
                    "filename": img_file,
                    "label": label,
                    "path": os.path.join(folder, img_file)
                })
    return pd.DataFrame(data)

df = load_image_paths()
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Images", len(df))
with col2:
    st.metric("Classes", len(df["label"].unique()))
st.subheader("Class Distribution")
st.bar_chart(df["label"].value_counts())
st.subheader("Image Preview")
selected_class = st.selectbox("Choose a class", df["label"].unique())
num_images = st.slider("Number of images to display", 1, 10, 5)
try:
    filtered_df = df[df["label"] == selected_class].sample(num_images)
    cols = st.columns(num_images)
    for i, row in enumerate(filtered_df.itertuples()):
        with cols[i]:
            img = Image.open(row.path)
            st.image(img, caption=row.filename, use_container_width=True)
except ValueError:
    st.warning(f"Not enough images in the {selected_class} class to display {num_images} samples.")

#tabel 
st.markdown("---") 
st.header("📊 Model Comparison")
st.write("""
    Sebelum berakhir memilih Random Forest, kami mencoba beberapa model yang cukup lazim digunakan untuk image classification, yaitu SVM, CNN dan Random Forest berikut adalah perbandingan Model SVM dan Random Forest:

    **Perbandingan SVM dan Random Forest:**
""")
comparison_data = {
    "Model": ["Random Forest", "Support Vector Machine", "CNN"],
    "Accuracy": ["97.00%", "69.00%", "94.00%"],
    "Weighted Avg Precision": ["0.97", "0.69", "0.95"],
    "Weighted Avg Recall": ["0.97", "0.69", "0.94"],
    "Weighted Avg F1-Score": ["0.97", "0.69", "0.94"]
}
comparison_df = pd.DataFrame(comparison_data)
st.table(comparison_df)
st.write("""
    Berdasarkan perbandingan hasil diatas, kami mengeliminasi SVM dari pilihan kami dan mencoba melakukan eksplorasi untuk CNN dan Random Forest. Tetapi, karena banyaknya existing paper mengenai Malaria Classification dengan CNN dan waktu training yang lebih lama, kami memutuskan untuk menggunakan Random Forest.
""")

#baseline 
st.markdown("---")
st.header("🔧 Random Forest Base Line Model")
st.write("""
        Kami membuat baseline atau model sederhana dengan parameter sederhana dengan tahapan sebagai berikut: 
         

        **1. Pengambilan Data dan Eksplorasi Awal**
         
         mengunduh dataset dengan `KaggleHub` dan menampilkan citra sejumlah citra untuk masing-masing kelas
         
        **2. Ektraksi Fitur (Contour Area)**
         
         * Membaca citra dan menerapkan Gaussian Blur untuk mengurangi noise 
         * mengubah citra menjadi abu-abu,
         * menerapkan tresholding sederhana untuk mengubah citra menjadi blur 
         * dilanjut dengan mengidentifikasi kontur dan menghitung luas 5 kontur terbesar dalam citra.
        
         
        **3. Persiapan Data untuk Model Training**

         Data dibagi menjadi fitur (X) dan target (y). Kemudian dipisahkan menjadi dataset training sebesar 75% dan dataset testing 25%, pembagian ini dilakukan secara stratifikasi untuk memastikan proposi kelasnya sama dikedua set dan mengurangi kemungkinan terjadinya bias.


         **4. Training**
         Parameter yang digunakan adalah 
         `n_estimators=100`


         **5. Evaluation** 
         Didapatkan bahwa akurasi model baseline adalah 88%, dengan precision 0.88, recall 0.88 dan f1 score 0.88. 

    """)



# bandingin hyoeparameter
st.markdown("---") 
st.header("⚙️ Hyperparameter Exploration")
st.write("Kami melakukan 3 macam eksplorasi hyperparameter tuning, dengan menggunakan GridSearchCV, Bayesian Optimization, dan RandomizedSearchCV.")

col_hp1, col_hp2, col_hp3 = st.columns(3)

with col_hp1:
    st.subheader("GridSearchCV")
    st.write("""
        GridSearchCV bekerja dengan mencoba semua kombinasi dari kamus hyperparameter atau parameter grid dengan menggunakan cross-validation, dan menentukan mana kombinasi yang memiliki hasil paling baik.
            Parameter grid yang digunakan adalah sebagai berikut:
            \n `n_estimator = [50, 100, 200]`
            \n `max_depth = [None, 10, 20]`
            \n `min_samples_split = [2, 5]`
            \n `min_sampples_leaf = [1, 2]`
            \n Dari proses tadi, ditemukan bahwa hyperparameter terbaik adalah sebagai berikut:
            \n `max_depth= None`
            \n `min_samples_leaf=1`
            \n `min_samples_split=2`
            \n `n_estimators=200`  
        \n**Impact:** dengan memaksimalkan support, Random Forest dengan GridSearchCV berhasil meraih accuracy sebesar 97%
    """)

with col_hp2:
    st.subheader("Bayesian Optimization Optuna")
    st.write("""
         Optimasi hyperparameter pada eksperimen ini menggunakan Optuna, sebuah pustaka open-source untuk Bayesian optimization. Optuna bekerja dengan membangun model probabilistik untuk memprediksi performa hyperparameter dan mengeksplorasi kombinasi parameter yang diperkirakan akan menghasilkan skor terbaik.
            \n Optuna kemudian menjalankan pencarian kombinasi parameter secara cerdas berdasarkan hasil percobaan sebelumnya (Bayesian approach). Setelah iterasi selesai, Optuna menghasilkan kombinasi hyperparameter terbaik yaitu:
            \n `n_estimators=113`  
            \n `max_depth= 9`
            \n `min_samples_leaf=3`
            \n `min_samples_split=7`
            
        \n**Impact:** Dengan sistem cerdasnya, optuna berhasil mendapatkan akurasi sebesar 90%
    """)

with col_hp3:
    st.subheader("RandomizedSearchCV")
    st.write("""
         RandomizedSearchCV bekerja untuk mencari hyperparameter terbaik, hal ini dilakukan secara random, namun diaplikasikan RandomForestClassifier `random_state=4` agar pembagian data selalu sama. Selain itu diaplikasikan juga objek StratifiedKFold cv untuk cross-validation, dengan `n_splits=5` yang berarti data akan dibagi menjadi 5 lipatan, 4 untuk pelatihan dan 1 untuk validasi.
             Parameter grid yang digunakan adalah sebagai berikut:
            \n `n_estimator = radiant(200, 1000)`
            \n `max_depth = [None] + list(range(10,60,10))`
            \n `min_samples_split = radiant(2,20)`
            \n `min_sampples_leaf = radiant(1,10)`
            \n `max_features = ["sqrt", "log2", None]`
            \n `bootstrap = [True]`
            \n `class_weight = [None, "balanced", "balanced_subsample"]`
            \n `criterion = ["gini", "entropy", "log_loss"]`
            \n Dari proses tadi, ditemukan bahwa hyperparameter terbaik adalah sebagai berikut:
            \n `bootstrap = True`
            \n `class_weight = 'balanced_subsample'`
            \n `min_samples_leaf=4`
            \n `min_samples_split=17`
            \n `n_estimators=810`  
            \n `max_depth = 10`
            \n `max_features = 'sqrt'`
            \n `criterion = 'entropy'`  
        \n**Impact:** dengan mengaplikasikan proses RandomizedSearchCV dalam hyperparameter tuning, akurasi model sebesar 90% berhasil didapatkan
    """)