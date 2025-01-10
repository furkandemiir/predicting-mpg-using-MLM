import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Veri setini yükle ve temizle
df = sns.load_dataset('mpg')
df = df.dropna()  # Eksik değerleri temizle

# Bağımsız ve bağımlı değişkenleri seç
X = df[['horsepower', 'acceleration', 'weight']]
y = df['mpg']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 2. KNN Modeli
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"KNN Model MSE: {mse_knn}")

# 3. Random Forest Modeli
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Model MSE: {mse_rf}")

# 4. ANN Modeli
ann_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)
mse_ann = mean_squared_error(y_test, y_pred_ann)
print(f"ANN Model MSE: {mse_ann}")

# 5. Modellerin karşılaştırılması
mse_values = {'KNN': mse_knn, 'Random Forest': mse_rf, 'ANN': mse_ann}
best_model_name = min(mse_values, key=mse_values.get)
print(f"En başarılı model: {best_model_name}")

# 6. Belirli bir otomobil için mpg tahmini
sample = pd.DataFrame([[130, 13, 3500]], columns=['horsepower', 'acceleration', 'weight'])
if best_model_name == 'KNN':
    predicted_mpg = knn_model.predict(sample)
elif best_model_name == 'Random Forest':
    predicted_mpg = rf_model.predict(sample)
else:
    predicted_mpg = ann_model.predict(sample)

print(f"Predicted MPG for the car (horsepower=130, acceleration=13, weight=3500): {predicted_mpg[0]}")
print("Eğitim seti boyutu:", len(X_train + y_train))
print("Test seti boyutu:", len(X_test + y_test))

# 7. Modellerin tahminlerini görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, color='blue', label='KNN Tahminleri')
plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest Tahminleri')
plt.scatter(y_test, y_pred_ann, color='green', label='ANN Tahminleri')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Gerçek MPG')
plt.ylabel('Tahmini MPG')
plt.legend()
plt.title('Farklı Modeller İçin Gerçek ve Tahmini MPG')
plt.show()


# Test verileriyle tahminler
knn_predictions = knn_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
ann_predictions = ann_model.predict(X_test)

# Performans metrikleri
knn_mse = mean_squared_error(y_test, knn_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
ann_mse = mean_squared_error(y_test, ann_predictions)

knn_r2 = r2_score(y_test, knn_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
ann_r2 = r2_score(y_test, ann_predictions)

# Performans metriklerini içeren tablo
results = {
    "Model": ["KNN", "Random Forest", "ANN"],
    "MSE": [knn_mse, rf_mse, ann_mse],
    "R2 Score": [knn_r2, rf_r2, ann_r2]
}

results_df = pd.DataFrame(results)
print("Model Performans Karşılaştırması:")
print(results_df)

# Performans metriklerini görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# MSE grafiği
ax[0].bar(results_df["Model"], results_df["MSE"], color=["blue", "green", "orange"])
ax[0].set_title("Model MSE Karşılaştırması")
ax[0].set_ylabel("Mean Squared Error (MSE)")

# R2 Score grafiği
ax[1].bar(results_df["Model"], results_df["R2 Score"], color=["blue", "green", "orange"])
ax[1].set_title("Model R2 Score Karşılaştırması")
ax[1].set_ylabel("R2 Score")

plt.tight_layout()
plt.show()

# Korelasyon matrisi
correlation_matrix = df[['horsepower', 'acceleration', 'weight', 'mpg']].corr()

# Isı haritası
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi - Özellikler ve mpg")
plt.show()

# Boxplot: MPG'in ağırlık ve beygir gücü ile ilişkisi
plt.figure(figsize=(12, 6))
sns.boxplot(x="origin", y="mpg", data=df)
plt.title("MPG'in Araç Menşei Bazında Dağılımı (Boxplot)")
plt.ylabel("MPG")
plt.xlabel("Menşei")
plt.show()

# Histogram: MPG Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(df['mpg'], kde=False, bins=20, color="blue")
plt.title("MPG Dağılımı (Histogram)")
plt.xlabel("MPG")
plt.ylabel("Frekans")
plt.show()

# Yoğunluk Grafiği: Ağırlık ve MPG İlişkisi
plt.figure(figsize=(10, 6))
sns.kdeplot(x=df['weight'], y=df['mpg'], cmap="viridis", fill=True)
plt.title("Ağırlık ve MPG İlişkisi (Yoğunluk Grafiği)")
plt.xlabel("Ağırlık")
plt.ylabel("MPG")
plt.show()
