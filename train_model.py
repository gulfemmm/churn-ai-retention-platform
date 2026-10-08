import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Veri Seti Oluşturma (2000 Örnek)
np.random.seed(42)
n_samples = 2000

tenure = np.random.randint(1, 120, n_samples)
satisfaction = np.random.randint(1, 6, n_samples)
complain = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
support_calls = np.random.randint(0, 15, n_samples)
cashback = np.random.uniform(10.0, 500.0, n_samples)
order_count = np.random.randint(1, 50, n_samples)
day_since_last_order = np.random.randint(0, 180, n_samples)

# Mantıksal Churn Skoru ve Rastgele Gürültü (Noise) Ekleme
churn_score = (
    (5 - satisfaction) * 2.0 +
    complain * 3.5 +
    support_calls * 0.5 +
    (day_since_last_order / 30) * 1.2 -
    (tenure / 12) * 0.7 -
    (cashback / 100) * 0.4
)

# Aşırı ezberlemeyi önlemek için gürültü ekleme
noise = np.random.normal(0, 1.0, n_samples)
prob = 1 / (1 + np.exp(-(churn_score - 4 + noise)))
churn = (prob > 0.5).astype(int)

# Sütun sırasını app.py ile tamamen eşleşecek şekilde düzenliyoruz
df = pd.DataFrame({
    "Tenure": tenure,
    "SatisfactionScore": satisfaction,
    "Complain": complain,
    "SupportCalls": support_calls,
    "CashbackAmount": cashback,
    "OrderCount": order_count,
    "DaySinceLastOrder": day_since_last_order,
    "Churn": churn
})

# 2. X ve y Ayrımı
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 3. Model Eğitimi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# 4. Performans Kontrolü
y_pred = model.predict(X_test)
print(f"Model Başarımı (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# 5. Dosyaları Kaydetme
joblib.dump(model, "best_churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("\n✅ 'best_churn_model.pkl' ve 'model_columns.pkl' başarıyla yeniden oluşturuldu!")