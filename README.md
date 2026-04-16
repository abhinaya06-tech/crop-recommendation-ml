# 🌾 Crop Recommendation System (ML + Data Analysis)

A machine learning-based crop recommendation system that suggests the most suitable crops based on soil nutrients and environmental conditions.

---

## 📌 Features

* Predicts **Top 3 suitable crops** using ML
* Uses **Random Forest Classifier**
* Includes **Exploratory Data Analysis (EDA)**
* Visualizes:

  * Crop distribution
  * Feature importance
* Built with **Streamlit UI**

---

## 📊 Key Insights

* Rainfall and humidity are dominant factors
* Soil nutrients (N, P, K) influence crop selection
* Model provides probabilistic recommendations

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

---

## ⚠️ Limitations

* Uses generalized dataset (not region-specific)
* Does not include economic or seasonal factors
* Should be used as a **decision-support tool**, not final authority

---

## 📈 Future Improvements

* Use region-specific agricultural data
* Add weather API integration
* Include market price analysis

---

## 👤 Author

* Abhinaya
