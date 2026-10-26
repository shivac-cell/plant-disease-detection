# PlantCare AI: Plant Disease Detection & Smart Pesticide and Fertilizer Recommendation System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.39%25-success)](models/candidates/evaluation_report.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, bilingual agronomic decision-support platform combining deep learning computer vision for foliar disease detection with source-verified, statutory pesticide protocols and balanced crop nutrition advice.

Powered by a fine-tuned **MobileNetV2** architecture and an authoritative agricultural recommendation knowledge base standardized under **ICAR**, **ICAR-NCIPM**, **TNAU**, and **CIBRC** guidelines.

---

## 🌟 Key Features

### 1. 🌿 Interactive AI Diagnostic Dashboard (`app.py`)
- **Leaf Specimen Upload & Preloaded Samples**: Upload JPG/PNG leaf photos or quickly test preloaded specimens across all disease classes.
- **Deep Learning Neural Inference**: MobileNetV2 architecture with real-time classification, confidence percentages, and top-k probability distributions.
- **🛡️ CIBRC-Approved Pesticide Protocols**: Statutory chemical treatments, active ingredients, formulations, spray dosages, and Pre-Harvest Intervals (PHI).
- **🌿 Cultural IPM & Sanitation Checklist**: Non-chemical cultural management, bio-control agents (*Trichoderma*, *Ampelomyces*), and seed sanitation.
- **🧪 Crop Nutrition & Fertilizer Guidance**: Basal NPK, secondary nutrients (Sulfur), and foliar recovery sprays (DAP, MOP, Pulse Wonder).
- **📋 Blackgram Recommendation Dataset Explorer**: Searchable, filterable interactive table with instant **CSV download button**.
- **📊 Benchmark Evaluation Metrics**: Held-out test performance charts, 5x5 confusion matrix, and transfer learning convergence curves.
- **🧮 Smart Organic Fertilizer Calculator**: Custom farmyard manure (FYM), vermicompost, and bio-fertilizer acreage estimator.

### 2. 🌐 Full Bilingual Interface (English & Kannada / ಕನ್ನಡ)
- Native bilingual toggling across both the Streamlit dashboard and Flask web application.
- Natural Kannada translations tailored for local farming communities, preserving botanical names and chemical formulations alongside regional terms.

### 3. ⚖️ Strict Agronomic Separation Rule
- **Pesticide recommendations and fertilizer guidance are kept completely separate**.
- The system prominently enforces the agricultural reality: **Fertilizers provide crop nutrition and soil fertility — they do NOT cure viral or fungal plant diseases**.
- Explicit fallback handling: Any unknown pathogen or uncertain diagnosis directs the user to **"Consult an agricultural expert"** or a local Krishi Vigyan Kendra (KVK).

### 4. 🏛️ Flask Web Portal (`flask_app.py`)
- Enterprise modular architecture with user authentication, cryptographic password hashing (`scrypt`), CSRF protection, and MySQL diagnostic persistence.

---

## 🎯 Supported Crop & Disease Conditions

The production model detects 5 canonical conditions in **Blackgram (*Vigna mungo* / Urad Bean)**:

| Class Key | Condition & Causal Organism | Category | CIBRC Treatment / Protocol |
| :--- | :--- | :--- | :--- |
| `anthracnose` | **Anthracnose** (*Colletotrichum lindemuthianum*) | Fungal Blight | Carbendazim 50% WP @ 1.0 g/L OR Mancozeb 75% WP @ 2.0 g/L (PHI: 15–20 days) |
| `healthy` | **Healthy Foliage** (*Normal crop physiology*) | Healthy | No chemical sprays required. Preserve natural predatory fauna and honeybees. |
| `leaf_crinckle` | **Leaf Crinkle Disease** (*Urdbean Leaf Crinkle Virus - ULCV*) | Viral Pathogen | Systemic virus (no chemical cure). Rogue infected plants. Vector spray: Imidacloprid 17.8% SL @ 0.3 ml/L. |
| `powdery_mildew` | **Powdery Mildew** (*Erysiphe polygoni DC*) | Fungal Mildew | Wettable Sulfur 80% WP @ 2.5–3.0 g/L OR Hexaconazole 5% EC @ 1.0 ml/L (PHI: 15–20 days). |
| `yellow_mosaic` | **Yellow Mosaic Disease** (*Mungbean Yellow Mosaic Virus - MYMV*) | Viral Pathogen | Non-curable chemically. Rogue out. Vector control: Diafenthiuron 50% WP @ 1.0 g/L OR Thiamethoxam 25% WG @ 0.25 g/L. |

---

## 📊 Model Architecture & Quantitative Performance

- **Base Convolutional Backbone**: MobileNetV2 (ImageNet weights, fine-tuned on pulse leaves)
- **Parameters**: 2,264,453 parameters
- **Input Resolution**: `224 × 224 × 3` RGB, normalized to `[-1, 1]`
- **Model File**: `plant_disease_model.keras` (Native Keras 3 Archive, **22.78 MB**)

### Held-Out Test Evaluation (152 Test Images)
- **Test Accuracy**: **95.39%**
- **Macro F1-Score**: **95.10%**
- **Macro Precision**: **95.71%**
- **Macro Recall**: **95.02%**
- **Average CPU Latency**: **~88.6 ms / image**

```
Confusion Matrix (152 Test Images):
               Predicted
                Anthracnose  Healthy  Leaf_Crinkle  Powdery_Mildew  Yellow_Mosaic
Anthracnose         35          0          0               0              0
Healthy              0         31          0               3              0
Leaf Crinkle         1          0         20               1              1
Powdery Mildew       0          0          0              27              0
Yellow Mosaic        0          0          0               1             32
```

---

## 📁 Project Directory Structure

```text
plant_disease_detection/
├── .env.example                     # Environment configuration template
├── .gitignore                       # Standardized Git exclusion rules
├── LICENSE                          # MIT Open-Source License
├── README.md                        # Project documentation and setup guide
├── requirements.txt                 # Python dependencies
├── app.py                           # Streamlit Interactive AI Dashboard
├── flask_app.py                     # Flask Web Application entry point
├── config.py                        # Centralized application configuration
│
├── plant_disease_model.keras        # Production MobileNetV2 model (22.78 MB)
├── class_names.json                 # Model class mapping (5 classes)
├── blackgram_recommendations.csv    # Source-verified CIBRC/ICAR recommendation dataset
│
├── database/                        # Database layer
│   ├── db.py                        # MySQL connection pooling and health checks
│   ├── schema.sql                   # MySQL relational schema (users, predictions, recommendations)
│   ├── seed.py                      # Database seeder script
│   └── migrations/                  # Schema migrations
│
├── frontend/                        # Web assets & templates
│   ├── static/css/                  # CSS stylesheets (Agrotech theme)
│   ├── static/js/                   # Client-side JavaScript
│   └── templates/                   # Jinja2 HTML templates (bilingual)
│
├── models/                          # Machine learning definitions
│   ├── disease_classifier.py        # MobileNetV2 architecture builder
│   ├── candidates/                  # Evaluation reports, manifests & training histories
│   │   ├── evaluation_report.json   # Held-out benchmark metrics
│   │   ├── training_history.csv     # Training loss/accuracy progression
│   │   └── class_counts.csv         # Class distribution summary
│   └── saved_model/                 # Saved model directory
│
├── routes/                          # Flask blueprints
│   ├── auth_routes.py               # User authentication routes
│   └── main_routes.py               # Image diagnosis & history routes
│
├── services/                        # Business logic services
│   ├── image_preprocessing.py       # Bilinear resizing, RGB conversion & normalization
│   ├── model_service.py             # Model loading & inference
│   ├── prediction_service.py        # Prediction orchestration & persistence
│   └── recommendation_service.py   # Agricultural advisory lookup
│
├── translations/                    # Bilingual localization dictionaries
│   ├── en.json                      # English interface strings
│   ├── kn.json                      # Kannada (ಕನ್ನಡ) interface strings
│   └── i18n.py                      # Translation lookup helper
│
├── tests/                           # Automated test suite
│   ├── test_recommendations.py      # E2E recommendation & model inference tests
│   ├── test_predictions.py          # Prediction pipeline tests
│   └── test_bilingual_i18n.py       # Translation integrity tests
│
└── utils/                           # Agronomic & dataset utilities
    ├── disease_mapping.py           # Class name formatting
    └── evaluate_model.py            # Evaluation and matrix plotting utilities
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/plant-disease-detection.git
cd plant-disease-detection
```

### 3. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
```bash
# On Windows:
copy .env.example .env

# On Linux / macOS:
cp .env.example .env
```

---

## 🖥️ Running the Application

### Option A: Streamlit Interactive AI Dashboard (Recommended)
Launch the full interactive dashboard:
```bash
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### Option B: Flask Web Portal (With User Auth & MySQL)
Launch the Flask application:
```bash
python flask_app.py
```
Open your browser to: **`http://localhost:5000`**

---

## 🧪 Running Automated Tests

Run the test suite verifying CSV integrity, model predictions, and recommendation logic:
```bash
python tests/test_recommendations.py
```
Expected output:
```text
[OK] CSV successfully parsed. Shape: (5, 10)
[OK] All 5 classes aligned with class_names.json
[OK] Separation rule verified (Fertilizers disclaim curing disease)
[OK] Fallback logic verified ("Consult an agricultural expert")
[OK] End-to-end inference tested on all 5 leaf classes
ALL 5 INTEGRATION & RECOMMENDATION TESTS PASSED WITH 100% ACCURACY!
```

---

## 📦 Large Files & GitHub Upload Guidelines

When pushing this project to GitHub, observe the following file-size policies:

| File / Component | Size | GitHub Status | Recommendation & Handling |
| :--- | :--- | :--- | :--- |
| **`plant_disease_model.keras`** | **22.78 MB** | **Safe for Direct Push** | Under GitHub's 50 MB warning threshold and 100 MB hard limit. Included in the repository so users can run inference immediately upon cloning. |
| **`blackgram_recommendations.csv`** | **9.2 KB** | **Safe for Direct Push** | Lightweight structured CSV. Committed directly. |
| **`blackgram_repo.zip`** | **109.11 MB** | **Exceeds 100 MB Limit** | **Excluded via `.gitignore`**. Raw zip archives must not be committed to Git. |
| **`dataset/` (Raw Leaf Images)** | **~130 MB** | **Excluded via `.gitignore`** | Thousands of individual leaf images clutter Git history. Excluded via `.gitignore`. Download instructions provided below. |
| **`.venv/` (Virtual Environment)** | **~1.8 GB** | **Excluded via `.gitignore`** | Never commit virtual environments. Users recreate it via `requirements.txt`. |

### How to Acquire the Dataset:
The dataset is available from the original academic open-source repository:
- **Blackgram Leaf Disease Dataset**: [https://github.com/ai-agriculture-circuits-and-systems/blackgram_plant_leaf_disease_classification](https://github.com/ai-agriculture-circuits-and-systems/blackgram_plant_leaf_disease_classification)

To populate the local `dataset/` directory:
1. Download or clone the dataset repository into `dataset/blackgram/`.
2. The folder structure should follow:
   ```text
   dataset/blackgram/blackgrams/
   ├── anthracnose/images/*.jpg
   ├── healthy/images/*.jpg
   ├── leaf_crinckle/images/*.jpg
   ├── powdery_mildew/images/*.jpg
   └── yellow_mosaic/images/*.jpg
   ```

---

## 🏛️ Authoritative Agricultural Sources

All disease symptoms, chemical active ingredients, spray dosages, Pre-Harvest Intervals (PHI), and fertilizer protocols were compiled from:

1. **ICAR-IIPR** (Indian Institute of Pulses Research, Kanpur) — *Pulse Production Guidelines & Crop Protection Packages*.
2. **ICAR-NCIPM** (National Centre for Integrated Pest Management) — *Integrated Pest Management Package for Urdbean (Blackgram)*.
3. **TNAU Agritech Portal** — *Tamil Nadu Agricultural University Crop Protection: Pulses & Agronomy Guidelines*.
4. **CIBRC** (Central Insecticides Board & Registration Committee, Directorate of Plant Protection, Quarantine & Storage, Ministry of Agriculture & Farmers Welfare, Government of India) — *Major Uses of Pesticides (Registered Active Ingredients & Pre-Harvest Waiting Periods)*.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
