# 📧 NLP Spam Detection with GUI

An interactive **Natural Language Processing (NLP)** application designed to classify messages as **Spam** or **Ham**. The project features a user-friendly Graphical User Interface (GUI) built with **Tkinter** and compares the performance of five different Machine Learning algorithms.

---

### 🖥️ Application Screenshots

| Testing 'Ham' Message | Testing 'Spam' Message |
|:---:|:---:|
| ![Ham Detection](./Image/Ham.png) | ![Spam Detection](./Image/Spam.png) |

<p align="center">
  <b>Detailed Performance Metrics & Model Comparison</b><br>
  <img src="./Image/show_metrics.png" width="700" title="Model Comparison Metrics">
</p>

---

### 🚀 Key Features
* **Interactive GUI:** Test any message in real-time to see if it's classified as Spam or Ham.
* **Multi-Model Comparison:** Implements and evaluates five classifiers:
    * **Passive Aggressive (PA)**
    * **SGD Classifier**
    * **Naïve Bayes (NB)**
    * **Random Forest (RF)**
    * **Logistic Regression (LR)**
* **Advanced Evaluation:** Displays Test/Train accuracy, Confusion Matrix, and Detailed Classification Reports for each model.
* **Overfitting Detection:** Built-in logic to alert the user if a model shows signs of overfitting.

---

### 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn
* **GUI Framework:** Tkinter
* **Vectorization:** TfidfVectorizer (Max features: 2500)

---

### 📊 Methodology
1.  **Data Preprocessing:** Handled message cleaning, label encoding, and handled class imbalance using `class_weight`.
2.  **Feature Extraction:** Used **TF-IDF** to convert text into high-quality numerical features.
3.  **Model Training:** Trained 5 different classifiers to compare linear and non-linear performance.
4.  **Evaluation:** Models were evaluated using Accuracy, Precision, Recall, and F1-Score.

---

### 💻 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Banda-11/nlp-spam-detection-gui.git](https://github.com/Banda-11/nlp-spam-detection-gui.git)
    ```
2.  **Navigate to the folder:**
    ```bash
    cd nlp-spam-detection-gui
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python src/NLP_Project.py
    ```

---

### 📄 License
This project is licensed under the **MIT License**.
