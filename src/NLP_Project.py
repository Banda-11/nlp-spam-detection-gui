import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


class SpamDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Project - Spam Detection")
        self.root.geometry("780x700")
        self.root.resizable(False, False)

        self.models = {}
        self.metrics = {}

        self.tfidf = TfidfVectorizer(max_features=2500, stop_words='english')

        self.algo_names = ["PA", "SGD", "NB", "RF", "LR"]
        self.algo_full_names = {
            "PA":  "Passive Aggressive",
            "SGD": "SGD Classifier",
            "NB":  "Naïve Bayes",
            "RF":  "Random Forest",
            "LR":  "Logistic Regression",
        }

        self.result_fields = []

        self.setup_ui()
        self.prepare_data_and_models()

    # ─────────────────────────── UI ───────────────────────────
    def setup_ui(self):
        tk.Label(
            self.root, text="📧  Email Spam Detection",
            font=("Helvetica", 17, "bold")
        ).pack(pady=(14, 4))

        input_frame = tk.LabelFrame(self.root, text="Input Email / Message", padx=8, pady=6)
        input_frame.pack(fill="x", padx=20, pady=6)

        self.input_text = tk.Text(input_frame, height=6, width=88, font=("Courier", 10))
        self.input_text.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)

        tk.Button(
            btn_frame, text="⚡  Process Email",
            command=self.process_input,
            bg="#1565C0", fg="white", font=("Helvetica", 11, "bold"),
            width=18, relief="flat", padx=6, pady=4
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="📊  Show Metrics",
            command=self.show_metrics,
            bg="#2E7D32", fg="white", font=("Helvetica", 11, "bold"),
            width=18, relief="flat", padx=6, pady=4
        ).pack(side="left", padx=10)

        self.status_label = tk.Label(
            self.root, text="⏳  Loading models…", fg="orange", font=("Helvetica", 10)
        )
        self.status_label.pack(pady=2)

        results_frame = tk.LabelFrame(self.root, text="Algorithm Results", padx=10, pady=8)
        results_frame.pack(fill="x", padx=20, pady=6)

        header = tk.Frame(results_frame)
        header.pack(fill="x", pady=(0, 4))
        tk.Label(header, text="Algorithm",   width=20, anchor="w", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Label(header, text="Prediction",  width=14, anchor="center", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Label(header, text="Test Acc",    width=10, anchor="center", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Label(header, text="Train Acc",   width=10, anchor="center", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Label(header, text="Overfit?",    width=10, anchor="center", font=("Helvetica", 10, "bold")).pack(side="left")

        tk.Frame(results_frame, height=1, bg="#cccccc").pack(fill="x", pady=2)

        for name in self.algo_names:
            row = tk.Frame(results_frame)
            row.pack(fill="x", pady=3)

            tk.Label(
                row, text=f"{self.algo_full_names[name]} ({name})",
                width=20, anchor="w", font=("Helvetica", 10)
            ).pack(side="left")

            pred_var = tk.StringVar(value="—")
            pred_lbl = tk.Label(
                row, textvariable=pred_var,
                width=14, anchor="center",
                font=("Helvetica", 10, "bold"),
                relief="groove", bg="#f5f5f5"
            )
            pred_lbl.pack(side="left", padx=4)

            test_acc_var  = tk.StringVar(value="—")
            train_acc_var = tk.StringVar(value="—")
            overfit_var   = tk.StringVar(value="—")

            tk.Label(row, textvariable=test_acc_var,  width=10, anchor="center", font=("Courier", 10)).pack(side="left")
            tk.Label(row, textvariable=train_acc_var, width=10, anchor="center", font=("Courier", 10)).pack(side="left")
            tk.Label(row, textvariable=overfit_var,   width=10, anchor="center", font=("Courier", 10)).pack(side="left")

            self.result_fields.append({
                "pred_var":      pred_var,
                "pred_lbl":      pred_lbl,
                "test_acc_var":  test_acc_var,
                "train_acc_var": train_acc_var,
                "overfit_var":   overfit_var,
            })

    # ─────────────────────────── TRAIN ───────────────────────────
    def prepare_data_and_models(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'emails.csv')

        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Dataset not found!\n\nLooking for: {file_path}")
            self.status_label.config(text="❌ emails.csv not found", fg="red")
            return

        df = pd.read_csv(file_path)
        X = df.iloc[:, 0].astype(str)
        y = df.iloc[:, 1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_tfidf = self.tfidf.fit_transform(X_train)   
        X_test_tfidf  = self.tfidf.transform(X_test)        

        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

        models = {
            "PA":  PassiveAggressiveClassifier(max_iter=50),
            "SGD": SGDClassifier(loss="hinge", class_weight="balanced", max_iter=1000, random_state=42),
            "NB":  MultinomialNB(),                          
            "RF":  RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "LR":  LogisticRegression(max_iter=1000,        class_weight="balanced", random_state=42),
        }

        for i, key in enumerate(self.algo_names):
            model = models[key]
            model.fit(X_train_tfidf, y_train)

            y_pred_test  = model.predict(X_test_tfidf)
            y_pred_train = model.predict(X_train_tfidf)

            test_acc  = accuracy_score(y_test,  y_pred_test)
            train_acc = accuracy_score(y_train, y_pred_train)
            cm        = confusion_matrix(y_test, y_pred_test)
            report    = classification_report(y_test, y_pred_test, digits=4)

            overfitting = (train_acc - test_acc) > 0.10

            self.models[key] = model
            self.metrics[key] = {
                "test_acc":  test_acc,
                "train_acc": train_acc,
                "cm":        cm,
                "report":    report,
                "overfit":   overfitting,
            }

            f = self.result_fields[i]
            f["test_acc_var"].set(f"{test_acc:.2%}")
            f["train_acc_var"].set(f"{train_acc:.2%}")
            f["overfit_var"].set("⚠️ Yes" if overfitting else "✅ No")

        self.status_label.config(text="✅  All models trained and ready", fg="green")

    # ─────────────────────────── PREDICT ───────────────────────────
    def process_input(self):
        text = self.input_text.get("1.0", "end-1c").strip()

        if not text:
            messagebox.showwarning("Warning", "Please enter an email / message first.")
            return

        if not self.models:
            messagebox.showerror("Error", "Models not loaded yet.")
            return

        vec = self.tfidf.transform([text])

        for i, key in enumerate(self.algo_names):
            pred   = self.models[key].predict(vec)[0]
            result = "🚫  SPAM" if pred == 1 else "✅  HAM"
            color  = "#C62828" if pred == 1 else "#1B5E20"

            f = self.result_fields[i]
            f["pred_var"].set(result)
            f["pred_lbl"].config(fg=color)

    # ─────────────────────────── METRICS WINDOW ───────────────────────────
    def show_metrics(self):
        if not self.metrics:
            messagebox.showinfo("Info", "Models are not trained yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Full Evaluation Report")
        win.geometry("860x640")

        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Courier", 10))
        text_area.pack(fill="both", expand=True, padx=8, pady=8)

        for key in self.algo_names:
            m = self.metrics[key]
            sep = "=" * 55

            text_area.insert(tk.END, f"\n{sep}\n")
            text_area.insert(tk.END, f"  Algorithm : {self.algo_full_names[key]} ({key})\n")
            text_area.insert(tk.END, f"{sep}\n")
            text_area.insert(tk.END, f"  Test  Accuracy : {m['test_acc']:.4f}  ({m['test_acc']:.2%})\n")
            text_area.insert(tk.END, f"  Train Accuracy : {m['train_acc']:.4f}  ({m['train_acc']:.2%})\n")

            if m["overfit"]:
                text_area.insert(tk.END, "  ⚠️  Possible Overfitting Detected  "
                                         "(train_acc - test_acc > 10%)\n")

            text_area.insert(tk.END, "\n  Confusion Matrix:\n")
            text_area.insert(tk.END, f"  {m['cm']}\n")

            text_area.insert(tk.END, "\n  Classification Report "
                                     "(precision / recall / F1-score):\n")
            # indent each line of the report
            for line in m["report"].splitlines():
                text_area.insert(tk.END, f"  {line}\n")

            text_area.insert(tk.END, "\n")

        text_area.config(state="disabled")


# ─────────────────────────── ENTRY POINT ───────────────────────────
if __name__ == "__main__":# start from here
    root = tk.Tk()
    app = SpamDetectionApp(root)
    root.mainloop()
