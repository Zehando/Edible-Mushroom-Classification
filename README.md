# Kaggle Mushroom Classification 🍄

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-336791?style=for-the-badge&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9C4668?style=for-the-badge&logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-E6472A?style=for-the-badge&logo=catboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

This project is a solution for a classification competition to determine if a mushroom is poisonous or edible.

The primary goal is not just accuracy, but **safety**. The model is heavily weighted to avoid the critical error of classifying a poisonous mushroom as edible.

## The Safety-Critical Objective

In this problem, the two types of errors have vastly different costs:

* **False Negative (Most Dangerous):** Predicting a **poisonous** mushroom as **edible**. This is a critical, high-cost error that must be avoided at all costs.
* **False Positive:** Predicting an **edible** mushroom as **poisonous**. This is a minor inconvenience (you throw away a good mushroom).

Therefore, the entire modeling process is optimized to **minimize False Negatives** and achieve the highest possible **Recall** for the 'poisonous' class.

## Workflow

This notebook follows a comprehensive workflow to build the most robust and safe model possible.

1.  **Preprocessing:** All features were treated as categorical. The notebook first compares the performance of `OneHotEncoder` vs. `TargetEncoder` using a baseline `RandomForestClassifier`.

2.  **Advanced Modeling (Stacking):** A `StackingClassifier` was built to combine the predictions of several powerful models:
    * **Base Models:**
        * Random Forest (with OneHotEncoder)
        * LightGBM (with OneHotEncoder)
        * XGBoost (with OneHotEncoder)
        * CatBoost (with native categorical feature support)
    * **Meta-Model:** `LogisticRegression`

3.  **Cost-Sensitive Threshold Tuning:** This is the most important step. A custom cost function was defined to heavily penalize the dangerous error:
    * **False Negative Cost (Poisonous as Edible): 100**
    * **False Positive Cost (Edible as Poisonous): 3**

    This custom score was used with `TunedThresholdClassifierCV` to find the optimal probability threshold (e.g., 0.06) that minimizes this weighted cost, forcing the model to be extremely cautious.

## 📊 Results

* The `StackingClassifier` outperformed the baseline `RandomForest` in raw accuracy.
* After applying the cost-sensitive threshold tuning, the final stacking model achieved **99.8% - 100% Recall** on the test set.
* This result means the model successfully identified (almost) all poisonous mushrooms, achieving the primary safety objective.

## 💡 How to Run

1.  Ensure you have all the libraries from the `imports` cell installed.
2.  Place the `mush_train.csv` and `mush_test.csv` files in the correct directory.
3.  Run the `mushroom competition.ipynb` notebook.
4.  The final predictions will be saved as `submission-sahand.csv` using the tuned stacking model.

---

## 🤝 Contact

For any questions or feedback, feel free to reach out:

* **GitHub:** [@zehando](https://github.com/zehando)
* **LinkedIn:** [Sahand Azizi](https://www.linkedin.com/in/sahandazizi/)
* **Email:** azizisahand@gmail.com
