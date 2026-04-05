import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def show_plot():
    print("\n[INFO] A graph window has opened.")
    print("Please close the graph window to continue...\n")
    plt.show()

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset and prepare target column."""
    df = df.copy()

    # Drop customerID because it is just an identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric, forcing bad values to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0 for new customers / no billing history yet
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target variable to numeric
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Print summary stats and show basic charts."""
    print("\n--- Dataset Preview ---")
    print(df.head())

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Churn Distribution ---")
    print(df["Churn"].value_counts(normalize=True))

    # Churn distribution chart
    df["Churn"].value_counts().plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = Stay, 1 = Leave)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Monthly charges by churn
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("Monthly Charges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn (0 = Stay, 1 = Leave)")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.show()

    # Tenure by churn
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure by Churn")
    plt.suptitle("")
    plt.xlabel("Churn (0 = Stay, 1 = Leave)")
    plt.ylabel("Tenure")
    plt.tight_layout()
    plt.show()

    # Contract type churn rate
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    print("\n--- Churn Rate by Contract Type ---")
    print(contract_churn)

    contract_churn.plot(kind="bar", stacked=True)
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()


def prepare_features(df: pd.DataFrame):
    """Prepare X and y for training."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Convert categorical columns into dummy variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    print("\n--- Encoded Feature Preview ---")
    print(X_encoded.head())

    print("\n--- Remaining Missing Values After Encoding ---")
    print(X_encoded.isnull().sum().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, X_encoded.columns


def train_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression and Random Forest, then compare results."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"\n--- {name} ---")
        print("Accuracy:", round(acc, 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("Classification Report:")
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    return best_model_name, best_model, best_accuracy


def show_feature_importance(model, feature_names):
    """Show top feature importances for Random Forest."""
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importance.sort_values(ascending=False).head(10)

        print("\n--- Top 10 Important Features ---")
        print(top_features)

        top_features.sort_values().plot(kind="barh")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        show_plot()
    else:
        print("\nFeature importance is not available for this model.")


def main():
    path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_data(path)
    df = clean_data(df)
    perform_eda(df)

    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)
    best_model_name, best_model, best_accuracy = train_models(X_train, X_test, y_train, y_test)

    print(f"\n--- Best Model ---")
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    show_feature_importance(best_model, feature_names)


if __name__ == "__main__":
    main()