import joblib
import os
from app.scraper import load_data
from app.features import create_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "gender",
    "age",
    "discount_ratio",
    "avg_quantity_per_product",
    "inactivity_ratio",
    "spend_per_product",
    "purchases_per_day",
    "large_cart",
    "high_spender",
]


def train_and_export():

    print("Loading data...")
    df = load_data()

    print("Engineering features...")
    df = create_features(df)

    # Check class balance
    print("\nChurn distribution:")
    print(df["churned"].value_counts())
    print(f"Churn rate: {df['churned'].mean():.1%}\n")

    X = df[FEATURE_COLUMNS]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",  # handles class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")

    # Save model.pkl inside app/ so Docker can find it
    os.makedirs("app", exist_ok=True)
    joblib.dump(model, "app/model.pkl")
    print("Model saved to app/model.pkl")


if __name__ == "__main__":
    train_and_export()