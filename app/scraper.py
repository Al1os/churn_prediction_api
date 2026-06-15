import requests
import pandas as pd
import random
import os


def load_data() -> pd.DataFrame:

    users = requests.get(
        "https://dummyjson.com/users?limit=0"
    ).json()["users"]

    carts = requests.get(
        "https://dummyjson.com/carts?limit=0"
    ).json()["carts"]

    users_df = pd.DataFrame(users)

    random.seed(42)
    records = []

    for cart in carts:

        user = users_df[users_df["id"] == cart["userId"]]

        if user.empty:
            continue

        user = user.iloc[0]

        # Simulate 5 purchase sessions per cart
        for _ in range(5):

            total_products = max(
                1, cart["totalProducts"] + random.randint(-1, 1)
            )
            total_quantity = max(
                1, cart["totalQuantity"] + random.randint(-2, 2)
            )
            total_amount = round(
                cart["total"] * random.uniform(0.9, 1.1), 2
            )
            discounted_total = round(
                cart["discountedTotal"] * random.uniform(0.9, 1.1), 2
            )
            days_inactive = random.randint(1, 365)
            days_since_registration = random.randint(30, 2000)

            records.append({
                "user_id":                  cart["userId"],
                "gender":                   1 if user["gender"] == "male" else 0,
                "age":                      user["age"],
                "total_products":           total_products,
                "total_quantity":           total_quantity,
                "total_amount":             total_amount,
                "discounted_total":         discounted_total,
                "days_inactive":            days_inactive,
                "days_since_registration":  days_since_registration,
            })

    df = pd.DataFrame(records)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/customer_data.csv", index=False)

    print(f"Fetched {len(df)} records from DummyJSON")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Shape:", df.shape)