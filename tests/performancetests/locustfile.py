import random

import pandas as pd
from locust import HttpUser, between, task


class NewsUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 2)

    def on_start(self):
        fake = pd.read_csv(f"data/raw/Fake.csv")
        real = pd.read_csv(f"data/raw/True.csv")

        fake["label"] = 0
        real["label"] = 1

        df = pd.concat([fake, real], ignore_index=True)

        self.data = df[["title", "text", "label"]].dropna().reset_index(drop=True)

    @task
    def get_pred(self) -> None:
        idx = random.randint(0, self.data.shape[0] - 1)

        payload = self.data.iloc[idx][["title", "text"]].to_dict()

        self.client.post("/news/", json=payload)
