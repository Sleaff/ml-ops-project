import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

# from project.dataset import NewsDataset
# from project.model import Model
from dataset import NewsDataset
from model import Model
from dotenv import load_dotenv
load_dotenv()
import os
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import typer

app = typer.Typer()

@app.command()
def train(lr: float = 2e-5, epochs : int = 10, batch_size : int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NewsDataset("data/processed/news.csv", tokenizer)

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    
    print(f"Logging to: {wandb_entity}/{wandb_project}")

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "architecture": "DistilBert",
        }
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    print(f'Lenght train dataset: {len(train_ds)}')
    print(f'Lenght val dataset: {len(val_ds)}')

    model = Model().to(device)
    wandb.watch(model, log_freq=100)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for e in range(int(epochs)):

        run_train_loss = 0
        run_train_acc = 0

        run_val_loss = 0
        run_val_acc = 0

        model.train()

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            # Loss
            loss = criterion(logits, labels)
            run_train_loss += loss

            loss.backward()
            optimizer.step()

            # Accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == labels.view_as(preds)).float().mean().item()
    
            run_train_acc += accuracy
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)

                # Loss
                loss = criterion(logits, labels)
                run_val_loss += loss

                # Accuracy
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                accuracy = (preds == labels.view_as(preds)).float().mean().item()

                run_val_acc += accuracy

        avg_train_loss = run_train_loss/len(train_loader)
        avg_train_acc = run_train_acc/len(train_loader)

        avg_val_loss = run_val_loss/len(val_loader)
        avg_val_acc = run_val_acc/len(val_loader)

        print(f'Average train loss at epoch {int(e)+1}: {avg_train_loss}')
        print(f'Average val loss at epoch {int(e)+1}: {avg_val_loss}')

        wandb.log({
            "train/loss": avg_train_loss,
            "val/loss" : avg_val_loss,
            "train/acc": avg_train_acc,
            "val/acc": avg_val_acc,
            "epoch": e
        })
    
    # Save model
    torch.save(model.state_dict(), "models/model.pt")
    artifact = wandb.Artifact("news_model", type="model")
    artifact.add_file("models/model.pt")
    run.log_artifact(artifact)
    
    run.finish()

if __name__ == "__main__":
    train()
