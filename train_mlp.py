import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import datetime
import yaml


# ✅ Flexible Dataset from preprocessed .pt file
class FlexibleFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, features):
        self.y = y
        self.X = self.select_features(X, features)

    def select_features(self, X, features):
        selected = []
        # ✅ Assumes shape: [N, D] where D=48 (not [N, T, D])
        if "pose" in features:
            selected.append(X[:, :34])
        if "gaze" in features:
            selected.append(X[:, 34:36])
        if "fm" in features:
            selected.append(X[:, 36:48])
        return torch.cat(selected, dim=-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




# ✅ LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


# ✅ Deep MLP
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)



# ✅ Confusion matrix 저장
def print_and_save_confusion_matrix(y_true, y_pred, file_path):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    with open(file_path, "w") as f:
        f.write("Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report)


# ✅ feature 파싱
def parse_feature_selection(raw_string):
    allowed = {"pose", "gaze", "fm"}
    tokens = raw_string.lower().replace(" ", "").split("+")
    selected = set()
    for t in tokens:
        if t in allowed:
            selected.add(t)
        else:
            print(f"[WARNING] Unknown feature '{t}' ignored.")
    return selected


# ✅ args 저장
def save_args_to_yaml(args, path):
    args_dict = vars(args)
    with open(path, 'w') as f:
        yaml.dump(args_dict, f)


# ✅ Train 함수
def train(args):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_str = "+".join(sorted(args.features))
    model_str = args.model.lower()
    result_dir = os.path.join("results", f"{now}_{feature_str}_{model_str}")
    os.makedirs(result_dir, exist_ok=True)
    config_path = os.path.join(result_dir, "config.yaml")
    save_args_to_yaml(args, config_path)
    print(f"[INFO] Saving all results to: {result_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Selected features: {args.features}")
    print(f"[INFO] Selected model: {args.model}")

    # Load dataset
    X, y = torch.load("features/pose_gaze_fm_1f.pt")
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

    train_dataset = FlexibleFeatureDataset(train_X, train_y, args.features)
    val_dataset = FlexibleFeatureDataset(val_X, val_y, args.features)
    test_dataset = FlexibleFeatureDataset(test_X, test_y, args.features)

    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 1-frame: input is [D], not [T, D]
    input_shape = train_dataset[0][0].shape  # shape = [D]
    input_dim = input_shape[0]

    num_classes = len(torch.unique(train_y))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 모델 선택
    if args.model == "lstm":
        model = LSTMClassifier(input_dim, args.hidden_dim, num_classes,
                       num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.model == "mlp":
        model = DeepMLP(input_dim, num_classes).to(device)
    else:
        raise ValueError("Model must be 'lstm' or 'mlp'")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    patience_counter = 0
    print("[INFO] Training started...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            if args.model == "mlp":
                x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                if args.model == "mlp":
                    x = x.view(x.size(0), -1)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Balanced Acc={train_bal_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Balanced Val Acc={val_bal_acc:.4f}\n")

        if val_bal_acc > best_acc:
            best_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"[INFO] 🔥 Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1

        scheduler.step(val_loss)
        if patience_counter >= args.patience:
            print("[INFO] Early stopping triggered.")
            break

    # 🧪 Test
    print("[INFO] Evaluating best model...")
    model.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pt")))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="[Test]"):
            x, y = x.to(device), y.to(device)
            if args.model == "mlp":
                x = x.view(x.size(0), -1)
            outputs = model(x)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"[RESULT] Test Accuracy: {acc:.4f}")
    print(f"[RESULT] Test Balanced Accuracy: {bal_acc:.4f}")
    cm_path = os.path.join(result_dir, "confusion_matrix.txt")
    print_and_save_confusion_matrix(all_labels, all_preds, cm_path)


# ✅ Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="pose+gaze+fm",
                        help="Choose from pose, gaze, fm. e.g., pose+fm")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "mlp"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout rate for LSTM model (default: 0.5)")


    args = parser.parse_args()
    args.features = parse_feature_selection(args.features)
    train(args)
