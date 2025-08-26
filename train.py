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
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
import time
from torchvision.utils import save_image
from thop import profile


# ✅ Flexible Dataset from preprocessed .pt file
class FlexibleFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, features):
        self.y = y
        self.X = self.select_features(X, features)

    def select_features(self, X, features):
        selected = []
        if "pose" in features:
            selected.append(X[:, :, :34])
        if "gaze" in features:
            selected.append(X[:, :, 34:36])
        if "fm" in features:
            selected.append(X[:, :, 36:48])
        return torch.cat(selected, dim=-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.0):  # ✅ 추가
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = self.norm(output.mean(dim=1))
        return self.fc(x)




# ✅ Deep MLP
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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

def plot_and_save_top20_confusion_matrix(y_true, y_pred, output_dir):
    top20_classes = [cls for cls, _ in Counter(y_true).most_common(20)]
    mask = [yt in top20_classes for yt in y_true]
    y_true_top = [yt for yt, keep in zip(y_true, mask) if keep]
    y_pred_top = [yp for yp, keep in zip(y_pred, mask) if keep]
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top20_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=top20_classes, yticklabels=top20_classes, cmap="Blues")
    plt.title("Confusion Matrix (Top-20 Frequent Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_top20.png"))
    plt.close()

def plot_and_save_bottom20_confusion_matrix(y_true, y_pred, output_dir):
    class_counts = Counter(y_true)
    bottom20_classes = [cls for cls, _ in sorted(class_counts.items(), key=lambda x: x[1])[:20]]
    mask = [yt in bottom20_classes for yt in y_true]
    y_true_bottom = [yt for yt, keep in zip(y_true, mask) if keep]
    y_pred_bottom = [yp for yp, keep in zip(y_pred, mask) if keep]
    cm = confusion_matrix(y_true_bottom, y_pred_bottom, labels=bottom20_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=bottom20_classes, yticklabels=bottom20_classes, cmap="Oranges")
    plt.title("Confusion Matrix (Bottom-20 Least Frequent Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_bottom20.png"))
    plt.close()


def save_error_case_images(y_true, y_pred, test_dataset, output_dir, max_cases=5):
    os.makedirs(os.path.join(output_dir, "error_images"), exist_ok=True)
    error_indices = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    
    for i, idx in enumerate(error_indices[:max_cases]):
        x, y = test_dataset[idx]  # x: [seq_len, feat_dim]
        x_img = x.permute(1, 0).unsqueeze(0)  # [1, feat_dim, seq_len] 형태로 변형 (이미지처럼 저장)
        save_path = os.path.join(output_dir, "error_images", f"error_{i+1}_true{y_true[idx]}_pred{y_pred[idx]}.png")
        save_image(x_img, save_path)

def save_top_bottom_f1_scores(y_true, y_pred, output_dir):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # class_id: (f1-score, support)
    f1_support_dict = {
        int(k): {"f1": v["f1-score"], "support": v["support"]}
        for k, v in report.items()
        if k.isdigit()
    }

    # 정렬 (F1 기준 내림차순)
    sorted_items = sorted(f1_support_dict.items(), key=lambda x: x[1]["f1"], reverse=True)

    result = {
        "top5": [{"class_id": k, "f1": round(v["f1"], 4), "support": v["support"]} for k, v in sorted_items[:5]],
        "bottom5": [{"class_id": k, "f1": round(v["f1"], 4), "support": v["support"]} for k, v in sorted_items[-5:]]
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "f1_top5_bottom5.json"), "w") as f:
        json.dump(result, f, indent=2)

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
    start_time = time.time()
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
    # X, y = torch.load("features/pose_gaze_fm_30f10s.pt")
    X, y = torch.load(args.data_path)
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

    train_dataset = FlexibleFeatureDataset(train_X, train_y, args.features)
    val_dataset = FlexibleFeatureDataset(val_X, val_y, args.features)
    test_dataset = FlexibleFeatureDataset(test_X, test_y, args.features)

    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    input_shape = train_dataset[0][0].shape
    input_dim = input_shape[1] if args.model == "lstm" else input_shape[0] * input_shape[1]
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")
    
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
        
    plot_and_save_top20_confusion_matrix(all_labels, all_preds, result_dir)
    plot_and_save_bottom20_confusion_matrix(all_labels, all_preds, result_dir)
    save_top_bottom_f1_scores(all_labels, all_preds, result_dir)
    save_error_case_images(all_labels, all_preds, test_dataset, result_dir, max_cases=5)


    end_time = time.time()
    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

    ##########################################
    # ✅ Computational cost 측정
    print("[INFO] Measuring computational cost...")

    # 하나의 sample 선택
    x_sample, _ = test_dataset[0]
    x_sample = x_sample.unsqueeze(0).to(device)  # shape: (1, seq_len, feat_dim)

    if args.model == "mlp":
        x_sample = x_sample.view(1, -1)

    # FLOPs 및 파라미터 수 측정
    flops, params = profile(model, inputs=(x_sample,), verbose=False)

    # Inference time 측정
    model.eval()
    with torch.no_grad():
        timings = []
        for _ in range(100):
            x = x_sample.clone()
            torch.cuda.synchronize()
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.time()
            timings.append((end - start) * 1000)  # ms

    avg_time = np.mean(timings)
    std_time = np.std(timings)

    # 저장
    stats_path = os.path.join(result_dir, "computational_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"FLOPs: {flops / 1e6:.2f} MFLOPs\n")
        f.write(f"Trainable Parameters: {params / 1e6:.2f} M\n")
        f.write(f"Inference Time per Sample: {avg_time:.2f} ± {std_time:.2f} ms (100 runs)\n")

    print("[INFO] Saved computational stats.")
    ##########################################

# ✅ Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input .pt file (e.g., features/pose_gaze_fm_30f10s.pt)")
    parser.add_argument("--features", type=str, default="pose+gaze+fm",
                        help="Choose from pose, gaze, fm. e.g., pose+fm")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "mlp"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for LSTM model (default: 0.0)")

    args = parser.parse_args()
    args.features = parse_feature_selection(args.features)
    train(args)

