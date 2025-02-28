import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Gradient Reversal Layer (GRL)
# ---------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

# ---------------------------
# Top-level Scaling Transform (picklable)
# ---------------------------
class ScalingTransform:
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, features):
        # features: numpy array of shape (num_tiles, 2048)
        scaled = self.scaler.transform(features)
        return torch.tensor(scaled, dtype=torch.float)

# ---------------------------
# Dataset for Training/Validation
# ---------------------------
class HistopathologyDataset(Dataset):
    def __init__(self, features_dir, output_csv, metadata_csv, transform=None):
        """
        features_dir : directory containing .npy files (one per sample)
        output_csv   : CSV with columns "Sample ID" and "Target"
        metadata_csv : CSV with at least "Sample ID" and "Center ID"
        transform    : callable to process a numpy array (shape: [1000, 2048]) into a torch.Tensor
        """
        self.features_dir = Path(features_dir)
        self.transform = transform

        self.outputs_df = pd.read_csv(output_csv)
        self.samples = self.outputs_df["Sample ID"].tolist()
        self.targets = self.outputs_df["Target"].tolist()

        self.metadata_df = pd.read_csv(metadata_csv)
        self.center_map = dict(zip(self.metadata_df["Sample ID"], self.metadata_df["Center ID"]))
        centers = sorted(self.metadata_df["Center ID"].unique())
        self.center_to_idx = {center: idx for idx, center in enumerate(centers)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        npy_path = self.features_dir / sample_id
        # Load file: expected shape (1000, 2051)
        features = np.load(npy_path)
        features = features[:, 3:]  # keep only the MoCo features (2048-d)
        if self.transform:
            features = self.transform(features)
        else:
            features = torch.tensor(features, dtype=torch.float)

        target = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(0)
        center = self.center_map[sample_id]
        domain = self.center_to_idx[center]
        domain = torch.tensor(domain, dtype=torch.long)
        return features, target, domain

# ---------------------------
# Dataset for Test Samples (no labels)
# ---------------------------
class HistopathologyTestDataset(Dataset):
    def __init__(self, features_dir, transform=None):
        self.features_dir = Path(features_dir)
        self.transform = transform
        self.samples = sorted([f.name for f in self.features_dir.glob("*.npy")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        npy_path = self.features_dir / sample_id
        features = np.load(npy_path)  # shape: (1000, 2051)
        features = features[:, 3:]    # only MoCo features
        if self.transform:
            features = self.transform(features)
        else:
            features = torch.tensor(features, dtype=torch.float)
        return features, sample_id

# ---------------------------
# MIL with Multi-Level Domain Adaptation Model
# ---------------------------
class MIL_DANN_Multi(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, num_domains=3, dropout_prob=0.5, use_gated=True):
        """
        input_dim   : Dimension of tile features (2048)
        hidden_dim  : Projection dimension (1024)
        num_domains : Number of distinct centers/domains
        dropout_prob: Dropout probability (0.5)
        use_gated   : Use gated attention mechanism
        """
        super(MIL_DANN_Multi, self).__init__()
        # Tile feature projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        self.use_gated = use_gated
        if use_gated:
            # Gated attention: intermediate dimension 256
            self.attention_V = nn.Linear(hidden_dim, 256)
            self.attention_U = nn.Linear(hidden_dim, 256)
            self.attention_weights = nn.Linear(256, 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        # Main classification branch (bag-level)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 1)
        )

        # Domain classifier branch for bag-level features
        self.domain_classifier_bag = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_domains)
        )
        # Additional domain classifier branch for tile-level aggregated features
        self.domain_classifier_tile = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_domains)
        )

    def forward(self, x, lambda_adapt=1.0):
        # x: (batch, num_tiles, input_dim)
        h = self.fc1(x)            # (B, T, hidden_dim)
        h = self.relu(h)
        h = self.dropout(h)

        # Attention mechanism (gated if enabled)
        if self.use_gated:
            a = torch.tanh(self.attention_V(h))   # (B, T, 256)
            b = torch.sigmoid(self.attention_U(h))  # (B, T, 256)
            gated = a * b
            att_scores = self.attention_weights(gated)  # (B, T, 1)
        else:
            att_scores = self.attention(h)  # (B, T, 1)

        att_weights = torch.softmax(att_scores, dim=1)  # (B, T, 1)
        # Aggregate tile features to get bag representation
        bag_rep = torch.sum(att_weights * h, dim=1)  # (B, hidden_dim)

        # Main classification output
        logits = self.classifier(bag_rep)  # (B, 1)

        # Domain adaptation: apply gradient reversal
        # 1. Domain classifier at bag level
        reversed_bag = grad_reverse(bag_rep, lambda_adapt)
        domain_logits_bag = self.domain_classifier_bag(reversed_bag)

        # 2. Domain classifier at tile level: use simple average of tile features
        tile_avg = torch.mean(h, dim=1)  # (B, hidden_dim)
        reversed_tile = grad_reverse(tile_avg, lambda_adapt)
        domain_logits_tile = self.domain_classifier_tile(reversed_tile)

        return logits, domain_logits_bag, domain_logits_tile

# ---------------------------
# Training & Evaluation Functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_domain, device, lambda_adapt, domain_loss_weight):
    model.train()
    running_loss = 0.0
    for features, target, domain in dataloader:
        features = features.to(device)  # (batch, num_tiles, 2048)
        target = target.to(device)
        domain = domain.to(device)

        optimizer.zero_grad()
        logits, dom_logits_bag, dom_logits_tile = model(features, lambda_adapt=lambda_adapt)
        loss_cls = criterion_cls(logits, target)
        loss_dom_bag = criterion_domain(dom_logits_bag, domain)
        loss_dom_tile = criterion_domain(dom_logits_tile, domain)
        loss_domain = (loss_dom_bag + loss_dom_tile) / 2.0
        loss = loss_cls + domain_loss_weight * loss_domain
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion_cls, criterion_domain, device, lambda_adapt, domain_loss_weight):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, target, domain in dataloader:
            features = features.to(device)
            target = target.to(device)
            domain = domain.to(device)
            logits, dom_logits_bag, dom_logits_tile = model(features, lambda_adapt=lambda_adapt)
            loss_cls = criterion_cls(logits, target)
            loss_dom_bag = criterion_domain(dom_logits_bag, domain)
            loss_dom_tile = criterion_domain(dom_logits_tile, domain)
            loss_domain = (loss_dom_bag + loss_dom_tile) / 2.0
            loss = loss_cls + domain_loss_weight * loss_domain
            running_loss += loss.item() * features.size(0)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
    auc = roc_auc_score(all_targets, all_preds)
    return running_loss / len(dataloader.dataset), auc

# ---------------------------
# Preprocessing: Fit StandardScaler on all training tile features
# ---------------------------
def compute_scaler(features_dir, output_csv):
    features_dir = Path(features_dir)
    df = pd.read_csv(output_csv)
    samples = df["Sample ID"].tolist()
    all_features = []
    for sample in samples:
        feat = np.load(features_dir / sample)  # shape: (1000, 2051)
        feat = feat[:, 3:]  # keep only the 2048-d MoCo features
        all_features.append(feat)
    all_features = np.concatenate(all_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler

# ---------------------------
# Main Training & Testing Pipeline with Early Stopping & Scheduler
# ---------------------------
def main():
    # Adjust these paths as needed
    data_dir = Path("C:/Users/ulyss/modele_ML/ENS_Challenge")
    train_features_dir = data_dir / "train_input" / "moco_features"
    test_features_dir  = data_dir / "test_input" / "moco_features"
    train_metadata_csv = data_dir / "supplementary_data" / "train_metadata.csv"
    train_output_csv   = data_dir / "train_output.csv"

    print("Fitting scaler on training data...")
    scaler = compute_scaler(train_features_dir, train_output_csv)
    transform = ScalingTransform(scaler)

    full_dataset = HistopathologyDataset(train_features_dir, train_output_csv, train_metadata_csv, transform=transform)
    print(f"Total training samples: {len(full_dataset)}")

    # 80/20 split for training and validation
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    metadata_df = pd.read_csv(train_metadata_csv)
    num_domains = len(metadata_df["Center ID"].unique())

    # Instantiate the multi-level domain adaptation model
    model = MIL_DANN_Multi(input_dim=2048, hidden_dim=1024, num_domains=num_domains,
                           dropout_prob=0.5, use_gated=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use AdamW optimizer with a low learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_domain = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    num_epochs = 50
    lambda_adapt = 0.2    # GRL coefficient (tunable)
    domain_loss_weight = 1.0
    best_val_auc = 0.0
    best_model_path = "best_mil_dann_multi_model.pth"
    epochs_no_improve = 0
    early_stop_patience = 5

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_cls, criterion_domain,
                                 device, lambda_adapt, domain_loss_weight)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion_cls, criterion_domain,
                                       device, lambda_adapt, domain_loss_weight)
        scheduler.step(val_auc)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print("==> Saved best model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Testing: load best model and run on test set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_dataset = HistopathologyTestDataset(test_features_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_sample_ids = []
    all_preds = []
    with torch.no_grad():
        for features, sample_id in test_loader:
            features = features.to(device)
            logits, _, _ = model(features, lambda_adapt=0.0)
            prob = torch.sigmoid(logits).item()
            all_preds.append(prob)
            all_sample_ids.append(sample_id[0])

    submission = pd.DataFrame({"Sample ID": all_sample_ids, "Target": all_preds})
    submission = submission.sort_values("Sample ID")
    assert submission["Target"].between(0, 1).all(), "Target values must be in [0, 1]"
    assert submission.shape[1] == 2, "Submission file must have 2 columns"
    submission_file = data_dir / "benchmark_test_output_ulysse.csv"
    submission.to_csv(submission_file, index=False)
    print("Submission saved to", submission_file)

if __name__ == "__main__":
    main()
