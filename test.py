import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from model import SiameseFusionNet_test
from dataloader import FlexiblePairedCXRDataset_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm

model = SiameseFusionNet_test()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda")
model = model.to(device)

def metrics_with_ci(trues, preds, n_bootstrap=1000, ci=95, random_seed=42):
    trues = np.array(trues)
    preds = np.array(preds)
    n = len(trues)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
  
    rmse_samples = []
    mae_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)

        sample_trues = trues[indices]
        sample_preds = preds[indices]

        sample_rmse = np.sqrt(mean_squared_error(sample_trues, sample_preds))
        sample_mae = mean_absolute_error(sample_trues, sample_preds)

        rmse_samples.append(sample_rmse)
        mae_samples.append(sample_mae)

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct

    rmse_ci = (np.percentile(rmse_samples, lower_pct), np.percentile(rmse_samples, upper_pct))
    mae_ci = (np.percentile(mae_samples, lower_pct), np.percentile(mae_samples, upper_pct))

    return{
        "rmse": rmse, 
        "rmse_ci": rmse_ci,
        "mae": mae, 
        "mae_ci": mae_ci
    }

def predict(model, dataloader):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            img1, img2, meta1, meta2, y1, y2, sample_type = batch
            img1 = img1.to(device)
            img2 = img2.to(device)
            meta1 = meta1.to(device)
            meta2 = meta2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            y1_hat, y2_hat = model(img1, img2, meta1, meta2, y1, y2, sample_type)
            y1_hat = y1_hat.to(device)
            y2_hat = y2_hat.to(device)

            y_true.extend(torch.cat([y1, y2]).cpu().numpy())
            y_pred.extend(torch.cat([y1_hat, y2_hat]).cpu().numpy())

            # y1_hat = y1_hat.cpu().numpy()
            # y2_hat = y2_hat.cpu().numpy()

            # plt.scatter(y1_hat, y2_hat, c='dodgerblue')
            # plt.xlabel('predicted fev1 by img1')
            # plt.ylabel('predicted fev1 by img2')
            # plt.title('predicted fev1 values of img1 vs img2')
            # plt.savefig('scatter_plot.png')

    results = metrics_with_ci(y_true, y_pred)
    print(f"RMSE: {results['rmse']:.3f}, 95% CI: ({results['rmse_ci'][0]:.3f}, {results['rmse_ci'][1]:.3f})")
    print(f"MAE: {results['mae']:.3f}, 95% CI: ({results['mae_ci'][0]:.3f}, {results['mae_ci'][1]:.3f})")
    print()

    return y_pred, y_true

# Load all pairs from CSV or list
meta_csv_path = './train.csv'
data = pd.read_csv(meta_csv_path)
pairs = list(data.itertuples(index=False, name=None))

img_root_896 = './CXR'
img_root_mask = './mask'

meta = pd.read_csv(meta_csv_path).dropna(subset=['fev1', 'age', 'sex', 'height'])
test_ratio = 0.2

_, test_pids = train_test_split(meta['patient_id'].unique(), test_size=test_ratio, random_state=42)
test_meta = meta[meta['patient_id'].isin(test_pids)]

# ==================== dataloader ====================
dataset_test = FlexiblePairedCXRDataset_test(img_root_896, img_root_mask, test_meta)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=True)

batch_size = 16

# ==================== prediction =====================
model_path = './ckpt'

ckpts = [f for f in os.listdir(model_path) if f.endswith(".pth")]
results_list = []

for ckpt in ckpts:
    print(ckpt)
    ckpt_path = os.path.join(model_path, ckpt)

    model.load_state_dict(torch.load(ckpt_path), strict=False)
    predictions, true_labels = predict(model, test_loader)

    metrics = metrics_with_ci(true_labels, predictions)

    results_list.append({
        'checkpoint': ckpt,
        'rmse': metrics['rmse'],
        'rmse_ci_lower': metrics['rmse_ci'][0],
        'rmse_ci_upper': metrics['rmse_ci'][1],
        'mae': metrics['mae'],
        'mae_ci_lower': metrics['mae_ci'][0],
        'mae_ci_upper': metrics['mae_ci'][1],
    })

results_df = pd.DataFrame(results_list)

save_csv_path = os.path.join(model_path, 'test_results.csv')
results_df.to_csv(save_csv_path, index=False)

print('---------------DONE---------------')

