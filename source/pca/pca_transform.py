import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Apply PCA to extracted DINOv2 features.")
    parser.add_argument("--input_npz", type=str, required=True,
                        help="Path to .npz file with 'ids' and 'feats' arrays.")
    parser.add_argument("--n_components", type=int, default=30,
                        help="Number of PCA components (default: 30).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file path.")
    args = parser.parse_args()

    # Load features
    data = np.load(args.input_npz)
    ids = data["ids"].reshape(-1)
    feats = data["feats"]
    print(f"Loaded features: {feats.shape[0]} samples, {feats.shape[1]} dimensions")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feats)

    # PCA
    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Print explained variance
    cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"Cumulative explained variance ({args.n_components} PCs): {cumulative_var[-1]:.2f}%")
    for i in [0, 4, 9, 19, 29]:
        if i < args.n_components:
            print(f"  PC{i+1}: {cumulative_var[i]:.2f}%")

    # Build DataFrame
    pc_columns = [f"PC{i+1}" for i in range(args.n_components)]
    df = pd.DataFrame(X_pca, columns=pc_columns)
    df.insert(0, "eid", ids.astype(int))

    # Save
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved PCA results to {args.output_csv} | shape: {df.shape}")


if __name__ == "__main__":
    main()
