import os
import argparse
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm


def most_common_label(tensor_1d):
    """
    Return the most frequent label in a 1D tensor.
    """
    return Counter(tensor_1d.tolist()).most_common(1)[0][0]


def uniform_subsample_indices(span_len, sample_len):
    """
    Return `sample_len` integer indices uniformly spaced over [0, span_len-1].
    Always includes first (0) and last (span_len-1) when sample_len >= 2.
    """
    if sample_len == 1:
        return np.array([span_len // 2], dtype=int)
    return np.linspace(0, span_len - 1, num=sample_len, dtype=int)


def generate_subsampled_sequences(data_path, span_frames, sample_frames, step, save_path,
                                  label_from="span"):
    """
    Slide a `span_frames`-wide window; uniformly subsample `sample_frames` from inside it.

    label_from:
        - "span": majority vote over ALL frames in the span (recommended; more robust)
        - "sample": majority vote over only the sampled frames
        - "center": label of the temporal center frame
    """
    print(f"[INFO] Loading data from: {data_path}")
    X, y = torch.load(data_path)  # X: [N_frames, 48], y: [N_frames]
    print(f"[INFO] Loaded data: X={X.shape}, y={y.shape}")

    assert X.ndim == 2, "[ERROR] Expected X shape [N, 48]."
    assert X.shape[1] == 48, "[ERROR] Input feature dim must be 48 (pose + gaze + fm)."
    assert y.ndim == 1 and len(y) == len(X), "[ERROR] y must be 1D w/ same length as X."

    assert 1 <= sample_frames <= span_frames, \
        "[ERROR] sample_frames must be >=1 and <= span_frames."

    idx_map = uniform_subsample_indices(span_frames, sample_frames)  # numpy array
    sequences, seq_labels = [], []

    for start in tqdm(range(0, len(X) - span_frames + 1, step), desc="Sliding span"):
        end = start + span_frames
        span_X = X[start:end]          # [span_frames, 48]
        span_y = y[start:end]          # [span_frames]

        sampled_X = span_X[idx_map]    # [sample_frames, 48]

        if label_from == "span":
            label = most_common_label(span_y)
        elif label_from == "sample":
            label = most_common_label(span_y[idx_map])
        elif label_from == "center":
            label = span_y[span_frames // 2].item() if torch.is_tensor(span_y) else span_y[span_frames // 2]
        else:
            raise ValueError(f"Unknown label_from='{label_from}'")

        sequences.append(sampled_X)
        seq_labels.append(label)

    X_seq = torch.stack(sequences)       # [N_seq, sample_frames, 48]
    y_seq = torch.tensor(seq_labels)     # [N_seq]

    print(f"[INFO] Created {X_seq.shape[0]} sequences of length {sample_frames} (from spans of {span_frames}).")
    print(f"[INFO] Saving to: {save_path}")
    torch.save((X_seq, y_seq), save_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to 1-frame feature .pt file (e.g., pose_gaze_fm_1frame.pt)")
    parser.add_argument("--span_frames", type=int, default=30,
                        help="Raw temporal span size (e.g., 30 frames ≈ 3s).")
    parser.add_argument("--sample_frames", type=int, default=10,
                        help="How many frames to keep from each span (uniformly sampled).")
    parser.add_argument("--step", type=int, default=10,
                        help="Sliding stride between span starts.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save output .pt file (e.g., features_seq_subsampled.pt)")
    parser.add_argument("--label_from", type=str, default="span",
                        choices=["span", "sample", "center"],
                        help="How to derive sequence label.")
    args = parser.parse_args()

    generate_subsampled_sequences(
        data_path=args.data_path,
        span_frames=args.span_frames,
        sample_frames=args.sample_frames,
        step=args.step,
        save_path=args.save_path,
        label_from=args.label_from,
    )
