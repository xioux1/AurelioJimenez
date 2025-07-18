import numpy as np


def calculate_hit_rate_at_3(df_preds_with_true_and_rank):
    """Calculate HitRate@3 for ranking predictions."""
    hits = 0
    valid_queries_count = 0
    for ranker_id, group in df_preds_with_true_and_rank.groupby('ranker_id'):
        if len(group) <= 10:
            continue
        valid_queries_count += 1
        true_selected_item = group[group['selected'] == 1]
        if not true_selected_item.empty:
            rank_of_true_item = true_selected_item.iloc[0]['predicted_rank']
            if rank_of_true_item <= 3:
                hits += 1
    if valid_queries_count == 0:
        return 0.0
    return hits / valid_queries_count


def lgb_hit_rate_at_3(labels, preds, weight=None, group=None):
    """LightGBM evaluation metric for HitRate@3."""
    if group is None:
        raise ValueError("Group information is required to compute HitRate@3")

    hits = 0
    valid_queries = 0
    idx = 0
    for g in group:
        end = idx + g
        if g > 10:
            valid_queries += 1
            group_preds = preds[idx:end]
            group_labels = labels[idx:end]
            if np.any(group_labels == 1):
                true_idx = np.where(group_labels == 1)[0][0]
                rank = (-group_preds).argsort()
                rank_pos = np.where(rank == true_idx)[0][0] + 1
                if rank_pos <= 3:
                    hits += 1
        idx = end

    score = hits / valid_queries if valid_queries else 0.0
    return "hr@3", score, True


def check_rank_permutation(group):
    N = len(group)
    sorted_ranks = sorted(list(group['selected']))
    expected_ranks = list(range(1, N + 1))
    if sorted_ranks != expected_ranks:
        print(f"Invalid rank permutation for ranker_id: {group['ranker_id'].iloc[0]}")
        print(f"Expected: {expected_ranks}, Got: {sorted_ranks}")
        return False
    return True
