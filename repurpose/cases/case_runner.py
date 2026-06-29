"""Q4 case studies.

Reads the saved TxGNN zero-shot result file, pulls the top-K drug predictions
for two diseases fixed *before* predictions were seen, and traces a couple of
2-hop KG paths for the known approved drugs. Everything written here comes from
files, not from hand-typed numbers.

Diseases (locked in advance):
    Case A — Familial Hypertrophic Cardiomyopathy (id 24573, n_pos=1, rare)
    Case B — Staphylococcus Aureus Infection (id 5545, n_pos=45, well studied)

Both come from the zero-shot seed-42 run because only that file stores the
per-disease ``top_k_drugs``.
"""

import json
from pathlib import Path

import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PKG_ROOT / "results" / "predictions"

CASE_A_NAME = "familial hypertrophic cardiomyopathy"   # id 24573, n_pos=1
CASE_B_NAME = "staphylococcus aureus infection"         # id 5545, n_pos=45
MODEL, SEED, SPLIT = "txgnn", 42, "zeroshot"


def resolve_disease_id(name: str, edges: pd.DataFrame):
    """Case-insensitive name -> disease id, checking both edge endpoints."""
    for type_col, name_col, id_col in (("y_type", "y_name", "y_id"),
                                       ("x_type", "x_name", "x_id")):
        hit = edges[(edges[type_col] == "disease")
                    & edges[name_col].str.lower().str.contains(name.lower(), na=False)]
        ids = hit[id_col].unique()
        if len(ids):
            if len(ids) > 1:
                print(f"  [warn] {len(ids)} matches for '{name}', using first")
            return str(ids[0])
    return None


def top_predictions(disease_id: str, per_disease: list, edges: pd.DataFrame, k: int = 20):
    rows_for_disease = [r for r in per_disease if str(r["disease_id"]) == str(disease_id)]
    cols = ["disease_id", "relation", "rank", "drug_id", "drug_name", "score", "is_positive"]
    if not rows_for_disease:
        return pd.DataFrame(columns=cols)

    drug_name = (
        edges[edges["x_type"] == "drug"][["x_id", "x_name"]]
        .drop_duplicates("x_id").set_index("x_id")["x_name"].to_dict()
    )
    records = []
    for entry in rows_for_disease:
        for drug in entry.get("top_k_drugs", [])[:k]:
            did = str(drug["drug_id"])
            records.append({
                "disease_id": disease_id,
                "relation": entry["relation"],
                "rank": drug["rank"],
                "drug_id": did,
                "drug_name": drug_name.get(did, "unknown"),
                "score": drug["score"],
                "is_positive": drug["is_positive"],
            })
    return pd.DataFrame(records, columns=cols)


def two_hop_paths(drug_id: str, disease_id: str, edges: pd.DataFrame, limit: int = 3):
    """drug -> intermediate -> disease, checking both disease edge directions."""
    drug_side = edges[edges["x_id"] == drug_id][["y_id", "y_name", "y_type", "relation"]]
    drug_side = drug_side.rename(columns={"y_id": "mid_id", "y_name": "mid_name",
                                          "y_type": "mid_type", "relation": "rel1"})
    inbound = edges[edges["y_id"] == disease_id][["x_id", "relation"]].rename(
        columns={"x_id": "mid_id", "relation": "rel2"})
    outbound = edges[edges["x_id"] == disease_id][["y_id", "relation"]].rename(
        columns={"y_id": "mid_id", "relation": "rel2"})

    found = []
    for disease_side in (inbound, outbound):
        merged = drug_side.merge(disease_side, on="mid_id")
        for _, row in merged.head(limit).iterrows():
            found.append({
                "drug_id": drug_id,
                "via_entity": row["mid_name"],
                "via_type": row["mid_type"],
                "relation_drug_to_entity": row["rel1"],
                "relation_entity_to_disease": row["rel2"],
                "disease_id": disease_id,
            })
        if len(found) >= limit:
            break
    return found[:limit]


def run_cases(edges: pd.DataFrame) -> None:
    result_file = PKG_ROOT / "results" / MODEL / SPLIT / f"seed_{SEED}" / f"{MODEL}.json"
    if not result_file.exists():
        print(f"[error] {result_file} not found — run the TxGNN zeroshot job first.")
        return
    per_disease = json.loads(result_file.read_text()).get("per_disease_results", [])
    if not per_disease:
        print("[error] result file has no per_disease_results.")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for tag, name in (("caseA", CASE_A_NAME), ("caseB", CASE_B_NAME)):
        disease_id = resolve_disease_id(name, edges)
        if disease_id is None:
            print(f"[warn] disease not found: {name}")
            continue
        print(f"\n=== {tag}: {name} (id={disease_id}) ===")

        preds = top_predictions(disease_id, per_disease, edges)
        preds.to_csv(PRED_DIR / f"case_study_{tag}_{MODEL}.csv", index=False)
        if not preds.empty:
            print(preds.to_string(index=False))

        known = edges[(edges["y_id"] == disease_id)
                      & edges["relation"].isin(["indication", "contraindication"])]["x_id"].unique()[:3]
        paths = [p for drug in known for p in two_hop_paths(str(drug), str(disease_id), edges)]
        if paths:
            pd.DataFrame(paths).to_csv(PRED_DIR / f"case_study_{tag}_paths_{MODEL}.csv", index=False)
            print(f"  paths -> case_study_{tag}_paths_{MODEL}.csv")


if __name__ == "__main__":
    print("Loading kg.csv ...")
    kg = pd.read_csv(PKG_ROOT / "data" / "raw" / "kg.csv", low_memory=False)
    run_cases(kg)
