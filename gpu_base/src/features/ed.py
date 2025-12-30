import pandas as pd

def build_ed_features(cfg):
    edstays = pd.read_csv(
        cfg["paths"]["edstays"],
        usecols=["subject_id","hadm_id","stay_id","intime","outtime","disposition"]
    )

    triage = pd.read_csv(
        cfg["paths"]["triage"],
        usecols=[
            "stay_id","acuity","chiefcomplaint","pain",
            "temperature","heartrate","sbp","dbp"
        ]
    )

    vitals = pd.read_csv(
        cfg["paths"]["vitalsign"],
        usecols=[
            "stay_id","charttime","heartrate","sbp","dbp",
            "resprate","o2sat","temperature"
        ]
    )

    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals = vitals.sort_values("charttime")
    vitals = vitals.groupby("stay_id").first().reset_index()

    df = edstays.merge(triage, on="stay_id", how="left")
    df = df.merge(vitals, on="stay_id", how="left")

    return df
