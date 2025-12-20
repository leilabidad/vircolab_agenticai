import torch
import pandas as pd
import torch.nn.functional as F


def extract_cm(model, loader, device, out_csv):
    model.eval()
    rows = []

    with torch.no_grad():
        for img_fr, img_la, _, sid, stid in loader:
            img_fr = img_fr.to(device)
            img_la = img_la.to(device)

            _, emb = model(img_fr, img_la)
            emb = F.normalize(emb, dim=1)

            for i in range(emb.size(0)):
                rows.append(
                    [sid[i], stid[i]] + emb[i].cpu().tolist()
                )

    cols = ["subject_id", "study_id"] + [
        f"Cm_{i}" for i in range(emb.size(1))
    ]

    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
