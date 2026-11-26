"""Download Rx drug labels from openFDA with a pharmacologic class (EPC) and an indications section."""
import requests, pandas as pd, time
rows = []
for skip in range(0, 5000, 100):
    r = requests.get("https://api.fda.gov/drug/label.json", params={
        "search": "_exists_:openfda.pharm_class_epc AND _exists_:indications_and_usage AND openfda.product_type:\"HUMAN PRESCRIPTION DRUG\"",
        "limit": 100, "skip": skip}, timeout=60)
    if r.status_code != 200: print("stop at", skip, r.status_code); break
    for x in r.json()["results"]:
        o = x.get("openfda", {})
        rows.append({"set_id": x.get("set_id"), "generic": "; ".join(o.get("generic_name", [])),
                     "epc": "; ".join(o.get("pharm_class_epc", [])),
                     "indications": " ".join(x["indications_and_usage"]),
                     "has_boxed_warning": int("boxed_warning" in x),
                     "adverse": " ".join(x.get("adverse_reactions", []))[:4000]})
    time.sleep(0.3)
df = pd.DataFrame(rows).drop_duplicates("set_id"); df.to_csv("data/labels_raw.csv", index=False); print(len(df))
