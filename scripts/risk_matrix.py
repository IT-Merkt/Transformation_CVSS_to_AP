_LEVELS = ["3", "2", "1", "0"]
_TIERS  = ["High", "Medium", "Low", "Very Low"]

# gleiche Logik für Safety, Financial, Operational, Privacy
def _row(lvl):
    return {
        "High"      : "Critical" if lvl == "3" else ("High" if lvl == "2" else ("Medium" if lvl == "1" else "Very Low")),
        "Medium"    : "High"     if lvl == "3" else ("Medium" if lvl == "2" else "Low"),
        "Low"       : "Medium"   if lvl == "3" else ("Low"    if lvl == "2" else "Very Low"),
        "Very Low"  : "Very Low",
    }

MATRIX = {
    f"{q}{lvl}" : _row(lvl)
    for q in "S F O P".split()
    for lvl in _LEVELS
}

def adjust_risk(ap_tier, impacts):
    """impacts = dict(S=.., F=.., O=.., P=..) ; returns adjusted risk"""
    highest = max(impacts.values())  # e.g. "S3"
    table   = MATRIX.get(highest[:2], MATRIX["S3"])
    return table.get(ap_tier, "Unknown")
