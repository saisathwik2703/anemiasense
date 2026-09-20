FEATURES = ["age", "sex", "hemoglobin", "rbc", "hematocrit", "mcv", "mch", "mchc"]


def add_indices(df):
    """Derive standard red-cell indices from the values a lab report gives."""
    df["mcv"] = df["hematocrit"] / df["rbc"] * 10        # fL
    df["mch"] = df["hemoglobin"] / df["rbc"] * 10        # pg
    df["mchc"] = df["hemoglobin"] / df["hematocrit"] * 100  # g/dL
    return df
