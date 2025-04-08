import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# === Load Data ===
gpu_df = pd.read_csv("./Data/GPU Data.csv")
cpu_df = pd.read_csv("./Data/CPU Data.csv")
mobo_df = pd.read_csv("./Data/Motherboard Data.csv")
ram_df = pd.read_csv("./Data/RAM Data.csv")
psu_df = pd.read_csv("./Data/PSU Data.csv")
ssd_df = pd.read_csv("./Data/SSD Data.csv")

# === Budgeting ===
while True:
    try:
        budget = float(input("Enter your total budget (e.g. 3000): "))
        break
    except ValueError:
        print("ERROR Invalid input. Please enter a number (e.g. 3000).")

# === Budget Feasibility Check ===
def get_min_valid_part(df, price_col="Price", extra_filter=None):
    if extra_filter:
        df = df.query(extra_filter)
    df = df[df[price_col] > 0]
    return df[price_col].min() if not df.empty else float("inf")

min_gpu_price = get_min_valid_part(gpu_df, "Price", "VRAM >= 16000")
min_cpu_price = get_min_valid_part(cpu_df)
min_mobo_price = get_min_valid_part(mobo_df)
min_ram_price = get_min_valid_part(ram_df, "Price", "Capacity >= 8")
min_ssd_price = get_min_valid_part(ssd_df, "Price", "Capacity >= 500")
min_gpu_tdp = gpu_df.query("VRAM >= 16000")["TDP"].min()
min_cpu_tdp = cpu_df["TDP"].min()
estimated_tdp = (min_gpu_tdp + min_cpu_tdp) * 2
min_psu_price = get_min_valid_part(psu_df, "Price", f"Wattage >= {estimated_tdp}")

min_required_budget = sum([
    min_gpu_price, min_cpu_price, min_mobo_price,
    min_ram_price, min_ssd_price, min_psu_price
])

if budget < min_required_budget - 10:
    print(f"ERROR Budget too low to build a system. Minimum required: ${min_required_budget:.2f}")
    exit()

r_budget = budget
weights = {"GPU": 0.4, "CPU": 0.2, "MOBO": 0.15, "RAM": 0.09, "PSU": 0.09, "SSD": 0.07}

# === ML Trainer ===
def train_value_model(df, feature_cols, target_expr):
    df = df.copy()
    df = df[df["Price"] > 0]
    df["ValueScore"] = df.eval(target_expr)

    X = df[feature_cols]
    y = df["ValueScore"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline

# === Hybrid ML Selector ===
def select_best_component_ml(df, model, feature_cols, budget, filters=None, price_top_percentile=0.75):
    candidates = df.copy()
    if filters:
        for condition in filters:
            candidates = candidates.query(condition)
    candidates = candidates[candidates["Price"] <= budget]
    if candidates.empty:
        return None

    price_threshold = candidates["Price"].quantile(price_top_percentile)
    top_candidates = candidates[candidates["Price"] >= price_threshold]
    if top_candidates.empty:
        top_candidates = candidates

    X = top_candidates[feature_cols]
    top_candidates = top_candidates.copy()
    top_candidates["PredictedScore"] = model.predict(X)
    best = top_candidates.sort_values("PredictedScore", ascending=False).iloc[0]
    return best

# === Rule-Based Selectors ===
def select_mobo(socket, budget):
    df = mobo_df[(mobo_df["Socket"] == socket) & (mobo_df["Price"] <= budget)]
    if df.empty:
        return None
    return df.sort_values("Price", ascending=False).iloc[0]

def select_ram(ram_type, min_ram, budget):
    df = ram_df[(ram_df["Type"] == ram_type) & (ram_df["Capacity"] >= min_ram) & (ram_df["Price"] <= budget)]
    if df.empty:
        return None
    return df.sort_values("Capacity", ascending=False).iloc[0]

def select_psu(total_tdp, budget):
    df = psu_df[(psu_df["Wattage"] >= total_tdp) & (psu_df["Price"] <= budget)]
    if df.empty:
        return None
    return df.sort_values("Wattage", ascending=True).iloc[0]

def select_ssd(min_storage, budget):
    df = ssd_df[(ssd_df["Capacity"] >= min_storage) & (ssd_df["Price"] <= budget)]
    if df.empty:
        return None
    return df.sort_values("Capacity", ascending=False).iloc[0]

# === Train Models ===
gpu_model = train_value_model(gpu_df, ["VRAM", "TDP", "G3D"], "G3D / Price")
cpu_model = train_value_model(cpu_df, ["Benchmark", "TDP"], "Benchmark / Price")
ssd_model = train_value_model(ssd_df, ["Capacity"], "Capacity / Price")

# === GPU Selection ===
gpu = select_best_component_ml(
    gpu_df, gpu_model, ["VRAM", "TDP", "G3D"], budget * weights["GPU"],
    ["VRAM >= 16000", "Brand in ['AMD', 'Nvidia', 'Intel']"]
)

# === Rebalance Budget if GPU selection fails ===
if gpu is None:
    print("WARNING GPU could not be selected with default weight. Rebalancing budgets...")
    gpu = gpu_df.query("VRAM >= 16000 and Price == @min_gpu_price").iloc[0]
    cpu = cpu_df[cpu_df["Price"] == min_cpu_price].iloc[0]
    socket = cpu["Socket"]
    mobo = mobo_df[(mobo_df["Socket"] == socket)].sort_values("Price").head(1)
    if mobo.empty:
        print("ERROR No suitable Motherboard found for the selected CPU socket.")
        exit()
    mobo = mobo.iloc[0]
    ram = ram_df[(ram_df["Type"] == mobo["RAM Type"]) & (ram_df["Capacity"] >= 8)].sort_values("Price").head(1)
    if ram.empty:
        print("ERROR No compatible RAM found within the budget.")
        exit()
    ram = ram.iloc[0]
    total_tdp = (gpu["TDP"] + cpu["TDP"]) * 2
    psu = psu_df[psu_df["Wattage"] >= total_tdp].sort_values("Price").head(1)
    if psu.empty:
        print("ERROR Rebalanced PSU not found for required wattage.")
        exit()
    psu = psu.iloc[0]
    ssd = ssd_df[ssd_df["Price"] == min_ssd_price].iloc[0]
    r_budget = budget - sum([gpu["Price"], cpu["Price"], mobo["Price"], ram["Price"], psu["Price"], ssd["Price"]])

    print("Final PC Build (Fallback Mode):")
    print("----------------------------------")
    print(f"GPU: {gpu['GPU Name']} (${gpu['Price']})")
    print(f"CPU: {cpu['CPU Name']} (${cpu['Price']})")
    print(f"Motherboard: {mobo['Chipset']} (${mobo['Price']})")
    print(f"RAM: {ram['Capacity']}GB {ram['Type']} (${ram['Price']})")
    print(f"PSU: {psu['Wattage']}W 80+ {psu['Efficiency']} (${psu['Price']})")
    print(f"SSD: {ssd['Capacity']}GB {ssd['Type']} (${ssd['Price']})")
    print(f"Remaining Budget: ${r_budget:.2f}")
    exit()

# === Standard Selection ===
r_budget -= gpu["Price"]
cpu = select_best_component_ml(cpu_df, cpu_model, ["Benchmark", "TDP"], budget * weights["CPU"])
if cpu is None:
    print("ERROR No suitable CPU found within the budget.")
    exit()
r_budget -= cpu["Price"]
socket = cpu["Socket"]

mobo = select_mobo(socket, budget * weights["MOBO"])
if mobo is None:
    print("ERROR No suitable Motherboard found for the selected CPU socket.")
    exit()
r_budget -= mobo["Price"]

ram = select_ram(mobo["RAM Type"], 8, budget * weights["RAM"])
if ram is None:
    print("ERROR No compatible RAM found within the budget.")
    exit()
r_budget -= ram["Price"]

total_tdp = (gpu["TDP"] + cpu["TDP"]) * 2
psu = select_psu(total_tdp, budget * weights["PSU"])
if psu is None:
    print("ERROR No suitable PSU found for the power requirement.")
    exit()
r_budget -= psu["Price"]

ssd = select_ssd(500, budget * weights["SSD"])
if ssd is None:
    print("ERROR No suitable SSD found within the budget.")
    exit()
r_budget -= ssd["Price"]

# === Final Build Summary ===
final_build = {
    "GPU": f'{gpu["GPU Name"]} (${gpu["Price"]})',
    "CPU": f'{cpu["CPU Name"]} (${cpu["Price"]})',
    "Motherboard": f'{mobo["Chipset"]} (${mobo["Price"]})',
    "RAM": f'{ram["Capacity"]}GB {ram["Type"]} (${ram["Price"]})',
    "PSU": f'{psu["Wattage"]}W 80+ {psu["Efficiency"]} (${psu["Price"]})',
    "SSD": f'{ssd["Capacity"]}GB {ssd["Type"]} (${ssd["Price"]})',
    "Remaining Budget": r_budget
}

for part, info in final_build.items():
    print(f"{part}: {info}")
