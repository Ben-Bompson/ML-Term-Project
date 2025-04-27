import pandas as pd
import streamlit as st
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

st.title("PC Build Optimizer")
budget = st.number_input("Enter your total budget", min_value=100.0, value=1500.0, step=50.0)

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

def select_best_component_ml(df, model, feature_cols, budget, filters=None):
    candidates = df.copy()
    if filters:
        for condition in filters:
            candidates = candidates.query(condition)
    candidates = candidates[candidates["Price"] <= budget]
    if candidates.empty:
        return None

    X = candidates[feature_cols]
    candidates = candidates.copy()
    candidates["PredictedScore"] = model.predict(X)
    return candidates.sort_values("PredictedScore", ascending=False).iloc[0]

def select_mobo(socket, budget):
    df = mobo_df[(mobo_df["Socket"] == socket) & (mobo_df["Price"] <= budget)]
    if df.empty: return None
    return df.sort_values("Price", ascending=False).iloc[0]

def select_ram(ram_type, min_ram, budget):
    df = ram_df[(ram_df["Type"] == ram_type) & (ram_df["Capacity"] >= min_ram) & (ram_df["Price"] <= budget)]
    if df.empty: return None
    return df.sort_values(["Capacity", "Price"], ascending=[False, False]).iloc[0]

def select_psu(tdp, budget):
    df = psu_df[(psu_df["Wattage"] >= tdp) & (psu_df["Price"] <= budget)]
    if df.empty: return None
    return df.sort_values(["Wattage", "Price"], ascending=[False, False]).iloc[0]

def select_ssd(min_storage, budget):
    df = ssd_df[(ssd_df["Capacity"] >= min_storage) & (ssd_df["Price"] <= budget)]
    if df.empty: return None
    return df.sort_values(["Capacity", "Price"], ascending=[False, False]).iloc[0]

def attempt_build_with_gpu(gpu_row, budget, weights):
    build = {}
    remaining_budget = budget - gpu_row["Price"]
    build["GPU"] = gpu_row

    cpu = select_best_component_ml(cpu_df, cpu_model, ["Benchmark", "TDP"], budget * weights["CPU"])
    if cpu is None: return None
    remaining_budget -= cpu["Price"]
    build["CPU"] = cpu
    socket = cpu["Socket"]

    mobo = select_mobo(socket, budget * weights["MOBO"])
    if mobo is None: return None
    remaining_budget -= mobo["Price"]
    build["MOBO"] = mobo

    ram = select_ram(mobo["RAM Type"], 8, budget * weights["RAM"])
    if ram is None: return None
    remaining_budget -= ram["Price"]
    build["RAM"] = ram

    total_tdp = (gpu_row["TDP"] + cpu["TDP"]) * 2
    psu = select_psu(total_tdp, budget * weights["PSU"])
    if psu is None: return None
    remaining_budget -= psu["Price"]
    build["PSU"] = psu

    ssd = select_ssd(500, budget * weights["SSD"])
    if ssd is None: return None
    remaining_budget -= ssd["Price"]
    build["SSD"] = ssd

    build["Remaining Budget"] = remaining_budget
    return build

if budget >= min_required_budget:
    gpu_model = train_value_model(gpu_df, ["VRAM", "TDP", "G3D"], "G3D")
    cpu_model = train_value_model(cpu_df, ["Benchmark", "TDP"], "Benchmark")
    ssd_model = train_value_model(ssd_df, ["Capacity"], "Capacity / Price")

    gpu_candidates = gpu_df.query("VRAM >= 16000 and Brand in ['AMD', 'Nvidia', 'Intel']")
    gpu_candidates = gpu_candidates[gpu_candidates["Price"] > 0].copy()
    gpu_candidates["PredictedScore"] = gpu_model.predict(gpu_candidates[["VRAM", "TDP", "G3D"]])
    gpu_candidates = gpu_candidates.sort_values("PredictedScore", ascending=False).reset_index(drop=True)

    final_build = None
    for _, gpu_row in gpu_candidates.iterrows():
        result = attempt_build_with_gpu(gpu_row, budget, weights)
        if result and result["Remaining Budget"] >= 0:
            final_build = result
            break

    if final_build is None:
        try:
            gpu = gpu_df.query("VRAM >= 16000 and Price == @min_gpu_price").iloc[0]
            cpu = cpu_df[cpu_df["Price"] == min_cpu_price].iloc[0]
            mobo = mobo_df[(mobo_df["Socket"] == cpu["Socket"])]
            mobo = mobo[mobo["Price"] == min_mobo_price].iloc[0]
            ram = ram_df[(ram_df["Type"] == mobo["RAM Type"]) & (ram_df["Capacity"] >= 8) & (ram_df["Price"] == min_ram_price)].iloc[0]
            psu = psu_df[psu_df["Wattage"] >= estimated_tdp]
            psu = psu[psu["Price"] == min_psu_price].iloc[0]
            ssd = ssd_df[(ssd_df["Capacity"] >= 500) & (ssd_df["Price"] == min_ssd_price)].iloc[0]

            remaining = budget - sum([gpu["Price"], cpu["Price"], mobo["Price"], ram["Price"], psu["Price"], ssd["Price"]])
            final_build = {
                "GPU": gpu, "CPU": cpu, "MOBO": mobo,
                "RAM": ram, "PSU": psu, "SSD": ssd,
                "Remaining Budget": remaining
            }
        except:
            pass

    if final_build:
        st.subheader("Final PC Build")
        st.write(f"**GPU:** {final_build['GPU']['GPU Name']} (${final_build['GPU']['Price']})")
        st.write(f"**CPU:** {final_build['CPU']['CPU Name']} (${final_build['CPU']['Price']})")
        st.write(f"**Motherboard:** {final_build['MOBO']['Chipset']} (${final_build['MOBO']['Price']})")
        st.write(f"**RAM:** {final_build['RAM']['Capacity']}GB {final_build['RAM']['Type']} (${final_build['RAM']['Price']})")
        st.write(f"**PSU:** {final_build['PSU']['Wattage']}W 80+ {final_build['PSU']['Efficiency']} (${final_build['PSU']['Price']})")
        st.write(f"**SSD:** {final_build['SSD']['Capacity']}GB {final_build['SSD']['Type']} (${final_build['SSD']['Price']})")
        st.success(f"Remaining Budget: ${final_build['Remaining Budget']:.2f}")
    else:
        st.error("Unable to build a system within the given budget even after all fallback attempts.")
else:
    st.warning(f"Minimum required budget to build a system is approximately ${min_required_budget:.2f}.")