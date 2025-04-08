from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

""" def find_gpu():
    global r_budget
    gpu_budget = budget * weights["GPU"]
    prev_gpu = gpu_df.iloc[0]
    for i in gpu_df.iloc:
        if i["Brand"] in gpu_brands and i["VRAM"] >= min_vram: 
            if i["Price"] <= gpu_budget:
                if abs(i["Price"] - gpu_budget) > abs(prev_gpu["Price"] - gpu_budget):
                    r_budget -= prev_gpu["Price"]
                    return prev_gpu["GPU Name"], prev_gpu["TDP"]
                r_budget -= i["Price"]
                return i["GPU Name"], i["TDP"]
            prev_gpu = i """

def train_gpu_model(gpu_df):
    # Clean and prepare training data
    gpu_df = gpu_df.copy()
    gpu_df = gpu_df[gpu_df["Price"] > 0]  # remove invalid rows
    gpu_df["ValueScore"] = gpu_df["VRAM"] / gpu_df["Price"]
    
    features = ["VRAM", "TDP"]
    X = gpu_df[features]
    y = gpu_df["ValueScore"]
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

def find_gpu_ml(model):
    global r_budget
    candidates = gpu_df[
        (gpu_df["Brand"].isin(gpu_brands)) &
        (gpu_df["VRAM"] >= min_vram) &
        (gpu_df["Price"] <= budget * weights["GPU"])
    ].copy()

    if candidates.empty:
        print("No suitable GPU found within budget.")
        return None, 0

    X_candidates = candidates[["VRAM", "TDP"]]
    candidates["PredictedScore"] = model.predict(X_candidates)

    best_gpu = candidates.sort_values(by="PredictedScore", ascending=False).iloc[0]
    r_budget -= best_gpu["Price"]
    return best_gpu["GPU Name"], best_gpu["TDP"]



def find_cpu():
    global r_budget
    cpu_budget = budget * weights["CPU"]
    prev_cpu = cpu_df.iloc[0]
    for i in cpu_df.iloc:
        if i["Brand"] in cpu_brands:
            if i["Price"] <= budget * weights["CPU"]:
                if abs(i["Price"] - cpu_budget) > abs(prev_cpu["Price"] - cpu_budget):
                    r_budget -= prev_cpu["Price"]
                    return prev_cpu["CPU Name"], prev_cpu["Socket"], prev_cpu["TDP"]
                r_budget -= i["Price"]
                return i["CPU Name"], i["Socket"], i["TDP"]
            prev_cpu = i
   
def find_mobo():
    global r_budget
    mobo_budget = budget * weights["MOBO"]
    prev_mobo = mobo_df.iloc[0]
    for i in mobo_df.iloc:
        if i["Socket"] == socket:
            if i["Price"] <= budget * weights["MOBO"]:
                if abs(i["Price"] - mobo_budget) > abs(prev_mobo["Price"] - mobo_budget):
                    r_budget -= prev_mobo["Price"]
                    return prev_mobo["Chipset"], prev_mobo["RAM Type"]
                r_budget -= i["Price"]
                return i["Chipset"], i["RAM Type"]
            prev_mobo = i
        
def find_ram():
    global r_budget
    ram_budget = budget * weights["RAM"]
    prev_ram = ram_df.iloc[0]
    for i in ram_df.iloc:
        if i["Type"] in ram_type and i["Capacity"] >= min_ram:
            if i["Price"] <= budget * weights["RAM"]:
                if abs(i["Price"] - ram_budget) > abs(prev_ram["Price"] - ram_budget):
                    r_budget -= prev_ram["Price"]
                    return prev_ram["Capacity"], prev_ram["Type"]
                r_budget -= i["Price"]
                return i["Capacity"], i["Type"]
            prev_ram = i
            
def find_psu():
    global r_budget
    psu_budget = budget * weights["PSU"]
    prev_psu = psu_df.iloc[0]
    for i in psu_df.iloc:
        if i["Wattage"] >= total_tdp:
            if i["Price"] <= budget * weights["PSU"]:
                if abs(i["Price"] - psu_budget) > abs(prev_psu["Price"] - psu_budget):
                    r_budget -= prev_psu["Price"]
                    return prev_psu["Wattage"], prev_psu["Efficiency"]
                r_budget -= i["Price"]
                return i["Wattage"], i["Efficiency"]
            prev_psu = i
        
def find_ssd():
    global r_budget
    ssd_budget = budget * weights["SSD"]
    prev_ssd = ssd_df.iloc[0]
    for i in ssd_df.iloc:
        if i["Capacity"] >= min_storage:
            if i["Price"] <= budget * weights["SSD"]:
                if abs(i["Price"] - ssd_budget) > abs(prev_ssd["Price"] - ssd_budget):
                    r_budget -= prev_ssd["Price"]
                    return prev_ssd["Capacity"], prev_ssd["Type"]
                r_budget -= i["Price"]
                return i["Capacity"], i["Type"]
            prev_ssd = i
        
if __name__ == "__main__":
    gpu_df = pd.read_csv("./Data/GPU Data.csv")
    cpu_df = pd.read_csv("./Data/CPU Data.csv")
    mobo_df = pd.read_csv("./Data/Motherboard Data.csv")
    ram_df = pd.read_csv("./Data/RAM Data.csv")
    psu_df = pd.read_csv("./Data/PSU Data.csv")
    ssd_df = pd.read_csv("./Data/SSD Data.csv")
    gpu_brands = ["AMD", "Nvidia", "Intel"]
    cpu_brands = ["AMD", "Intel"]
    min_vram = 16000
    min_ram = 8
    min_storage = 500
    
    #budget = 1500
    budget = float(input("Enter your total budget: "))
    r_budget = budget
    weights = {"GPU": 0.4, "CPU": 0.2, "MOBO": 0.15, "RAM": 0.09, "PSU": 0.09, "SSD": 0.07}

    gpu_model = train_gpu_model(gpu_df)
    gpu, g_tdp = find_gpu_ml(gpu_model)
    print(gpu+"\nRemaining budget: "+str(r_budget))
    
    cpu, socket, c_tdp = find_cpu()
    print(cpu+"\nRemaining budget: "+str(r_budget))
    total_tdp = (g_tdp + c_tdp) * 2
    
    mobo, ram_type = find_mobo()
    print(mobo+" Motherboard\nRemaining budget: "+str(r_budget))
    
    ram_capacity, ram_type = find_ram()
    print(str(ram_capacity) +"GB "+ ram_type + " RAM" + "\nRemaining budget: " + str(r_budget))
    
    psu_watts, psu_eff = find_psu()
    print(str(psu_watts)+"W 80+ "+psu_eff+" PSU\nRemaining budget: "+str(r_budget))
    
    ssd_size, ssd_type = find_ssd()
    print(str(ssd_size)+"GB "+ssd_type+" SSD\nRemaining budget: "+str(r_budget))