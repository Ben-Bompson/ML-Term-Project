import pandas as pd

def find_gpu():
    global r_budget
    for i in gpu_df.iloc:
        if i["Brand"] in gpu_brands and i["VRAM"] >= min_vram and i["Price"] <= r_budget/2.5:
            r_budget -= i["Price"]
            return i["GPU Name"]

def find_cpu():
    global r_budget
    for i in cpu_df.iloc:
        if i["Brand"] in cpu_brands and i["Price"] <= r_budget/2.5:
            r_budget -= i["Price"]
            return i["CPU Name"], i["Socket"]
   
def find_mobo():
    global r_budget
    for i in mobo_df.iloc:
        if i["Socket"] == socket and i["Price"] <= r_budget/2.5:
            r_budget -= i["Price"]
            return i["Chipset"], i["RAM Type"]
        
def find_ram():
    global r_budget
    for i in ram_df.iloc:
        if i["Type"] in ram_type and i["Capacity"] >= min_ram and i["Price"] <= r_budget/2.5:
            r_budget -= i["Price"]
            return i["Capacity"], i["Type"]

if __name__ == "__main__":
    gpu_df = pd.read_csv("./Data/GPU Data.csv")
    cpu_df = pd.read_csv("./Data/CPU Data.csv")
    mobo_df = pd.read_csv("./Data/Motherboard Data.csv")
    ram_df = pd.read_csv("./Data/RAM Data.csv")
    gpu_brands = ["AMD", "Nvidia", "Intel"]
    cpu_brands = ["AMD", "Intel"]
    min_vram = 16000
    min_ram = 16
    budget = 1600.0
    r_budget = budget

    gpu = find_gpu()
    cpu, socket = find_cpu()
    mobo, ram_type = find_mobo()
    ram_capacity, ram_type = find_ram()
    print(gpu + "\n" + cpu + "\n" + mobo + " Motherboard\n"
          + str(ram_capacity) +"GB "+ ram_type + " RAM" + "\nRemaining budget: " + str(r_budget))