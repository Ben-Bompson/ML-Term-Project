import pandas as pd

def find_gpu():
    global r_budget
    for i in gpu_df.iloc:
        if i["Brand"] in gpu_brands and i["VRAM"] >= min_vram and i["Price"] <= budget * 0.4:
            r_budget -= i["Price"]
            return i["GPU Name"], i["TDP"]

def find_cpu():
    global r_budget
    for i in cpu_df.iloc:
        if i["Brand"] in cpu_brands and i["Price"] <= budget * 0.2:
            r_budget -= i["Price"]
            return i["CPU Name"], i["Socket"], i["TDP"]
   
def find_mobo():
    global r_budget
    for i in mobo_df.iloc:
        if i["Socket"] == socket and i["Price"] <= budget * 0.15:
            r_budget -= i["Price"]
            return i["Chipset"], i["RAM Type"]
        
def find_ram():
    global r_budget
    for i in ram_df.iloc:
        if i["Type"] in ram_type and i["Capacity"] >= min_ram and i["Price"] <= budget * 0.09:
            r_budget -= i["Price"]
            return i["Capacity"], i["Type"]
        
def find_psu():
    global r_budget
    for i in psu_df.iloc:
        if i["Wattage"] >= total_tdp and i["Price"] <= budget * 0.09:
            r_budget -= i["Price"]
            return i["Wattage"], i["Efficiency"]
        
def find_ssd():
    global r_budget
    for i in ssd_df.iloc:
        if i["Capacity"] >= min_storage and i["Price"] <= budget * 0.07:
            r_budget -= i["Price"]
            return i["Capacity"], i["Type"]
        

if __name__ == "__main__":
    gpu_df = pd.read_csv("./Data/GPU Data.csv")
    cpu_df = pd.read_csv("./Data/CPU Data.csv")
    mobo_df = pd.read_csv("./Data/Motherboard Data.csv")
    ram_df = pd.read_csv("./Data/RAM Data.csv")
    psu_df = pd.read_csv("./Data/PSU Data.csv")
    ssd_df = pd.read_csv("./Data/SSD Data.csv")
    gpu_brands = ["AMD", "Nvidia", "Intel"]
    cpu_brands = ["AMD", "Intel"]
    min_vram = 0
    min_ram = 8
    min_storage = 500
    budget = 1200
    r_budget = budget

    gpu, g_tdp = find_gpu()
    print(gpu+"\nRemaining budget: "+str(r_budget))
    
    cpu, socket, c_tdp = find_cpu()
    print(cpu+"\nRemaining budget: "+str(r_budget))
    total_tdp = (g_tdp + c_tdp) * 2
    
    mobo, ram_type = find_mobo()
    print(mobo+"\nRemaining budget: "+str(r_budget))
    
    ram_capacity, ram_type = find_ram()
    print(str(ram_capacity) +"GB "+ ram_type + " RAM" + "\nRemaining budget: " + str(r_budget))
    
    psu_watts, psu_eff = find_psu()
    print(str(psu_watts)+psu_eff+"\nRemaining budget: "+str(r_budget))
    
    ssd_size, ssd_type = find_ssd()
    print(str(ssd_size)+ssd_type+"\nRemaining budget: "+str(r_budget))
    #print(gpu + "\n" + cpu + "\n" + mobo + " Motherboard\n"
          #+ str(ram_capacity) +"GB "+ ram_type + " RAM" + "\nRemaining budget: " + str(r_budget))