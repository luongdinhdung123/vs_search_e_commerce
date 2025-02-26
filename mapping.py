import os
import json
import random

imgs_folder = "/Users/luongdinhdung/Downloads/vs_search/ecommerce-visual-search/db"
json_folder = "/Users/luongdinhdung/Downloads/vs_search/ecommerce-visual-search/data.json"

types = {
    "jeans": "Jeans", 
    "sandals": "Sandals", 
    "tshirt": "Tshirt"
}
sizes = ["S", "M", "L"]

def find_type(file_name) -> str:
    for type in types.keys():
        if type in file_name:
            return types[type]
    return "New_type"

imgs = [f for f in os.listdir(imgs_folder) if f.endswith("jpg") or f.endswith("jpeg")]

json_datas = []
for idx, img in enumerate(imgs):
    img_name = os.path.splitext(img)[0]
    # print(f"Index: {idx+1}, Img_name: {img_name}")
    json_data = {
        "name": img_name,
        "size": random.choice(sizes),
        "type": find_type(img_name),
        "price": round(random.uniform(10, 200,), 3),
        "path": f"db/{img}"    
    }
    json_datas.append(json_data)

with open(json_folder, "w") as json_file:
    json.dump(json_datas, json_file, indent=4)



