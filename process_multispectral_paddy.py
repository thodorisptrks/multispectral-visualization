import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from collections import defaultdict

parent_dir = os.path.join(".", "Multispectral Images on Paddy- Sri Lanka")
all_files = os.listdir(parent_dir)

grouped_files = defaultdict(dict)

for file in all_files:
    base_name = os.path.basename(file)
    
    if "_D.JPG" in base_name:
        key = base_name.replace("_D.JPG","")
        grouped_files[key]["Original_grey"] = file
    
    elif "_MS_G.TIF" in base_name:
        key = base_name.replace("_MS_G.TIF","")
        grouped_files[key]["Green"] = file
        
    elif "_MS_R.TIF" in base_name:
        key = base_name.replace("_MS_R.TIF","")
        grouped_files[key]["Red"] = file
        
    elif "_MS_RE.TIF" in base_name:
        key = base_name.replace("_MS_RE.TIF","")
        grouped_files[key]["Red Edge"] = file

    elif "_MS_NIR.TIF" in base_name:
        key = base_name.replace("_MS_NIR.TIF","")
        grouped_files[key]["NIR"] = file        
        
for key in list(grouped_files.keys()):
    if not len(grouped_files[key]) == 5:
        del grouped_files[key]

# Function to read a band
def read_original(orig_path):
    orig_im = plt.imread(orig_path)
    return orig_im

def read_band(band_path):
    with rasterio.open(band_path) as src:
        band = src.read(1)
    return band

grouped_files_read = defaultdict(lambda: defaultdict(dict))

for key_1 in grouped_files:
    grouped_files_read[key_1]["Original"]["Original"] = read_original(parent_dir + "/{}".format(grouped_files[key_1]["Original_grey"]))
    
    for key_2 in grouped_files[key_1]:        
        grouped_files_read[key_1]["Bands"][key_2] = read_band(parent_dir + "/{}".format(grouped_files[key_1][key_2]))

    grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"] = np.dstack((
        grouped_files_read[key_1]["Bands"]["NIR"],
        grouped_files_read[key_1]["Bands"]["Red"],
        grouped_files_read[key_1]["Bands"]["Green"]))
    
    grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"] =  (
        grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"] - grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"].min()
        ) / (grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"].max() - grouped_files_read[key_1]["Composites"]["False-Color Comp. (NIR-R-G)"].min())
    
    grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"] = np.dstack((
        grouped_files_read[key_1]["Bands"]["Red Edge"], 
        grouped_files_read[key_1]["Bands"]["Red"], 
        grouped_files_read[key_1]["Bands"]["Green"]))
    
    grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"] =  (
        grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"] -  grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"].min()
        ) / (grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"].max() - grouped_files_read[key_1]["Composites"]["Custom Comp. (RE-R-G)"].min())
    
    grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"] = np.dstack((
        grouped_files_read[key_1]["Bands"]["NIR"], 
        grouped_files_read[key_1]["Bands"]["Red Edge"], 
        grouped_files_read[key_1]["Bands"]["Red"]))
    
    grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"] =  (
        grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"] -  grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"].min()
        ) / (grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"].max() - grouped_files_read[key_1]["Composites"]["Custom Comp. (NIR-RE-R)"].min())
   
new_order_orig = ["Original"]   
new_order_bands = ["Original_grey","Green", "Red", "Red Edge", "NIR"]
new_order_composites = ["False-Color Comp. (NIR-R-G)", "Custom Comp. (RE-R-G)", "Custom Comp. (NIR-RE-R)"]

for key_1 in grouped_files:
    grouped_files_read[key_1]["Bands"] = {key: grouped_files_read[key_1]["Bands"][key] for key in new_order_bands}
    grouped_files_read[key_1]["Composites"] = {key: grouped_files_read[key_1]["Composites"][key] for key in new_order_composites}
    
    
# Plot Original, Bands and Composites
base_photos_dir = os.path.join(".", "images_bands")

for image_id, inner_dict in grouped_files_read.items():
    for category, bands_dict in inner_dict.items():  # category = "Bands" or "Composites"
        
        # Use correct band order based on category
        if category == "Original":
            band_names = new_order_orig
        elif category == "Bands":
            band_names = new_order_bands
        elif category == "Composites":
            band_names = new_order_composites
        
        num_bands = len(band_names)
        fig, axs = plt.subplots(num_bands, 1, figsize=(10 * num_bands, 10), dpi=100)
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]  # handle case with only 1 subplot

        for i, band_name in enumerate(band_names):
            band = bands_dict[band_name]
            if band.ndim == 3:
                axs[i].imshow(band)
            elif band.ndim == 2:
                axs[i].imshow(band, cmap="gray")
            else:
                axs[i].imshow(band)  # RGB composite
            axs[i].set_title(band_name)
            axs[i].axis("off")

        plt.suptitle(f"{category} for {image_id}", fontsize=24)
        plt.tight_layout()
        
        # Create output directory per category
        image_dir = os.path.join(base_photos_dir, image_id)
        os.makedirs(image_dir, exist_ok=True)
        
        # Save the figure
        save_path = os.path.join(image_dir, f"{category}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)