# import torch

# checkpoint = torch.load("models/best_model_fusion.pt", map_location="cpu")
# print(type(checkpoint))
# print(checkpoint["config"])
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch




path = "models/best_model_fusion.pt"
checkpoint = torch.load(path, map_location="cpu")

print("TYPE:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nTop-level keys:")
    print(checkpoint.keys())

    if "config" in checkpoint:
        print("\nCONFIG:")
        print(checkpoint["config"])

    if "model_state_dict" in checkpoint:
        print("\nFirst 5 weight keys:")
        for i, (k, v) in enumerate(checkpoint["model_state_dict"].items()):
            print(k, v.shape)
            if i >= 4:
                break
else:
    print("\nState dict keys:")
    for i, (k, v) in enumerate(checkpoint.items()):
        print(k, v.shape)
        if i >= 4:
            break
