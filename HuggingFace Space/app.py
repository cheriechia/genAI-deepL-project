import torch
import gradio as gr
import os
import yaml
import pandas as pd
import joblib
import re
import wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))

from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image

from fusion_model import FusionModel
from src.sub_models import load_best_bert, load_best_mlp, load_best_cnn, get_run_by_id

DEVICE = "cpu"


# =============================
# Load metadata + dataset
# =============================

test_df = pd.read_csv("data/test_df.csv")

numeric_cols = [
    "following",
    "follower_following_ratio",
    "is_weekend",
    "has_location",
    "is_carousel",
    "num_images",
    "is_sponsored",
    "caption_word_count",
    "num_hashtags"
]

categorical_cols = ["day", "hour"]

# Test data
test_df = test_df.reset_index(drop=True)

# =============================
# Load preprocessors
# =============================

if os.environ.get("SPACE_ID"): # Huggingface space
    # Download fusion model from HF Hub
    preprocessor_path = hf_hub_download(
        repo_id="chiaruiqi/instagram-posts-model",
        filename="models/preprocessor_mlp.pkl",
        token=os.environ["HF_TOKEN"]
    )
else: # local
    # Temp fusion model local directory for local testing
    preprocessor_path = "models/preprocessor_mlp.pkl"

preprocessor = joblib.load(preprocessor_path)

# =============================
# Load Submodels
# =============================

with open("config/fusion_selected_runs.yaml") as f:
    run_ids = yaml.safe_load(f)

# BERT
best_bert_run = get_run_by_id(run_id=run_ids["bert"])
bert_model, tokenizer, bert_config = load_best_bert(best_bert_run)
bert_model.eval()

# MLP
best_mlp_run = get_run_by_id(run_id=run_ids["mlp"]) 
mlp_model, mlp_config = load_best_mlp(best_mlp_run)
mlp_model.eval()

# CNN
best_cnn_run = get_run_by_id(run_id=run_ids["cnn"])
cnn_model, cnn_config = load_best_cnn(best_cnn_run)
cnn_model.eval()

# =============================
# Load Fusion Model
# =============================
if os.environ.get("SPACE_ID"): # Huggingface space
    # Download fusion model from HF Hub
    model_path = hf_hub_download(
        repo_id="chiaruiqi/instagram-posts-model",
        filename="models/best_model_fusion.pt",
        token=os.environ["HF_TOKEN"]
    )
else: # local
    # Temp fusion model local directory for local testing
    model_path = "models/best_model_fusion.pt"

checkpoint = torch.load(model_path, map_location="cpu")

state_dict = checkpoint["model_state_dict"]

# Derive dimensions dynamically
first_weight = state_dict["classifier.0.weight"]
hidden_dim, input_dim = first_weight.shape

dropout = checkpoint["config"]["dropout"]

# Rebuild model dynamically
fusion_model = FusionModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    dropout=dropout
)

fusion_model.load_state_dict(state_dict)
fusion_model.eval()

# =============================
# Image transform (cached)
# =============================

IMAGE_SIZE = 224  # ResNet expects 224x224
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# Helper functions
# =============================
def load_image(image_path):
    """
    Convert dataset image path → local deployment image path
    Keeps subfolder structure after 'Data'
    """
    BASE_IMAGE_DIR = "data/images"

    if image_path is None:
        return None

    # Normalize slashes
    image_path = image_path.replace("\\", "/")

    # Extract relative path after "Data/"
    if "Data/" in image_path:
        relative_path = image_path.split("Data/")[1]
    else:
        relative_path = os.path.basename(image_path)

    full_path = os.path.join("data/images", relative_path)

    if os.path.exists(full_path):
        return Image.open(full_path).convert("RGB")

    print("Image not found:", full_path)  # debug line
    return None

def predict(caption, image, *metadata_inputs):
    """
    caption: str
    image: PIL Image
    metadata_list: list of metadata features
    """

    with torch.no_grad():

        # =============================
        # Text Feature (BERT)
        # =============================

        inputs = tokenizer(
            caption or "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=bert_config["max_len"]
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        bert_feat = bert_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_features=True
        )

        # =============================
        # Image Feature (CNN)
        # =============================
        if image is None:
            image = Image.new("RGB", (224, 224))
            
        img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        cnn_feat = cnn_model(
            img_tensor,
            return_features=True
        )

        # =============================
        # Metadata Feature (MLP)
        # =============================
        numeric_values = metadata_inputs[:len(numeric_cols)]
        categorical_values = metadata_inputs[len(numeric_cols):]

        metadata_dict = dict(zip(
            numeric_cols + categorical_cols,
            list(numeric_values) + list(categorical_values)
        ))

        meta_df = pd.DataFrame([metadata_dict])
        meta_processed = preprocessor.transform(meta_df)

        meta_tensor = torch.tensor(
            meta_processed,
            dtype=torch.float32
        ).to(DEVICE)


        mlp_feat = mlp_model(
            meta_tensor,
            return_features=True
        )

        # =============================
        # Fusion
        # =============================

        fusion_input = torch.cat(
            [bert_feat, cnn_feat, mlp_feat],
            dim=1
        )

        logits = fusion_model(fusion_input)

        probs = torch.softmax(logits, dim=1)

    return {
        "Low": probs[0,0].item(),
        "Medium": probs[0,1].item(),
        "High": probs[0,2].item()
    }

# =============================
# Sample predictor
# =============================
def predict_from_sample(idx):

    idx = int(idx)

    if idx >= len(test_df):
        return {"Error": "Sample index out of range"}

    row = test_df.iloc[idx]

    caption = row["caption"]
    image = load_image(row["image_path"])

    metadata_inputs = [
        row[col] for col in numeric_cols + categorical_cols
    ]

    return predict(caption, image, *metadata_inputs)


# ==================
# Gradio
# ==================
def compute_caption_stats(caption):
    # split text into words
    words = caption.lower().split()
    # only keep words that **do not start with #**
    words = [w for w in words if not w.startswith('#')]
    # optionally, keep only alphanumeric parts
    words = [re.sub(r'\W+', '', w) for w in words if re.sub(r'\W+', '', w)]
    words = len(words)
    
    hashtags = caption.count("#")
    return words, hashtags

def compute_ratio(following, followers):
    follower_following_ratio = (followers+1) / (following+1)
    return follower_following_ratio

def compute_is_weekend(day):
    return 1 if day in ["Sat", "Sun"] else 0

def predict_wrapper(caption, image,
                    following, followers,
                    day, hour,
                    is_carousel, is_sponsored,
                    has_location,
                    num_images):
    caption_word_count, num_hashtags = compute_caption_stats(caption)
    follower_following_ratio = compute_ratio(int(following), int(followers))
    is_weekend = compute_is_weekend(day)

    metadata = [
        int(following),
        follower_following_ratio,
        is_weekend,
        int(has_location),
        int(is_carousel),
        num_images,
        int(is_sponsored),
        caption_word_count,
        num_hashtags,
        day,
        hour
    ]

    # for reference
    # numeric_cols = [
    #     "following",
    #     "follower_following_ratio",
    #     "is_weekend",
    #     "has_location",
    #     "is_carousel",
    #     "num_images",
    #     "is_sponsored",
    #     "caption_word_count",
    #     "num_hashtags"
    # ]

    # categorical_cols = ["day", "hour"]

    return predict(caption, image, *metadata)



# UI
with gr.Blocks() as demo:

    gr.Markdown("## Instagram Engagement Predictor")

    # ============================
    # 3 Column Layout
    # ============================

    with gr.Row():

        # =====================
        # LEFT COLUMN — Samples
        # =====================
        with gr.Column(scale=1):

            gr.Markdown("### Sample Posts")

            sample_dropdown = gr.Dropdown(
                choices=[str(i) for i in range(len(test_df))],
                label="Select Sample Post"
            )

            load_sample_btn = gr.Button("Load Sample")
            

        # =====================
        # MIDDLE COLUMN — Inputs
        # =====================
        with gr.Column(scale=2):

            gr.Markdown("### Post Features")

            caption = gr.Textbox(label="Caption")

            word_count = gr.Number(label="Word Count", interactive=False)
            hashtag_count = gr.Number(label="Hashtag Count", interactive=False)

            image = gr.Image(type="pil", label="Post Image")

            following = gr.Number(label="Following")
            followers = gr.Number(label="Followers")

            # Example choices
            day_choices = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            hour_choices = list(range(24))
            day = gr.Dropdown(day_choices, label="Day")
            hour = gr.Dropdown(hour_choices, label="Hour")

            is_carousel = gr.Checkbox(label="Carousel Post?")
            is_sponsored = gr.Checkbox(label="Sponsored?")
            has_location = gr.Checkbox(label="Location provided?")

            num_images = gr.Number(label="Number of Images")

            

        # =====================
        # RIGHT COLUMN — Output
        # =====================
        with gr.Column(scale=1):

            gr.Markdown("### Engagement label")

            output_label = gr.Label()

            gr.Markdown('#### Expected engagement label (from test sample)')

            engagement_label = gr.Textbox(
                label="Expected Engagement",
                interactive=False
            )

            predict_btn = gr.Button("Predict")


    # ============================
    # Caption Auto Feature Extraction
    # ============================            
    # Event listener
    caption.change(
        fn=compute_caption_stats,
        inputs=caption,
        outputs=[word_count, hashtag_count]
    )

    # ============================
    # Sample Loader
    # ============================

    def load_sample(sample_idx):
        '''
        Load a sample from the test dataframe
        '''

        row = test_df.iloc[int(sample_idx)]

        caption_val = row["caption"]

        img = load_image(row["image_path"])
        
        # Dataframe contains follower_following_ratio instead of followers
        # but UI requires followers
        # wrapped again later to produce follower_following_ratio
        followers = (row["follower_following_ratio"] * (row["following"]+1) )-1

        label_map = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        true_label = label_map[int(row["engagement_label"])]
        
        return (
            caption_val,
            row["following"],
            int(followers),
            row["day"],
            int(row["hour"]),
            bool(row["is_carousel"]),
            bool(row["is_sponsored"]),
            bool(row["has_location"]),
            row["num_images"],
            img,
            true_label
        )

    load_sample_btn.click(
        load_sample,
        inputs=sample_dropdown,
        outputs=[
            caption,
            following,
            followers,
            day,
            hour,
            is_carousel,
            is_sponsored,
            has_location,
            num_images,
            image,
            engagement_label
        ]
    )

    # ============================
    # Prediction Button
    # ============================
    predict_btn.click(
        predict_wrapper,
        inputs=[
            caption, image,
            following, followers,
            day, hour,
            is_carousel, is_sponsored,
            has_location,
            num_images
        ],
        outputs=output_label
    )


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"): # Huggingface space
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else: # local
        demo.launch()

