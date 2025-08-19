import argparse
import torch
from torchvision import transforms

from dataset import ProductDataset
from model   import MultiTaskModel
from utils   import extract_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", required=True)
parser.add_argument("--image_root", required=True)
parser.add_argument("--model_ckpt", default="checkpoints/model_final.pth")
parser.add_argument("--image_size", type=int, default=224)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与训练一致的预处理
tfm = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ProductDataset(args.csv_path, args.image_root, transform=tfm)  # 内部会拼上 mask 通道
model = MultiTaskModel(pretrained=False).to(device).eval()  # 推理可不加载 ImageNet 预训

ckpt = torch.load(args.model_ckpt, map_location=device)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)

pids, feats = extract_embeddings(model, dataset, device=device)
print("向量库大小:", feats.shape)
torch.save({"pid": pids, "feat": feats}, "embeddings.pt")
print("✅ embeddings.pt 已生成，可用于检索")
