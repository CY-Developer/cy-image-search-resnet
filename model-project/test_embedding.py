import torch, argparse
from dataset import ProductDataset
from model   import MultiTaskModel
from utils   import extract_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", required=True)
parser.add_argument("--image_root", required=True)
parser.add_argument("--model_ckpt", default="checkpoints/best_model.pth")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = ProductDataset(args.csv_path, args.image_root,
                         transform=None)          # 与训练同一策略
model = MultiTaskModel()
ckpt  = torch.load(args.model_ckpt, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

pids, feats = extract_embeddings(model, dataset, device=device)
print("向量库大小:", feats.shape)
torch.save({"pid": pids, "feat": feats}, "embeddings.pt")
print("✅ embeddings.pt 已生成，可用于检索")
