import torch
import torch.nn as nn
import torchvision.models as models
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


models_dict = {
    'resnet18': {'model': models.resnet18, 'output_dim': 512, 'pre_trained_weights': models.ResNet18_Weights.IMAGENET1K_V1},
    'resnet50': {'model': models.resnet50, 'output_dim': 2048, 'pre_trained_weights': models.ResNet50_Weights.IMAGENET1K_V1},
    'vit_b_16': {'model': models.vit_b_16, 'output_dim': 768, 'pre_trained_weights': models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1}
}


class IQAEncoder(nn.Module):
    def __init__(self, feature_dim=128, model_name='resnet18', pretrained=True, freeze_encoder=True):
        super().__init__()

        base = models_dict[model_name]['model']
        encoder_output_dim = models_dict[model_name]['output_dim']
        encoder_weights = models_dict[model_name]['pre_trained_weights'] if pretrained else None

        # Initialize encoder
        encoder = base(weights=encoder_weights)
        encoder.fc = nn.Identity()
        self.encoder = encoder

        # Optionally freeze encoder parameters
        if pretrained and freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection head
        # Dynamically create projection layers with halving dimensions
        dims = []
        dim = encoder_output_dim
        while dim > feature_dim:
            next_dim = max(dim // 2, feature_dim)  # Prevent going below target
            dims.append((dim, next_dim))
            if next_dim == feature_dim:
                break
            dim = next_dim

        layers = []
        for i, (in_dim, out_dim) in enumerate(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 1:  # Apply BatchNorm and ReLU only between layers
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        self.projection_head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.encoder(x)
        out = self.projection_head(features)
        return nn.functional.normalize(out, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(features.shape[0], device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss
    

def extract_features(model, dataloader, label_map, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
            features = model(imgs).cpu()
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def plot_tsne(features, labels, label_map, epoch):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features.numpy())

    plt.figure(figsize=(8, 6))
    num_classes = len(label_map)
    for i in range(num_classes):
        idx = labels.numpy() == i
        label_name = list(label_map.keys())[list(label_map.values()).index(i)]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, alpha=0.6, s=10)

    plt.legend()
    plt.title(f"t-SNE of Embeddings (Epoch {epoch})")
    plt.savefig(f"eval_results/tsne_epoch_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    from datasets import VideoFrameDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from distortions import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAEncoder().to(device)
    print(model.projection_head)
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    distortions = [LensBlur(ksize=7), MotionBlur(degree=10, angle=30), Blackout()]

    transform = transforms.Compose([
        lambda img: RandomDistortion(distortions, p=0.5)(img),
        lambda tup: (transforms.ToPILImage()(tup[0]), tup[1]),
        lambda tup: (transforms.Resize((224, 224))(tup[0]), tup[1]),
        lambda tup: (transforms.ToTensor()(tup[0]), tup[1]),
    ])
    dataset = VideoFrameDataset("data/1.mp4", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    eval_dataset = VideoFrameDataset("data/2.mp4", transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)


    label_map = {distortion.__class__.__name__: i for i, distortion in enumerate(distortions)}
    label_map['None'] = len(label_map)

    for epoch in range(100):
        model.train()
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long, device=device)
            features = model(imgs)
            loss = criterion(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            eval_features, eval_labels = extract_features(model, eval_dataloader, label_map, device)
            plot_tsne(eval_features, eval_labels, label_map, epoch)
            
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")