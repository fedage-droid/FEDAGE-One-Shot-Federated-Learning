import argparse
import torch
from datasets import get_dataloaders
from backbones import get_encoder
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import json
from dp_utils import add_dp_noise
import math

# -------------------------------
# GPU preprocessing transform
# -------------------------------
def gpu_transform(x, device):
    # Resize (bilinear interpolation)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    # If grayscale (MNIST/FashionMNIST), expand to 3 channels
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    # Normalize (ImageNet-style)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x

def to_device(x, device):
    return x.to(device, non_blocking=torch.cuda.is_available())


# -------------------------------
# Parse arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "stl10", "svhn", "mnist", "fashionmnist"])
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet", "squeezenet", "vgg19"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument("--mixture_coef", type=float, default=0.75)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--dp_epsilon", type=float, default=float("inf"),
                    help="Privacy budget ε (default = inf = no DP).")
    parser.add_argument("--dp_delta", type=float, default=1e-5,
                        help="Target δ for DP (default = 1e-5).")
    parser.add_argument("--dp_clip_norm", type=float, default=1.0,
                        help="L2 clipping bound for latents before noise.")

    return parser.parse_args()


# -------------------------------
# Main pipeline
# -------------------------------
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Number of classes
    if args.dataset in ["cifar10", "stl10", "svhn", "mnist", "fashionmnist"]:
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    # Load client/test dataloaders (from datasets.py)
    client_loaders, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        alpha=args.alpha,
        num_clients=args.num_clients,
        batch_size=args.batch_size
    )
    print("Client and Test loaders loaded")

    # ===== Encoder (frozen) =====
    encoder = get_encoder(name=args.backbone, output_dim=args.output_dim).to(device)
    encoder.eval()
    print("Encoder Initialised and frozen")

    # ===== Simple Classifier =====
    class Classifier(nn.Module):
        def __init__(self, input_dim=64, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    client_classifiers = [Classifier(args.output_dim, num_classes).to(device) for _ in range(args.num_clients)]
    optimizers = [optim.Adam(clf.parameters(), lr=args.lr) for clf in client_classifiers]
    criterion = nn.CrossEntropyLoss()

    # ===== Training client classifiers =====
    for client_id, loader in enumerate(client_loaders):
        clf = client_classifiers[client_id]
        optimizer = optimizers[client_id]
        for epoch in range(args.epochs):
            clf.train()
            total, correct, running_loss = 0, 0, 0.0
            for inputs, labels in loader:
                inputs = to_device(inputs, device)
                labels = to_device(labels, device)

                inputs = gpu_transform(inputs, device)

                with torch.no_grad():
                    latent = encoder(inputs)

                optimizer.zero_grad()
                outputs = clf(latent)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            acc = 100 * correct / total
            print(f"Client {client_id+1} - Epoch {epoch+1} - Loss: {running_loss/len(loader):.4f} - Acc: {acc:.2f}%")

    # ===================== SERVER DISTILLATION =====================
    class ServerClassifier(nn.Module):
        def __init__(self, input_dim=64, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    server_classifier = ServerClassifier(args.output_dim, num_classes).to(device)
    optimizer = optim.Adam(server_classifier.parameters(), lr=args.lr)

    beta = args.beta
    gamma = args.mixture_coef
    temperature = args.temperature
    client_epochs = 50
    ensemble_epochs = 50

    def compute_confidence_weights(probs):
        return torch.max(probs, dim=1)[0]

    def confidence_weighted_distillation_loss(student_outputs, teacher_probs,
                                              confidence_weights, temperature=2.0):
        soft_student = F.log_softmax(student_outputs / temperature, dim=1)
        kl_div = F.kl_div(soft_student, teacher_probs, reduction='none').sum(dim=1)
        return (kl_div * confidence_weights).mean() * (temperature ** 2)

    # Collect all latents + teacher outputs (on GPU)
    all_latents_list, all_teacher_logits_list, pseudo_labels_list = [], [], []

    for client_id, loader in enumerate(client_loaders, start=1):
        clf = client_classifiers[client_id-1]
        latents, logits, pseudos, confs = [], [], [], []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.cuda(non_blocking=True)
                inputs = gpu_transform(inputs, device)
                latent = encoder(inputs)
                teacher_logits = clf(latent)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                pseudo = torch.argmax(teacher_probs, dim=1)
                conf = compute_confidence_weights(teacher_probs)

                latents.append(latent)
                logits.append(teacher_logits)
                pseudos.append(pseudo)
                confs.append(conf)

        client_latents = torch.cat(latents)
        client_logits = torch.cat(logits)
        client_pseudos = torch.cat(pseudos)
        client_confs = torch.cat(confs)

        # --- Differential Privacy Noise (optional) ---
        if args.dp_epsilon != float("inf"):
            client_latents = add_dp_noise(
                client_latents,
                epsilon=args.dp_epsilon,
                delta=args.dp_delta,
                clip_norm=args.dp_clip_norm,
                device=device
            )

        all_latents_list.append(client_latents)
        all_teacher_logits_list.append(client_logits)
        pseudo_labels_list.append(client_pseudos)

        client_dataset = TensorDataset(client_latents, client_logits, client_pseudos, client_confs)
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)

        # ===== Phase 1: Client → Server Distillation (with γ-mixing) =====
        for epoch in range(client_epochs):
            server_classifier.train()
            total_loss = 0.0
            for batch_latents, batch_teachers, batch_pseudo, batch_confidence in client_loader:
                optimizer.zero_grad()

                # Server student prediction
                student_outputs = server_classifier(batch_latents)

                # Teacher (client) signal
                teacher_probs = F.softmax(batch_teachers / temperature, dim=1)

                # Knowledge Mixing Eq. (6): mix current teacher with previous server
                if client_id > 1:  # not the first client
                    with torch.no_grad():
                        server_prev_probs = F.softmax(server_classifier(batch_latents) / temperature, dim=1)
                        mixed_teacher_probs = gamma * teacher_probs + (1 - gamma) * server_prev_probs
                else:
                    mixed_teacher_probs = teacher_probs

                # Soft KD loss (Eq. 7, confidence-weighted)
                kd_loss = confidence_weighted_distillation_loss(student_outputs,
                                                                mixed_teacher_probs,
                                                                batch_confidence,
                                                                temperature)
                # Hard pseudo-label loss (Eq. 2)
                hard_loss = F.cross_entropy(student_outputs, batch_pseudo)

                # Combined loss (Eq. 1)
                loss = beta * kd_loss + (1 - beta) * hard_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Client {client_id}, Epoch {epoch+1}/{client_epochs}, Loss {total_loss/len(client_loader):.4f}")
    

    # Final Ensemble Distillation
    all_latents = torch.cat(all_latents_list)
    all_teacher_logits = torch.cat(all_teacher_logits_list)
    pseudo_labels = torch.cat(pseudo_labels_list)

    with torch.no_grad():
        ensemble_preds = []
        for clf in client_classifiers:
            probs = F.softmax(clf(all_latents) / temperature, dim=1)
            ensemble_preds.append(probs)
        ensemble_probs = torch.stack(ensemble_preds).mean(dim=0)
        confidence_weights = compute_confidence_weights(ensemble_probs)

    train_dataset = TensorDataset(all_latents, ensemble_probs, pseudo_labels, confidence_weights)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(server_classifier.parameters(), lr=args.lr/2)

    for epoch in range(ensemble_epochs):
        server_classifier.train()
        total_loss = 0.0
        for batch_latents, batch_teachers, batch_pseudo, batch_confidence in train_loader:
            optimizer.zero_grad()
            student_outputs = server_classifier(batch_latents)
            kd_loss = confidence_weighted_distillation_loss(student_outputs, batch_teachers,
                                                            batch_confidence, temperature)
            hard_loss = F.cross_entropy(student_outputs, batch_pseudo)
            loss = beta * kd_loss + (1 - beta) * hard_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Ensemble Epoch {epoch+1}/{ensemble_epochs}, Loss {total_loss/len(train_loader):.4f}")

    # ===== Combined Server Model =====
    class ServerModel(nn.Module):
        def __init__(self, encoder, server_classifier, device):
            super().__init__()
            self.encoder = encoder
            self.server_classifier = server_classifier
            self.device = device
        def forward(self, x):
            latent = self.encoder(gpu_transform(x.cuda(non_blocking=True), self.device)) 
            return self.server_classifier(latent)

    server_model = ServerModel(encoder, server_classifier, device).to(device)
    server_model.eval()

    # ===== Evaluation =====
    def evaluate_model(model, loader, is_combined=False):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = to_device(inputs, device)
                labels = to_device(labels, device)

                if is_combined:
                    outputs = model(inputs)
                else:
                    latent = encoder(gpu_transform(inputs, device))
                    outputs = model(latent)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total

    for i, clf in enumerate(client_classifiers):
        acc = evaluate_model(clf, test_loader, is_combined=False)
        print(f"Client {i+1} Accuracy: {acc:.2f}%")

    server_acc = evaluate_model(server_model, test_loader, is_combined=True)
    print(f"\nServer Model Accuracy: {server_acc:.2f}%")

    # Save results
    out_path = (
        f"results/{args.mixture_coef}/"
        f"run_lr{args.lr:.0e}_ep{args.epochs}_clients{args.num_clients}"
        f"_alpha{args.alpha}_seed{args.seed}_gamma{args.mixture_coef}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_dict = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "lr": args.lr,
        "epochs": args.epochs,
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "mixture_coef": args.mixture_coef,
        "beta": args.beta,
        "seed": args.seed,
        "accuracy": server_acc,
        "dp_epsilon": args.dp_epsilon,
        "dp_delta": args.dp_delta,
        "dp_clip_norm": args.dp_clip_norm,
    }
    for k, v in list(results_dict.items()):
        if isinstance(v, float) and math.isinf(v):
            results_dict[k] = "inf"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":
    main()
