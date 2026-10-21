from tqdm import tqdm
import numpy as np
import pathlib
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
import json

from grade.diffusion.AbstractDiffusionAlgorithm import DiffusionAlgorithm
from grade.utils.logger import get_logger
from grade.utils.io import load_checkpoint, save_checkpoint

def train(diffusion_alg: DiffusionAlgorithm, train_loader: DataLoader, val_loader: DataLoader, epochs: int, model_save_dirpath: pathlib.Path):
    logger = get_logger()
    model = diffusion_alg.get_model()
    optimizer = diffusion_alg.get_optimizer()
    scheduler = diffusion_alg.get_scheduler()
    model_save_dirpath.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint, checkpoint_path = load_checkpoint(model_save_dirpath, model.device if hasattr(model, 'device') else 'cpu')
        logger.info(f"Load checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = int(checkpoint_path.stem.split("_")[0]) + 1
    except:
        start_epoch = 0
    training_loss_min = 999
    checkpoint_metric_name = getattr(diffusion_alg, "checkpoint_metric_name", None)
    checkpoint_metric_mode = getattr(diffusion_alg, "checkpoint_metric_mode", "min")
    best_checkpoint_metric = (
        float("inf") if checkpoint_metric_mode == "min" else float("-inf")
    )
    loss_component_names = getattr(
        diffusion_alg,
        "loss_component_names",
        ("X", "E", "y"),
    )
    for epoch in range(start_epoch, epochs):
        logger.info(f"epoch={epoch}")
        train_nll, train_nll_X, train_nll_E, train_nll_y = train_epoch(diffusion_alg, train_loader)
        logger.info(f'Training loss: {train_nll:.3f}')
        logger.info(f'Training loss {loss_component_names[0]}: {train_nll_X:.3f}')
        logger.info(f'Training loss {loss_component_names[1]}: {train_nll_E:.3f}')
        logger.info(f'Training loss {loss_component_names[2]}: {train_nll_y:.3f}')

        if epoch % 1 == 0:
            metrics = test_epoch(diffusion_alg, val_loader)
            logger.info(f"Testing metrics: {metrics}")

            if checkpoint_metric_name is not None:
                checkpoint_metric = metrics[checkpoint_metric_name]
                improved = (
                    checkpoint_metric < best_checkpoint_metric
                    if checkpoint_metric_mode == "min"
                    else checkpoint_metric > best_checkpoint_metric
                )
                if improved:
                    best_checkpoint_metric = checkpoint_metric
                    best_validation_path = save_checkpoint(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_nll,
                            'validation_metrics': metrics,
                        },
                        model_save_dirpath / "best_validation.pt",
                    )
                    logger.info(
                        f"Best validation {checkpoint_metric_name}="
                        f"{checkpoint_metric:.6f} at epoch {epoch}; "
                        f"saved to {best_validation_path}."
                    )

        if epoch > 10 and train_nll < training_loss_min:
            model_save_path = save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_nll
                }, model_save_dirpath)            
            logger.info(f"Model of epoch {epoch} is saved to {model_save_path}.")

        if scheduler is not None:
            scheduler.step(train_nll)

        training_loss_min = min(training_loss_min, train_nll)

def train_epoch(diffusion_alg: DiffusionAlgorithm, train_loader: DataLoader):
    model = diffusion_alg.get_model()
    optimizer = diffusion_alg.get_optimizer()
    model.train()
    
    losses = []
    losses_X = []
    losses_E = []
    losses_y = []
    pbar = tqdm(train_loader, desc="Training Iter.")
    for data in pbar:
        loss_dict = diffusion_alg.train_step(data)
        
        loss = loss_dict["loss"]
        loss_X = loss_dict["loss_X"]
        loss_E = loss_dict["loss_E"]
        loss_y = loss_dict.get("loss_y", torch.tensor(0))

        losses.append(loss.item())
        losses_X.append(loss_X.item())
        losses_E.append(loss_E.item())
        losses_y.append(loss_y.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return np.mean(losses), np.mean(losses_X), np.mean(losses_E), np.mean(losses_y)

@torch.no_grad()
def test(diffusion_alg: DiffusionAlgorithm, test_loader: DataLoader, model_load_path: pathlib.Path):
    logger = get_logger()
    # load checkpoint
    model = diffusion_alg.get_model()
    
    if model_load_path.exists():
        checkpoint, checkpoint_path = load_checkpoint(model_load_path, model.device if hasattr(model, 'device') else 'cpu')
        logger.info(f"Load checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info(f"Warning: Checkpoint file {model_load_path} not found. Using current model weights.")

    metrics = test_epoch(diffusion_alg, test_loader)
    logger.info(f"Testing metrics: {metrics}")

@torch.no_grad()
def test_epoch(diffusion_alg: DiffusionAlgorithm, test_loader: DataLoader, repeats:int = 10):
    if hasattr(diffusion_alg, "evaluation_epoch"):
        return diffusion_alg.evaluation_epoch(test_loader)

    model = diffusion_alg.get_model()
    model.eval()

    all_pred_X = []
    all_true_X = []
    all_pred_E = []
    all_true_E = []
    
    pbar = tqdm(test_loader, desc="Testing Iter.")
    for data in pbar:
        for r in range(repeats):
            pred_X_0, pred_E_0, X_0_idx_masked, E_0_idx_masked = diffusion_alg.test_step(data)

            all_pred_X.append(pred_X_0) # (N, n_node, n_node_type)
            all_true_X.append(X_0_idx_masked) # (N, n_node, n_node_type)
            all_pred_E.append(pred_E_0) # (N, n_node, n_node, n_edge_type)
            all_true_E.append(E_0_idx_masked) # (N, n_node, n_nodem n_edge_type)

    X_metrics = calculate_metrics(all_pred_X, all_true_X, "X_")
    E_metrics = calculate_metrics(all_pred_E, all_true_E, "E_")
    metrics = {**X_metrics, **E_metrics}

    return metrics

def calculate_metrics(all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor], prefix: str):
    all_outputs = torch.cat(all_outputs, dim=0)
    all_goldens = torch.cat(all_goldens, dim=0)
    val_indices = all_goldens != all_outputs.shape[-1]

    all_outputs = all_outputs[val_indices]
    all_goldens = all_goldens[val_indices]

    logits = all_outputs.cpu()
    probs = torch.softmax(logits, dim=-1).numpy()
    pred = logits.max(1)[1].numpy()
    gold = all_goldens.cpu().numpy()

    acc = accuracy_score(gold, pred)
    
    return {f"{prefix}acc": acc}

@torch.no_grad()
def generate(diffusion_alg: DiffusionAlgorithm, model_load_path: pathlib.Path):
    logger = get_logger()
    model = diffusion_alg.get_model()
    model.eval()

    if model_load_path.exists():
        checkpoint, checkpoint_path = load_checkpoint(model_load_path, model.device if hasattr(model, 'device') else 'cpu')
        logger.info(f"Load checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info(f"Warning: Checkpoint file {model_load_path} not found. Using current model weights.")
    return diffusion_alg.generate_step()

@torch.no_grad()
def estimate(diffusion_alg: DiffusionAlgorithm, estimate_loader: DataLoader, model_load_path: pathlib.Path, logps_path: pathlib.Path, rounds: int = 100):
    logger = get_logger()
    model = diffusion_alg.get_model()
    model.eval()

    if model_load_path.exists():
        checkpoint, checkpoint_path = load_checkpoint(model_load_path, model.device if hasattr(model, 'device') else 'cpu')
        logger.info(f"Load checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info(f"Warning: Checkpoint file {model_load_path} not found. Using current model weights.")

    G_logps = []
    X_logps = []
    E_logps = []
    for round in range(rounds):
        pbar = tqdm(estimate_loader, desc=f"Estimating. Round={round}")
        G_logps.append([])
        X_logps.append([])
        E_logps.append([])
        for data in pbar:
            logp_dict = diffusion_alg.estimate_step(data)
            G_logp, X_logp, E_logp = logp_dict["G_logp"], logp_dict["X_logp"], logp_dict["E_logp"]
            G_logps[-1].append(G_logp)
            X_logps[-1].append(X_logp)
            E_logps[-1].append(E_logp)
        G_logps[-1] = torch.cat(G_logps[-1], dim=0)
        X_logps[-1] = torch.cat(X_logps[-1], dim=0)
        E_logps[-1] = torch.cat(E_logps[-1], dim=0)

        G_logps_array = torch.stack(G_logps).transpose(1, 0).cpu().numpy()
        X_logps_array = torch.stack(X_logps).transpose(1, 0).cpu().numpy()
        E_logps_array = torch.stack(E_logps).transpose(1, 0).cpu().numpy()
        logger.info(f"Round={round}, saving estimated log-probability to {logps_path}")
        with open(logps_path, "w") as f:
            json.dump(
                {
                    "G_logp": G_logps_array.tolist(),
                    "X_logp": X_logps_array.tolist(),
                    "E_logp": E_logps_array.tolist()
                }, f, indent=2
            )
        logger.info(f"Round={round}, saving finish")

    G_logps = torch.stack(G_logps).transpose(1, 0).cpu().numpy()
    X_logps = torch.stack(X_logps).transpose(1, 0).cpu().numpy()
    E_logps = torch.stack(E_logps).transpose(1, 0).cpu().numpy()
    
    return G_logps, X_logps, E_logps

if __name__ == "__main__":
    pass
