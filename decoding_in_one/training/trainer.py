import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import OptimConfig
from .optimizers import Lion, build_cosine_scheduler, build_warmup_then_decay_scheduler


class Trainer:
    """Generic PyTorch trainer for local experiments."""

    def __init__(self, model: nn.Module, config: OptimConfig):
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.model.to(self.device)

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_batch_size_for_epoch(self, epoch_index: int) -> int:
        if not self.config.batch_schedule_enabled:
            return int(self.config.batch_size)

        start_epoch = int(self.config.batch_schedule_start_epoch)
        end_epoch = int(self.config.batch_schedule_end_epoch)
        initial = int(self.config.batch_size_initial)
        final = int(self.config.batch_size_final)

        if epoch_index <= start_epoch:
            return initial
        if epoch_index > end_epoch:
            return final

        progress = (epoch_index - start_epoch) / max(1, end_epoch - start_epoch)
        raw_batch_size = initial + (final - initial) * progress
        batch_size = int(round(raw_batch_size / 8) * 8)
        return max(min(batch_size, final), initial)

    def _get_accumulate_steps(self, epoch_index: int) -> int:
        if self.config.accumulate_steps <= 1:
            return 1
        if not self.config.batch_schedule_enabled:
            return int(self.config.accumulate_steps)

        current_batch_size = self._get_batch_size_for_epoch(epoch_index)
        base_batch_size = int(self.config.batch_size_initial)
        unbounded_accumulate = current_batch_size // max(base_batch_size, 1)
        return min(max(unbounded_accumulate, 1), int(self.config.accumulate_steps))

    def _clone_dataloader(self, loader: DataLoader, *, batch_size: int, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": loader.dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
        }
        if loader.num_workers > 0:
            kwargs["persistent_workers"] = loader.persistent_workers
            if loader.prefetch_factor is not None:
                kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(**kwargs)

    def _build_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_type = self.config.optimizer_type.strip()
        if optimizer_type == "AdamW":
            return optim.AdamW(
                trainable_params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, self.config.beta2),
                eps=1e-5,
            )
        if optimizer_type == "Lion":
            return Lion(
                trainable_params,
                lr=self.config.lr,
                betas=(0.9, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")

    def _estimate_total_steps(self, train_loader: DataLoader) -> int:
        dataset_size = len(train_loader.dataset)
        total_steps = 0
        for epoch_index in range(self.config.epochs):
            batch_size = self._get_batch_size_for_epoch(epoch_index)
            accumulate_steps = self._get_accumulate_steps(epoch_index)
            num_batches = math.ceil(dataset_size / batch_size)
            total_steps += max(1, math.ceil(num_batches / accumulate_steps))
        return total_steps

    def _build_scheduler(self, optimizer, total_steps: int):
        scheduler_type = self.config.lr_scheduler_type.strip().lower()
        if scheduler_type == "warmup_then_decay":
            return build_warmup_then_decay_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=int(self.config.warmup_steps),
                milestones=list(self.config.lr_milestones),
                gamma=float(self.config.lr_gamma),
            )
        if scheduler_type == "cosine":
            return build_cosine_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=int(self.config.warmup_steps),
                base_lr=float(self.config.lr),
                min_lr=float(self.config.min_lr),
            )
        raise ValueError(f"Unsupported lr_scheduler_type: {self.config.lr_scheduler_type}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
    ) -> dict[str, Any]:
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = self._build_optimizer()
        total_steps = self._estimate_total_steps(train_loader)
        scheduler = self._build_scheduler(optimizer, total_steps)

        if hasattr(self, "run_dir"):
            run_dir = Path(self.run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(self.config.out_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, float]] = []
        iter_history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_ckpt = run_dir / "best_model.pt"

        print(f"[Train] Device: {self.device}")
        print(f"[Train] Epochs: {self.config.epochs}, Total optimizer steps: {total_steps}")
        print(f"[Train] Output dir: {run_dir}")

        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, self.config.epochs + 1):
            epoch_index = epoch - 1
            epoch_batch_size = self._get_batch_size_for_epoch(epoch_index)
            epoch_accumulate_steps = self._get_accumulate_steps(epoch_index)
            epoch_train_loader = self._clone_dataloader(
                train_loader,
                batch_size=epoch_batch_size,
                shuffle=True,
            )

            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            running_correct = 0.0
            seen = 0

            pbar = tqdm(epoch_train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
            optimizer_steps_this_epoch = 0
            for batch_idx, (x, y) in enumerate(pbar):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = criterion(logits, y)
                scaled_loss = loss / epoch_accumulate_steps
                scaled_loss.backward()

                batch_loss = float(loss.item())
                running_loss += batch_loss * x.size(0)
                seen += x.size(0)

                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    batch_correct = (preds == y).float().mean().item()
                    running_correct += batch_correct * x.size(0)

                should_step = ((batch_idx + 1) % epoch_accumulate_steps == 0) or (
                    batch_idx + 1 == len(epoch_train_loader)
                )
                if should_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps_this_epoch += 1
                    global_step += 1

                iter_history.append(
                    {
                        "epoch": float(epoch),
                        "batch": float(batch_idx + 1),
                        "global_step": float(global_step),
                        "loss": batch_loss,
                        "accuracy": batch_correct,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                pbar.set_postfix(
                    {
                        "loss": f"{batch_loss:.6f}",
                        "acc": f"{batch_correct:.4f}",
                        "bs": epoch_batch_size,
                        "accum": epoch_accumulate_steps,
                    }
                )

            train_loss = running_loss / max(seen, 1)
            train_acc = running_correct / max(seen, 1)

            epoch_val_loader = self._clone_dataloader(
                val_loader,
                batch_size=int(self.config.val_batch_size),
                shuffle=False,
            )
            val_loss, val_acc = self._evaluate(epoch_val_loader, criterion)

            epoch_time = time.time() - epoch_start
            eta = epoch_time * (self.config.epochs - epoch)
            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "time": float(epoch_time),
                "batch_size": float(epoch_batch_size),
                "accumulate_steps": float(epoch_accumulate_steps),
                "optimizer_steps": float(optimizer_steps_this_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)

            print(
                f"[Train] Epoch {epoch}/{self.config.epochs} | "
                f"train_loss: {train_loss:.6f} train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.6f} val_acc: {val_acc:.4f} | "
                f"bs: {epoch_batch_size} accum: {epoch_accumulate_steps} | "
                f"lr: {optimizer.param_groups[0]['lr']:.6g} | "
                f"time: {epoch_time:.1f}s | ETA: {self._format_time(eta)}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": {
                            "optimizer_type": self.config.optimizer_type,
                            "lr": self.config.lr,
                            "weight_decay": self.config.weight_decay,
                            "beta2": self.config.beta2,
                        },
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_ckpt,
                )
                print(f"[Train] New best model saved (val_loss: {val_loss:.6f})")

        (run_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (run_dir / "iter_history.json").write_text(
            json.dumps(iter_history, indent=2),
            encoding="utf-8",
        )

        final_report = {
            "run_dir": str(run_dir),
            "config": {
                "epochs": self.config.epochs,
                "optimizer_type": self.config.optimizer_type,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "beta2": self.config.beta2,
                "accumulate_steps": self.config.accumulate_steps,
                "lr_scheduler_type": self.config.lr_scheduler_type,
                "warmup_steps": self.config.warmup_steps,
                "lr_milestones": list(self.config.lr_milestones),
                "lr_gamma": self.config.lr_gamma,
                "min_lr": self.config.min_lr,
            },
            "device": str(self.device),
            "best_val_loss": float(best_val_loss),
            "best_checkpoint": str(best_ckpt),
            "history": history,
            "iter_count": len(iter_history),
        }

        (run_dir / "full_report.json").write_text(
            json.dumps(final_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_report

    def _evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total = 0
        sample_channels = None

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                total_correct += (preds == y).float().sum().item()
                total += y.numel()
                sample_channels = y.shape[1] if y.ndim >= 2 else 1

        denom = max(total // max(sample_channels or 1, 1), 1)
        return total_loss / denom, total_correct / max(total, 1)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
