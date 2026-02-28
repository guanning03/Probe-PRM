#!/usr/bin/env python3
"""
Async HuggingFace checkpoint sync monitor.

Polls latest_checkpointed_iteration.txt and uploads each new checkpoint
to a HuggingFace repository via background threads. Does NOT block training.

After training ends (SIGUSR1), uploads the final checkpoint then deletes
large local files (.pt, .safetensors, .bin) keeping only config/json files.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="[HF-Sync %(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hf_sync")

LARGE_FILE_EXTENSIONS = {".pt", ".safetensors", ".bin"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--hf_username", required=True)
    p.add_argument("--project_name", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--poll_interval", type=int, default=30)
    return p.parse_args()


def read_latest_step(experiment_dir: str):
    """Read latest_checkpointed_iteration.txt. Returns None if not ready."""
    tracker = os.path.join(experiment_dir, "latest_checkpointed_iteration.txt")
    if not os.path.exists(tracker):
        return None
    try:
        with open(tracker, "r") as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return None


def upload_checkpoint(api, repo_id, experiment_name, experiment_dir, step):
    """Upload a single checkpoint folder to HF."""
    local_folder = os.path.join(experiment_dir, f"global_step_{step}")
    hf_path = f"{experiment_name}/global_step_{step}"

    if not os.path.isdir(local_folder):
        logger.warning(f"Step {step}: folder {local_folder} not found, skipping")
        return step

    logger.info(f"Step {step}: uploading {local_folder} -> {repo_id}/{hf_path}")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_folder,
            path_in_repo=hf_path,
            repo_type="model",
            commit_message=f"checkpoint global_step_{step}",
        )
        logger.info(f"Step {step}: upload complete")
    except Exception as e:
        logger.error(f"Step {step}: upload failed: {e}")
    return step


def cleanup_large_files(experiment_dir: str, step: int):
    """Delete .pt/.safetensors/.bin from the final checkpoint, keep configs."""
    ckpt_dir = Path(experiment_dir) / f"global_step_{step}"
    if not ckpt_dir.exists():
        return
    deleted = 0
    freed = 0
    for f in ckpt_dir.rglob("*"):
        if f.is_file() and f.suffix in LARGE_FILE_EXTENSIONS:
            freed += f.stat().st_size
            f.unlink()
            deleted += 1
    logger.info(f"Cleanup step {step}: removed {deleted} files, freed {freed / (1024**3):.2f} GB")


def main():
    args = parse_args()
    api = HfApi()
    repo_id = f"{args.hf_username}/{args.project_name}"

    # Create repo if needed
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        logger.info(f"HF repo ready: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repo {repo_id}: {e}")
        sys.exit(1)

    uploaded_steps = set()
    pending_futures = {}
    training_done = threading.Event()

    def on_sigusr1(signum, frame):
        logger.info("Received SIGUSR1 - training has ended")
        training_done.set()

    signal.signal(signal.SIGUSR1, on_sigusr1)

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf_upload")
    logger.info(f"Monitoring: {args.experiment_dir}")
    logger.info(f"Uploading to: {repo_id} / {args.experiment_name}/")

    try:
        while True:
            # Collect completed uploads
            done = [s for s, fut in pending_futures.items() if fut.done()]
            for s in done:
                try:
                    pending_futures[s].result()
                except Exception as e:
                    logger.error(f"Step {s} upload error: {e}")
                del pending_futures[s]
                uploaded_steps.add(s)

            # Check for new checkpoint
            current_step = read_latest_step(args.experiment_dir)
            if current_step is not None and current_step not in uploaded_steps and current_step not in pending_futures:
                logger.info(f"New checkpoint detected: global_step_{current_step}")
                fut = executor.submit(upload_checkpoint, api, repo_id, args.experiment_name, args.experiment_dir, current_step)
                pending_futures[current_step] = fut

            # Training done -> drain and cleanup
            if training_done.is_set():
                # Final check
                final_step = read_latest_step(args.experiment_dir)
                if final_step is not None and final_step not in uploaded_steps and final_step not in pending_futures:
                    logger.info(f"Final checkpoint: global_step_{final_step}")
                    fut = executor.submit(upload_checkpoint, api, repo_id, args.experiment_name, args.experiment_dir, final_step)
                    pending_futures[final_step] = fut

                # Wait for all uploads
                logger.info(f"Waiting for {len(pending_futures)} pending upload(s)...")
                last_step = final_step
                for s, fut in pending_futures.items():
                    try:
                        fut.result(timeout=3600)
                        uploaded_steps.add(s)
                    except Exception as e:
                        logger.error(f"Step {s} final upload failed: {e}")

                # Cleanup large files from last checkpoint only
                if last_step is not None and last_step in uploaded_steps:
                    logger.info(f"Cleaning up large files for final step {last_step}")
                    cleanup_large_files(args.experiment_dir, last_step)

                logger.info("All done. Exiting.")
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
