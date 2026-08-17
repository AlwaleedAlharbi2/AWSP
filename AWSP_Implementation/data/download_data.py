"""
data/download_data.py - Dataset Download Script for AWSP++
Run from the project root: python data/download_data.py

This script provides:
  1. Instructions for downloading all three real traces
  2. Auto-download for the Bitbrains trace (publicly available zip)
  3. Synthetic-trace generation as a fallback
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import download_instructions, generate_synthetic_trace, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("download")


def download_bitbrains(target_dir: str) -> None:
    """
    Download Bitbrains GWA-T-12 trace.
    URL: https://atlarge-research.com/gwa-t-12/
    """
    import zipfile
    try:
        import requests
    except ImportError:
        logger.error("Install requests: pip install requests")
        return

    URL = "https://gwa.ewi.tudelft.nl/datasets/GWA-T-12-Bitbrains/GWA-T-12-BitBrains.zip"
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "GWA-T-12.zip")

    logger.info("Downloading Bitbrains GWA-T-12 from %s ...", URL)
    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        print()
        logger.info("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        os.remove(zip_path)
        logger.info("Bitbrains trace extracted to %s", target_dir)
    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info("Manual download: https://atlarge-research.com/gwa-t-12/")


def generate_all_synthetic(n_vms: int = 50_000, n_hosts: int = 500) -> None:
    """Generate and cache all three synthetic traces."""
    from config import DATA_DIR
    import pickle

    for dataset_name in ("google", "azure", "bitbrains"):
        logger.info("Generating synthetic %s trace ...", dataset_name)
        vms, hosts = load_dataset(dataset_name, n_vms=n_vms, n_hosts=n_hosts,
                                   use_synthetic=True)
        cache_path = os.path.join(DATA_DIR, f"synthetic_{dataset_name}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump({"vms": vms, "hosts": hosts}, f)
        logger.info("Saved %d VMs, %d hosts -> %s", len(vms), len(hosts), cache_path)


def main(args):
    print(download_instructions())

    if args.bitbrains:
        from config import DATA_DIR
        download_bitbrains(os.path.join(DATA_DIR, "bitbrains"))

    if args.synthetic:
        generate_all_synthetic(n_vms=args.n_vms, n_hosts=args.n_hosts)
        logger.info("Synthetic traces ready. Run: python train.py --use_synthetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate AWSP++ datasets")
    parser.add_argument("--bitbrains", action="store_true",
                        help="Auto-download Bitbrains GWA-T-12 trace")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Generate synthetic traces for all datasets (default: True)")
    parser.add_argument("--n_vms", type=int, default=50_000)
    parser.add_argument("--n_hosts", type=int, default=500)
    args = parser.parse_args()
    main(args)
