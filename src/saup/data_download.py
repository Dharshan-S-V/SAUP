"""Dataset download, validation, and extraction for UCI Adult, Bank Marketing, and Wine Quality."""

import hashlib
import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

import requests

from saup.config import resolve_path

logger = logging.getLogger(__name__)

UCI_ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
UCI_BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_WINE_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

ADULT_REQUIRED_FILES = ["adult.data", "adult.test", "adult.names"]
BANK_REQUIRED_FILES = ["bank-full.csv", "bank-names.txt"]
WINE_REQUIRED_FILES = ["winequality-red.csv", "winequality-white.csv"]

DOWNLOAD_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_zip(filepath: Path) -> bool:
    """Validate that a file is a valid ZIP archive.
    
    Args:
        filepath: Path to the ZIP file.
        
    Returns:
        True if valid ZIP.
    """
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                logger.error("Corrupted file in ZIP: %s", bad)
                return False
        return True
    except zipfile.BadZipFile:
        logger.error("File is not a valid ZIP archive: %s", filepath)
        return False


def _is_path_safe(name: str, target_dir: Path) -> bool:
    """Check for path traversal attacks in archive entries.
    
    Args:
        name: Archive entry name.
        target_dir: Target extraction directory.
        
    Returns:
        True if the path is safe.
    """
    resolved = (target_dir / name).resolve()
    return str(resolved).startswith(str(target_dir.resolve()))


def _download_file(url: str, target: Path) -> None:
    """Download a file with retries.
    
    Args:
        url: URL to download.
        target: Local path to save.
        
    Raises:
        RuntimeError: If download fails after retries.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Download attempt %d/%d from %s", attempt, MAX_RETRIES, url)
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(target, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            if total_size > 0 and downloaded != total_size:
                raise RuntimeError(
                    f"Incomplete download: {downloaded}/{total_size} bytes"
                )
            
            logger.info("Downloaded %d bytes to %s", downloaded, target)
            return
            
        except (requests.RequestException, RuntimeError) as e:
            last_error = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            if target.exists():
                target.unlink()
    
    raise RuntimeError(
        f"Failed to download after {MAX_RETRIES} attempts. Last error: {last_error}. "
        f"You can manually download the file from {url} and provide it via "
        f"--zip-path argument."
    )


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Safely extract a ZIP archive.
    
    Args:
        zip_path: Path to the ZIP file.
        target_dir: Directory to extract to.
        
    Raises:
        RuntimeError: If path traversal is detected.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not _is_path_safe(info.filename, target_dir):
                raise RuntimeError(
                    f"Path traversal detected in archive: {info.filename}"
                )
        zf.extractall(target_dir)
    logger.info("Extracted ZIP to %s", target_dir)


def download_dataset(
    raw_dir: str = "data/raw",
    zip_path: Optional[str] = None,
    url: str = UCI_ADULT_URL,
) -> Path:
    """Download or locate the UCI Adult dataset (kept for backward compatibility)."""
    return download_adult_dataset(raw_dir=raw_dir, zip_path=zip_path, url=url)


def download_adult_dataset(
    raw_dir: str = "data/raw",
    zip_path: Optional[str] = None,
    url: str = UCI_ADULT_URL,
) -> Path:
    """Download or locate the UCI Adult dataset.
    
    Args:
        raw_dir: Directory for raw data files.
        zip_path: Optional path to a pre-downloaded ZIP file.
        url: URL to download from if ZIP not available locally.
        
    Returns:
        Path to the raw data directory containing extracted files.
    """
    raw_path = resolve_path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Check if all required files already exist
    all_exist = all((raw_path / f).exists() for f in ADULT_REQUIRED_FILES)
    if all_exist:
        logger.info("All required Adult dataset files already exist in %s", raw_path)
        return raw_path
    
    target_zip = raw_path / "adult.zip"
    
    if zip_path:
        supplied = Path(zip_path)
        if not supplied.exists():
            raise FileNotFoundError(f"Supplied ZIP not found: {supplied}")
        logger.info("Using supplied ZIP: %s", supplied)
        if supplied != target_zip:
            import shutil
            shutil.copy2(supplied, target_zip)
    elif target_zip.exists():
        logger.info("Found existing ZIP: %s", target_zip)
    else:
        logger.info("Downloading Adult dataset from %s", url)
        _download_file(url, target_zip)
    
    if not validate_zip(target_zip):
        raise RuntimeError(f"Invalid ZIP archive: {target_zip}")
    
    checksum = compute_sha256(target_zip)
    logger.info("Adult ZIP SHA-256: %s", checksum)
    (raw_path / "adult.zip.sha256").write_text(checksum, encoding="utf-8")
    
    _extract_zip(target_zip, raw_path)
    
    # Check and locate missing files if in subdirectories
    missing = [f for f in ADULT_REQUIRED_FILES if not (raw_path / f).exists()]
    if missing:
        import shutil
        for f in missing:
            found = list(raw_path.rglob(f))
            if found:
                shutil.copy2(found[0], raw_path / f)
                logger.info("Found %s at %s, copied to %s", f, found[0], raw_path / f)
        still_missing = [f for f in ADULT_REQUIRED_FILES if not (raw_path / f).exists()]
        if still_missing:
            raise RuntimeError(f"Missing required Adult files: {still_missing}")
    
    logger.info("Adult dataset ready in %s", raw_path)
    return raw_path


def download_bank_dataset(
    raw_dir: str = "data/raw/bank",
    zip_path: Optional[str] = None,
    url: str = UCI_BANK_URL,
) -> Path:
    """Download, validate, and extract the UCI Bank Marketing dataset.
    
    Extracts bank.zip (containing bank-full.csv and bank-names.txt).
    
    Args:
        raw_dir: Directory for raw Bank Marketing data files.
        zip_path: Optional path to a pre-downloaded ZIP file.
        url: URL to download from.
        
    Returns:
        Path to the directory containing extracted files.
    """
    raw_path = resolve_path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    all_exist = all((raw_path / f).exists() for f in BANK_REQUIRED_FILES)
    if all_exist:
        logger.info("All required Bank Marketing files already exist in %s", raw_path)
        return raw_path
    
    target_zip = raw_path / "bank_marketing.zip"
    
    if zip_path:
        supplied = Path(zip_path)
        if not supplied.exists():
            raise FileNotFoundError(f"Supplied ZIP not found: {supplied}")
        if supplied != target_zip:
            import shutil
            shutil.copy2(supplied, target_zip)
    elif target_zip.exists():
        logger.info("Found existing Bank Marketing ZIP: %s", target_zip)
    else:
        logger.info("Downloading Bank Marketing dataset from %s", url)
        _download_file(url, target_zip)
    
    if not validate_zip(target_zip):
        raise RuntimeError(f"Invalid ZIP archive: {target_zip}")
    
    checksum = compute_sha256(target_zip)
    logger.info("Bank Marketing ZIP SHA-256: %s", checksum)
    (raw_path / "bank_marketing.zip.sha256").write_text(checksum, encoding="utf-8")
    
    # Extract outer zip into raw_path
    _extract_zip(target_zip, raw_path)
    
    # Outer zip contains bank.zip and bank-additional.zip
    # Extract bank.zip to get bank-full.csv and bank-names.txt
    inner_bank_zip = raw_path / "bank.zip"
    if inner_bank_zip.exists():
        if validate_zip(inner_bank_zip):
            _extract_zip(inner_bank_zip, raw_path)
        else:
            raise RuntimeError(f"Corrupted inner zip: {inner_bank_zip}")
    
    # Verify required files
    missing = [f for f in BANK_REQUIRED_FILES if not (raw_path / f).exists()]
    if missing:
        import shutil
        for f in missing:
            found = list(raw_path.rglob(f))
            if found:
                shutil.copy2(found[0], raw_path / f)
        still_missing = [f for f in BANK_REQUIRED_FILES if not (raw_path / f).exists()]
        if still_missing:
            raise RuntimeError(f"Missing required Bank Marketing files: {still_missing}")
    
    logger.info("Bank Marketing dataset ready in %s", raw_path)
    return raw_path


def download_wine_dataset(
    raw_dir: str = "data/raw/wine",
    zip_path: Optional[str] = None,
    url: str = UCI_WINE_URL,
) -> Path:
    """Download, validate, and extract the UCI Wine Quality dataset.
    
    Extracts winequality-red.csv, winequality-white.csv, and winequality.names.
    
    Args:
        raw_dir: Directory for raw Wine Quality data files.
        zip_path: Optional path to a pre-downloaded ZIP file.
        url: URL to download from.
        
    Returns:
        Path to the directory containing extracted files.
    """
    raw_path = resolve_path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    all_exist = all((raw_path / f).exists() for f in WINE_REQUIRED_FILES)
    if all_exist:
        logger.info("All required Wine Quality files already exist in %s", raw_path)
        return raw_path
    
    target_zip = raw_path / "wine_quality.zip"
    
    if zip_path:
        supplied = Path(zip_path)
        if not supplied.exists():
            raise FileNotFoundError(f"Supplied ZIP not found: {supplied}")
        if supplied != target_zip:
            import shutil
            shutil.copy2(supplied, target_zip)
    elif target_zip.exists():
        logger.info("Found existing Wine Quality ZIP: %s", target_zip)
    else:
        logger.info("Downloading Wine Quality dataset from %s", url)
        _download_file(url, target_zip)
    
    if not validate_zip(target_zip):
        raise RuntimeError(f"Invalid ZIP archive: {target_zip}")
    
    checksum = compute_sha256(target_zip)
    logger.info("Wine Quality ZIP SHA-256: %s", checksum)
    (raw_path / "wine_quality.zip.sha256").write_text(checksum, encoding="utf-8")
    
    _extract_zip(target_zip, raw_path)
    
    missing = [f for f in WINE_REQUIRED_FILES if not (raw_path / f).exists()]
    if missing:
        import shutil
        for f in missing:
            found = list(raw_path.rglob(f))
            if found:
                shutil.copy2(found[0], raw_path / f)
        still_missing = [f for f in WINE_REQUIRED_FILES if not (raw_path / f).exists()]
        if still_missing:
            raise RuntimeError(f"Missing required Wine Quality files: {still_missing}")
    
    logger.info("Wine Quality dataset ready in %s", raw_path)
    return raw_path


def download_dataset_by_name(
    dataset_name: str,
    raw_dir: Optional[str] = None,
    zip_path: Optional[str] = None,
) -> Path:
    """Download dataset by name ('adult', 'bank_marketing' / 'bank', 'wine_quality' / 'wine').
    
    Args:
        dataset_name: Name of dataset.
        raw_dir: Optional custom raw directory.
        zip_path: Optional local zip path.
        
    Returns:
        Path to raw directory with extracted files.
    """
    name = dataset_name.lower()
    if name in ("adult", "uci_adult"):
        return download_adult_dataset(raw_dir=raw_dir or "data/raw", zip_path=zip_path)
    elif name in ("bank", "bank_marketing", "uci_bank"):
        return download_bank_dataset(raw_dir=raw_dir or "data/raw/bank", zip_path=zip_path)
    elif name in ("wine", "wine_quality", "uci_wine"):
        return download_wine_dataset(raw_dir=raw_dir or "data/raw/wine", zip_path=zip_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'adult', 'bank', or 'wine'.")
