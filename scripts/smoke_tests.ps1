Write-Host "=================================================="
Write-Host "SMOKE TESTS: 100K PIPELINE"
Write-Host "=================================================="

python -u scripts/train_100k.py --quality HQ --noise_level 0 --dry_run --num_workers 0
python -u scripts/train_100k.py --quality HQ --noise_level 20 --dry_run --num_workers 0
python -u scripts/train_100k.py --quality LQ --noise_level 0 --dry_run --num_workers 0
python -u scripts/train_100k.py --quality LQ --noise_level 20 --dry_run --num_workers 0

Write-Host "=================================================="
Write-Host "SMOKE TESTS COMPLETE"
Write-Host "=================================================="
