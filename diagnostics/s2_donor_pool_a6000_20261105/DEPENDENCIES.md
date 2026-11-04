# Runtime requirements (no installation performed)

Reuse target `envs/augseg-bs16`. The stored `augmentation_pin.json` records versions actually used
for the original CPU input hashing (torch2.11.0+cu128, torchvision0.26.0+cu128, Pillow12.2.0,
numpy2.4.4); `check-deps` requires these exact versions before real actions. Hash mismatches stop,
not silently regenerate pins. Imports additionally need SciPy, scikit-image, OpenCV and PyYAML.
These are dependency names, not a complete historical lock file. Python must support strict
`load_state_dict(assign=True)` through that PyTorch version. No dependency installs/bootstrap here.

Target current installed packages, loaded libraries, GPU driver/CUDA/cuDNN and validation behavior
remain unverified until target checks/run. `runtime.json` records actual runtime after model setup;
CUDA runtime comes from torch.version.cuda, not a W&B cudaVersion field.

Source environment selected package metadata at preparation (not a full historical lock):

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Pillow | 12.2.0 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| scikit-image | 0.26.0 |
| opencv-python | 4.13.0.92 |
| PyYAML | 6.0.3 |
