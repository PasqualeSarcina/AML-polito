## DINOv3 setup

This project uses **DINOv3** as one of the backbones for semantic correspondence.
Therefore, we load the model from the **official DINOv3 repository cloned locally**
and require a **pretrained checkpoint** to be provided by the user.

Pretrained weights are **not included** in this repository.

---

### 1) Clone the DINOv3 repository

Clone the official DINOv3 repository somewhere in your project (suggested location):

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```
### DINOv3 pretrained weights

DINOv3 checkpoints require license acceptance and are distributed via **time-limited download links**.  
Please request access and download the **ViT-B/16 pretrained checkpoint**:
```
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```
After downloading, place the file for example in:
```
checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```
**Note**  
The download URLs are temporary and expire.  
For reproducibility, users must manually download the checkpoint and provide its **local path**
when running the code.