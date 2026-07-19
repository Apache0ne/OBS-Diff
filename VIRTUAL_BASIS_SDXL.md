# OBS-50 Virtual-Basis SDXL

This path uses the high-quality **50% unstructured OBS-zero UNet as a teacher** and converts its SDXL transformer FFNs into physically narrower dense GEGLU bases.

The first implementation is deliberately **FFN-first**:

- the zero teacher keeps the behavior discovered by OBS;
- every compact FFN is initialized from the teacher's highest-contribution neurons;
- both compact GEGLU projections are distilled against teacher input/output activations;
- sensitive layers retry at a wider basis and can remain full width;
- a second UNet-level replay stage corrects distribution shift;
- self-attention and cross-attention dimensions remain unchanged;
- the final `.pth` contains a complete smaller UNet made only from standard Diffusers modules.

The default target keeps 65% of each FFN inner width, retries sensitive FFNs at 80%, and protects layers whose local normalized reconstruction loss remains above the threshold. This is not expected to produce a 50% whole-UNet reduction. The 50% value describes the OBS-zero teacher, not the physical student width.

## Entry points

- `obs_diff_sdxl_virtual_basis_colab.py`: fresh-Colab end-to-end launcher.
- `obs_diff_sdxl_virtual_basis.py`: virtual-basis conversion, distillation, export, and comparison.
- `lib/virtual_basis_sdxl.py`: activation capture, compact FFN construction, local distillation, and UNet replay recovery.

## Final output

```text
/content/obs_diff_sdxl_virtual_basis_results/
  unets/virtual_basis_ffn/unet_virtual_basis.pth
  unets/virtual_basis_ffn/virtual_basis_manifest.json
  obs_sdxl_virtual_basis_compare.json
  obs_sdxl_virtual_basis_compare.html
```

The `.pth` is a complete UNet object. Load it with a compatible Diffusers/PyTorch environment and pass it as the `unet=` override when loading the original single-file SDXL checkpoint.
