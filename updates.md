# Puzzle Diffusion

## Supporting idea
Diffusion models solve problems of recovering $x_0$ by "denoising" a noisy data $x_T$ with a chain process such that $x_0=p(x_T)\prod^T{p(x_{t-1}|x_t)}$

For images, typically $x_0$ is the image, and $x_T$ is noise sampled from $\mathcal{N}(0,1)$.

In our case we want to recover position of 2D patches of in image. So $p_\theta(pos_t | pos_{t-1},rgb)$.

---
## TODO
- [x] Add DDIM, with skippable steps 
- [ ] Test rotations
- [ ] Test missing pieces
- [ ] **
## Updates:

**01/02/2023**
- New Test idea: Reconstruct missing patches

**31/01/2023**
- Implemented DDIM sampling-> sampling speedup 
- Tested the GAT as GNN on wikiart 12x12, doesn't work well 