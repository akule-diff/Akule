# Licensing and third-party code

Akule retains the research codebase's established MIT license in the root `LICENSE`, including the original MADiff copyright notice. This anonymous artifact account represents the research release, not an individual author.

| Component | Distribution | License / source |
|---|---|---|
| MMD | Pinned runtime source, model metadata and required model/data assets | [MIT](../external/mmd/LICENSE), [upstream](https://github.com/yoraish/mmd), revision `5c2597eb3e2def9f0a87d7972811d340343c0738` |
| MPD | Through MMD | MIT; copyright notice is included in MMD's license |
| Torch Robotics | MMD dependency source | [License](../external/mmd/deps/torch_robotics/LICENSE) |
| Motion Planning Baselines | MMD dependency source | [License](../external/mmd/deps/motion_planning_baselines/LICENSE) |
| Experiment Launcher | MMD dependency source | [License](../external/mmd/deps/experiment_launcher/LICENSE) |
| SMD | Native geometry and evaluation contract; no SMD baseline model is redistributed | [Upstream](https://github.com/RAISELab-atUVA/Diffusion-MRMP), MIT, revision `c87fc76044b350a37fcea7afc468c13c8371a237`; retained notice in `third_party/SMD_LICENSE` |
| RVO2 | Optional separately installed offline guide generator | [Upstream](https://github.com/snape/RVO2); retain its license when installing |

The vendored MMD dataset lookup is made independent of an upstream Git checkout by resolving assets relative to the runtime file. Algorithmic adapters and the separate native repair paths are in the Akule source. All upstream copyright notices are retained. Dependencies installed through Python keep their own licenses.

The Akule branding, paper figures, and recorded demo media accompany the anonymous paper and are attributed to the anonymous Akule research release. Paper figures are provided under the manuscript's CC BY 4.0 terms. See the paper for scientific attribution.
