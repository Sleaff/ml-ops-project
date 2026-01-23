# Project Checklist Status

## Week 1 (18/18 = 100%)

| Status | Item |
|--------|------|
| ✅ | Create a git repository (M5) |
| ✅ | Make sure that all team members have write access to the GitHub repository (M5) |
| ✅ | Create a dedicated environment for you project to keep track of your packages (M2) |
| ✅ | Create the initial file structure using cookiecutter with an appropriate template (M6) |
| ✅ | Fill out the data.py file such that it downloads whatever data you need and preprocesses it (M6) |
| ✅ | Add a model to model.py and a training procedure to train.py and get that running (M6) |
| ✅ | Remember to fill out the requirements.txt and requirements_dev.txt file with whatever dependencies (M2+M6) |
| ✅ | Remember to comply with good coding practices (pep8) while doing the project (M7) |
| ✅ | Do a bit of code typing and remember to document essential parts of your code (M7) |
| ✅ | Setup version control for your data or part of your data (M8) |
| ✅ | Add command line interfaces and project commands to your code where it makes sense (M9) |
| ✅ | Construct one or multiple docker files for your code (M10) |
| ✅ | Build the docker files locally and make sure they work as intended (M10) |
| ✅ | Write one or multiple configurations files for your experiments (M11) |
| ✅ | Used Hydra to load the configurations and manage your hyperparameters (M11) |
| ✅ | Use profiling to optimize your code (M12) |
| ✅ | Use logging to log important events in your code (M14) |
| ✅ | Use Weights & Biases to log training progress and other important metrics/artifacts (M14) |
| ✅ | Consider running a hyperparameter optimization sweep (M14) - *wandb sweep with Bayesian search* |
| ✅ | Use PyTorch-lightning to reduce the amount of boilerplate in your code (M15) |

## Week 2 (13/14 = 93%)

| Status | Item |
|--------|------|
| ✅ | Write unit tests related to the data part of your code (M16) |
| ✅ | Write unit tests related to model construction and or model training (M16) |
| ✅ | Calculate the code coverage (M16) |
| ✅ | Get some continuous integration running on the GitHub repository (M17) |
| ✅ | Add caching and multi-os/python/pytorch testing to your continuous integration (M17) |
| ✅ | Add a linting step to your continuous integration (M17) |
| ✅ | Add pre-commit hooks to your version control setup (M18) |
| ❌ | Add a continues workflow that triggers when data changes (M19) |
| ❌ | Add a continues workflow that triggers when changes to the model registry is made (M19) |
| ✅ | Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21) |
| ✅ | Create a trigger workflow for automatically building your docker images (M21) |
| ✅ | Get your model training in GCP using either the Engine or Vertex AI (M21) |
| ✅ | Create a FastAPI application that can do inference using your model (M22) |
| 🟡 | Deploy your model in GCP using either Functions or Run as the backend (M23) - *infra ready, not deployed* |
| ✅ | Write API tests for your application and setup continues integration for these (M24) |
| ✅ | Load test your application (M24) |
| ❌ | Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25) |
| ✅ | Create a frontend for your API (M26) - *Gradio frontend* |

## Week 3 (1/8 = 13%)

| Status | Item |
|--------|------|
| ❌ | Check how robust your model is towards data drifting (M27) |
| ❌ | Deploy to the cloud a drift detection API (M27) |
| ❌ | Instrument your API with a couple of system metrics (M28) |
| ❌ | Setup cloud monitoring of your instrumented application (M28) |
| ❌ | Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28) |
| ✅ | If applicable, optimize the performance of your data loading using distributed data loading (M29) |
| ❌ | If applicable, optimize the performance of your training pipeline by using distributed training (M30) |
| ❌ | Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31) |

## Summary

| Week | Done | Total | Percentage |
|------|------|-------|------------|
| Week 1 | 18 | 18 | 100% |
| Week 2 | 13 | 14 | 93% |
| Week 3 | 1 | 8 | 13% |
| **Total** | **32** | **40** | **80%** |

Week 1 complete. Week 2 missing only data/model change workflows and Cloud Run deployment. Week 3 (monitoring, drift, distributed training) not yet implemented.
