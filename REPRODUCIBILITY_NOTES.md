# Reproducibility Notes

This repository is a **reference implementation reconstructed from the manuscript**, not a claim of access to the original experimental source code.

## Explicitly documented by the manuscript

- 224 × 224 input images
- Min–Max normalization
- Adam optimizer
- Initial learning rate: 0.0001
- Capsule vector dimension: 16
- Dynamic routing iterations: 3
- Batch size: 16
- Maximum training epochs: 50
- Dropout probability: 0.5
- DSOA population: 30
- DSOA maximum iterations: 100
- Learning-rate search: 0.0001–0.01
- Capsule dimension search: 8–32
- Routing-iteration search: 2–5
- α = 0.5 and β = 1.5
- Candidate values are clipped to valid bounds
- Random seed: 42

## Details not fully specified

The manuscript states that DSOA also considers batch size and selected EfficientNet fine-tuning parameters, but does not provide complete numerical bounds for those search dimensions. The manuscript text available for reconstruction also does not expose the full mathematical expression of the DSOA position-update equation. Consequently, this repository does not silently invent those missing historical details.

The reference implementation uses a configurable fine-tuning fraction and provides a DSOA optimizer whose update structure follows the manuscript's algorithmic description: exploratory random displacement plus attraction toward the current best solution, followed by boundary clipping and validation-objective evaluation.

## Important distinction

The manuscript reports 98.17% accuracy, 98.36% precision, 98.11% recall, 98.23% F1-score and 98.42% specificity. These numbers should not be described as reproduced by this repository unless the authors independently execute and verify the same experiment.
