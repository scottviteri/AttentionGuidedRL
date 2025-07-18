# Mathematical Formulation Documents

This folder contains the mathematical formulation of the AttentionGuidedRL optimization objective in LaTeX format.

## Files

- `mathematical_formulation.tex` - Complete LaTeX document with mathematical formulation
- `Makefile` - Build automation for the LaTeX document
- `README.md` - This file

## Compilation Instructions

### Prerequisites

You need a LaTeX distribution installed:

- **macOS**: Install MacTeX from https://www.tug.org/mactex/
- **Linux**: Install TeX Live: `sudo apt-get install texlive-full` (Ubuntu/Debian)
- **Windows**: Install MiKTeX from https://miktex.org/

### Building the PDF

1. **Using Make (recommended)**:
   ```bash
   cd latex/
   make
   ```
   
   This will generate `mathematical_formulation.pdf`.

2. **Manual compilation**:
   ```bash
   cd latex/
   pdflatex mathematical_formulation.tex
   pdflatex mathematical_formulation.tex  # Run twice for cross-references
   ```

### Other Make targets

- `make clean` - Remove auxiliary files
- `make clean-all` - Remove all generated files including PDF
- `make rebuild` - Clean and rebuild everything
- `make check` - Verify LaTeX installation
- `make open` - Open PDF (macOS)
- `make view` - Open PDF (Linux)

## Alternative: Markdown Version

If you prefer viewing the mathematics in Markdown format with MathJax support, see the main repository file:

`MATHEMATICAL_FORMULATION.md`

This version should render properly in VSCode with a Markdown preview extension that supports MathJax.

## VSCode Integration

To view the math properly in VSCode:

1. **For Markdown**: Install "Markdown All in One" extension
2. **For LaTeX**: Install "LaTeX Workshop" extension
3. **For PDF preview**: The generated PDF can be opened in VSCode or any PDF viewer

## Document Contents

The mathematical formulation covers:

- Problem formulation (state/action spaces)
- Policy definition with vector queries and multi-head attention
- Reward function based on conditional log probabilities
- Main optimization objective with PPO and GRPO
- Complete training algorithm
- Implementation details and mathematical properties

## Equations Summary

Key equations include:

- **Main Objective**: $\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T \gamma^{T-t} r_t \right] - \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$
- **Policy Distribution**: $\pi_\theta(k | s_t) = \frac{1}{H} \sum_{h=1}^{H} p_{t,k}^{(h)}$
- **PPO Loss**: $\mathcal{L}(\theta) = -\sum_{t=1}^T \min\left( \rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) + \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$

For complete details, compile and view the PDF or see the Markdown version. 