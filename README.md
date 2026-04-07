# CS 251/252 Data Analysis and Visualization

This repository contains my project, lab, and notebook work from Colby College's Spring 2026 data analysis and visualization course sequence (`CS 251/252`). The course materials progress from structured data ingestion and exploratory visualization into regression, clustering, image segmentation, data transformations, and principal component analysis. This repo reflects that progression through implementation-focused Python assignments and Jupyter notebooks.

## Technical Highlights

- Built a custom `Data` abstraction for reading typed CSV files, handling missing values, encoding categorical variables, and selecting column subsets by header.
- Implemented vectorized statistical analysis utilities in NumPy, including min/max/range/mean/variance/standard deviation and reusable plotting helpers.
- Implemented least-squares regression workflows, including prediction, residual analysis, `R^2`, MSE, polynomial feature generation, and QR-based solving.
- Implemented K-means clustering from scratch with randomized initialization, iterative centroid updates, inertia tracking, multi-run selection, elbow plots, and image-segmentation utilities.
- Used Jupyter notebooks to prototype analyses, validate mathematical workflows, and communicate results visually with Matplotlib.

## Repository Overview

| Path | What it contains | Notes |
| --- | --- | --- |
| `lab/` | Lab notebooks and guided exercises | Practice with Jupyter, NumPy, matrix operations, regression workflows, and later-course topics |
| `project/p1/` | Completed data ingestion and analysis utilities | Core reusable `Data` and `Analysis` classes plus tests |
| `project/p2/extension/` | Completed regression implementation and extension notebook | Linear regression, polynomial regression, QR solver, and a solver-stability extension |
| `project/p3/` | Completed clustering project | `KMeans` implementation, clustering notebooks, and image-segmentation work |
| `project/p4/` | Later-course PCA / transformation starter files | Included here for course continuity, but not the strongest example of finished work |
| `lecture/` | Course demo notebooks and templates | Reference material aligned with lecture topics |

## Completed Project Work

### Project 1: Data Ingestion and Exploratory Analysis

Primary files:
- [project/p1/data.py](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p1/data.py)
- [project/p1/analysis.py](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p1/analysis.py)

Technical work completed:
- Implemented a CSV reader that parses course-formatted header/type rows.
- Supported both numeric and categorical columns in the same dataset.
- Added missing-data handling with `np.nan` for numeric values and `"Missing"` category insertion for categorical values.
- Built column-name-to-index mappings and category-level mappings for downstream analysis.
- Implemented row/column selection utilities and dataset access helpers.
- Wrote vectorized descriptive-statistics methods without relying on disallowed high-level shortcuts.
- Built scatter plots and pair plots for exploratory analysis.
- Validated the implementation against provided test scripts in `project/p1/test*.py`.

Why this matters technically:
- It demonstrates low-level data wrangling, careful parsing, edge-case handling, and API design for reusable analysis code.

### Project 2: Linear, Multiple, and Polynomial Regression

Primary files:
- [project/p2/extension/linear_regression.py](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p2/extension/linear_regression.py)
- [project/p2/extension/linear_regression.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p2/extension/linear_regression.ipynb)
- [project/p2/extension/polynomial_regression.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p2/extension/polynomial_regression.ipynb)
- [project/p2/extension/qr_solver.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p2/extension/qr_solver.ipynb)
- [project/p2/extension/solver_stability_extension.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p2/extension/solver_stability_extension.ipynb)

Technical work completed:
- Implemented linear regression with multiple solver paths:
  - SciPy least squares
  - Normal equations
  - QR decomposition
- Implemented Gram-Schmidt-based QR decomposition and triangular solve-based coefficient recovery.
- Added prediction, residual, `R^2`, and mean squared error calculations.
- Added regression plotting that overlays fitted lines on scatter plots and pair plots.
- Implemented polynomial feature expansion and polynomial regression workflows.
- Used train/validation style thinking in notebooks to reason about fit quality and overfitting.

Extension work completed:
- Built a notebook-based extension studying ill-conditioned regression on the provided `iris.csv`.
- Created a near-duplicate predictor to simulate multicollinearity.
- Compared normal-equation and QR behavior as the condition number increased.
- Showed that fit quality can remain similar while coefficient values become numerically unstable.

Why this matters technically:
- It demonstrates applied numerical linear algebra, model diagnostics, matrix conditioning awareness, and the ability to connect implementation details with statistical interpretation.

### Project 3: K-Means Clustering and Segmentation

Primary files:
- [project/p3/kmeans.py](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p3/kmeans.py)
- [project/p3/kmeans.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p3/kmeans.ipynb)
- [project/p3/image_segmentation.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/project/p3/image_segmentation.ipynb)

Technical work completed:
- Implemented Euclidean-distance computation for point-to-point and point-to-centroid comparisons.
- Implemented randomized centroid initialization.
- Implemented iterative label assignment and centroid recomputation.
- Handled empty-cluster cases by reseeding centroids from the dataset.
- Computed inertia as an optimization objective.
- Added `cluster_batch` support to run K-means multiple times and keep the best solution.
- Implemented elbow-plot generation for model-selection heuristics.
- Added utilities to replace image pixel colors with centroid colors and segment individual clusters.

Why this matters technically:
- It demonstrates iterative optimization, vectorized nearest-centroid assignment, unsupervised learning workflow, and practical application of clustering to image data.

## Labs and Supporting Coursework

Based on the course notes and the materials in this repo, the labs and lecture notebooks covered topics such as:

- Jupyter notebook workflow and reproducible computational analysis
- NumPy indexing, broadcasting, vectorization, and matrix operations
- Simple and multiple linear regression
- Polynomial regression and overfitting
- QR decomposition and least-squares solving
- Logical indexing and reshaping for data manipulation
- K-means clustering and image-compression/segmentation ideas
- Data transformations including normalization, centering, and rotation
- Dimensionality reduction concepts leading into PCA and SVD

Representative local materials:
- [lab/lab2/Lab2a.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/lab/lab2/Lab2a.ipynb)
- [lab/lab3/Lab2b.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/lab/lab3/Lab2b.ipynb)
- [lab/lab4/Lab2c.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/lab/lab4/Lab2c.ipynb)
- [lab/lab6/Lab4a.ipynb](/Users/lllgoyour/Documents/codeProjects/CS-252/lab/lab6/Lab4a.ipynb)

## Tools and Libraries Used

- Python
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook
- Pandas (introduced in later-course PCA materials)

## What This Repository Demonstrates

From a software engineering perspective, this work demonstrates:

- Implementing numerical methods from first principles
- Writing reusable analysis classes instead of one-off scripts
- Using vectorized array operations for performance and clarity
- Handling messy real-world data concerns such as missing values and mixed column types
- Combining code, tests, and notebooks to support both correctness and communication
- Translating mathematical concepts into working analytical tools

## Course References

These materials were organized using the course website and notes pages:

- [CS 251 course site](https://cs.colby.edu/owl/Courses/CS251/S26/index.html)
- [CS 252 notes / topic schedule](https://cs.colby.edu/courses/S26/cs25x/notes_cs252.html)
