# Advanced Data Science Systems Engineering — Companion Code Repository

Code examples, datasets, configuration files, and chapter-by-chapter implementations accompanying the book:

**Advanced Data Science Systems Engineering: Building Reproducible, Scalable, and Production-Ready Systems in Python**

by **Shouke Wei, PhD**

<img width="720" height="1022" alt="advanced_data_eng_cover_front" src="https://github.com/user-attachments/assets/75341952-fd72-4808-8b2a-9b732da61b7b" />


---

## About This Repository

This repository contains the source code developed throughout the book.

The purpose of this repository is educational. Each chapter introduces new concepts, techniques, and engineering practices that progressively transform a reusable workflow into a reproducible, scalable, and maintainable data science system.

The examples are organized chapter by chapter so that readers can follow the evolution of the project throughout the book.

The final framework developed in the book is maintained separately as the **dskit** project.

This repository focuses on learning and implementation, while the dskit repository contains the production-oriented framework.

---

## Companion Resources

### Book Website

Companion resources, downloadable files, updates, and supplementary materials:

https://press.deepsim.ca/data-engineering

### Final Framework Repository

The reusable framework developed throughout the book:

https://github.com/shoukewei/deepsim-dskit

### PyPI Package

```bash
pip install dskit
```

---

## Repository Structure

```text
advanced-data-science-systems-engineering/
│
├── chapter00/
├── chapter01/
├── chapter02/
├── chapter03/
├── chapter04/
├── chapter05/
├── chapter06/
├── chapter07/
├── chapter08/
├── chapter09/
├── chapter10/
  ...
  ├── configs/
  ├── notebooks/
  └── README.md
```

Each chapter directory contains the code relevant to that chapter.

Examples may include:

* Python scripts
* Jupyter notebooks
* configuration files
* datasets
* testing examples
* workflow artifacts

---

## Prerequisites

Recommended:

* Python 3.11+
* Basic data science experience
* Familiarity with pandas and scikit-learn
* Completion of *Practical Data Science Engineering* (helpful but not required)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shoukewei/advanced-data-science-systems-engineering.git

cd advanced-data-science-systems-engineering
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Examples

Each chapter contains independent examples.

For example:

```bash
cd chapter05

python logging_example.py
```

or

```bash
jupyter notebook
```

and open the notebook provided in the chapter directory.

---

## Relationship to dskit

This repository demonstrates the incremental development process presented throughout the book.

The production-oriented framework developed from these examples is maintained separately as:

```text
dskit
```

Readers interested in the final framework should visit:

https://github.com/shoukewei/dskit

---

## Reporting Issues

If you discover errors, typos, or issues with the code examples, please open an issue in this repository.

Constructive feedback and contributions are welcome.

---

## Citation

If you use this repository in research, teaching, or professional work, please cite:

**Wei, S. (2026). Advanced Data Science Systems Engineering: Building Reproducible, Scalable, and Production-Ready Systems in Python. Deepsim Press.**

---

## License

This repository is provided for educational purposes.

Refer to the LICENSE file for details.

---

## About the Author

**Shouke Wei, PhD** is a researcher, scientist, and entrepreneur specializing in data analysis and modeling, wavelet-based signal processing, and AI-driven applications.

He earned his Ph.D. from Brandenburg University of Technology Cottbus–Senftenberg (Germany), conducted postdoctoral research at Eawag (Switzerland), held research positions at the University of British Columbia (Canada), and served as a distinguished and adjunct professor at multiple universities in China.

His work focuses on practical, reusable, and engineering-oriented approaches to data science, machine learning systems, and AI applications.
