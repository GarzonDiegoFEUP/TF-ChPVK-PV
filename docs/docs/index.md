# TF-ChPVK-PV

## Description

This repository contains the datasets, analysis notebooks, and source code for the paper:

**ML-guided screening of chalcogenide perovskites as solar energy materials**

> D. A. Garzón, L. Himanen, L. Andrade, S. Sadewasser, J. A. Márquez,
> *"ML-guided screening of chalcogenide perovskites as solar energy materials"* (2026).
>
> **Preprint coming soon.**

## Abstract

Chalcogenide perovskites have emerged as promising absorber materials for next-generation photovoltaic devices,
yet their experimental realization remains limited by competing phases, structural polymorphism, and synthetic
challenges. Here, we present a fully data-driven and experimentally grounded screening and ranking framework
to assess the stability and experimental feasibility of chalcogenide perovskites, integrating interpretable analytical
descriptors, machine-learning models, and sustainability metrics.

## Pipeline Overview

The screening framework integrates:

1. **SISSO-derived tolerance factor (τ*)** — an interpretable analytical descriptor for perovskite structural stability
2. **CrystaLLM crystal structure generation** — generative prediction to validate perovskite-type topology
3. **CrabNet bandgap estimation** — composition-based prediction trained on experimental data
4. **Sustainability analysis** — multi-objective ranking using HHI, ESG scores, and supply risk metrics
5. **Experimental plausibility assessment** — crystal-likeness scoring for synthesizability likelihood

## Commands

The Makefile contains the central entry points for common tasks related to this project.

