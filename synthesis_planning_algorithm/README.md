# Synthesis planning algorithm

## Overview

Synthesis is a vital component of computational materials discovery. While high-throughput computation accelerates the identification of new ‘stable’ materials with functional properties, the actual realization of these materials is limited by their synthesis. This synthesis planning algorithm offer a physics motivated way to determine optimal synthesis recipes.

Our algorithm introduces a conceptual description of the convex hull to navigate optimal reaction pathways. The overarching principle is to identify precursors that save substantial reaction energy for the process from competing phases to target products, while avoiding low-energy geometrical subjects in the convex hull that may represent impurities or decomposition byproducts. 

Based on our algorithms, over 2000 recipes are high-throughput generated for potential high-component battery cathodes and electrolytes, such as Li/Na/K-based 4-component phosphates, borates, and redox-active/non-active transition metal oxides. We validate our theoretical framework with an automated robotic laboratory.

## Citation

This synthesis planning algorithm was created by Jiadong Chen, Wenhao Sun in University of Michigan. By using this synthesis planning algorithm, I am citing the following publication:

**Chen, J., Cross, S. R., Miara, L. J., Cho, J. J., Wang, Y., & Sun, W. (2024). Navigating phase diagram complexity to guide robotic inorganic materials synthesis. *Nature Synthesis*, 1-9.**
