# Maximum Driving Force (MDF)

This code is used to calculate the maximum driving force (MDF) parameter with both experimental data (or user-inputted) and high-throughput DFT data from Materials Project. 

Major Python packages required to run the code include: NumPy, SciPy, pandas, matplotlib, ASE, and pymatgen

The Jupyter notebooks can be used without reading the Python code in Packaged. Short descriptions of the notebooks below:
Experimental MDF (single-element only) - computes MDF for single element systems using experimental data derived from Pourbaix's Atlas Of Electrochemical Equilibria In Aqueous Solutions

Multi-element MDF with pymatgen - computes MDF for multi-element (or single element) systems using data (for both ions and solids) from the Materials Project database according to compositional constraints (more about this methodology can be found here: Thompson, W. T.; Kaye, M. H.; Bale, C. W.; Pelton, A. D. Uhlig’s Corrosion Handbook, 3rd ed.; John Wiley & Sons, Ltd: Hoboken, 2011; DOI: 10.1002/9780470872864.ch8)

Composition Independent MDF - computes MDF for single or multi-element systems with either experimental or Materials Project data without composition constraints; treats each possible solid in the system equally, computing driving forces for each in comparison with all ions in the system, and selecting those with the best driving forces

