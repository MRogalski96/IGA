# IGA
Iterative Gabor averaging (IGA) algorithm for low photon budget quantitative phase imaging in-line holographic microscopy

# Contents and codes
- main_example.m - main code with examples how to reconstruct synthetic and experimental data <br>
- AS_propagate_p.m - function for optical field propagation with angular spectrum (AS) method <br>
- IGA.m - IGA algorithm <br>
- USAFmask.m - function for generating a synthetic USAF mask <br>
- ./Data - directory from which the experimental data will be loaded (see Experimental data section) <br>

# How does it work
IGA algorithm combines (1) iterative Gerchberg-Saxton (GS) phase retrieval method from multiplexed data (input holograms collected with different defocus and/or wavelength) with (2) straightforward averaging of multiple hologram reconstructions (Gabor averaging - GA) to obtain accurate reconstructions for low signal-to-noise ratio data. See the below article for details. <br> <br>
![](https://github.com/MRogalski96/IGA/blob/main/gifs/IGAdemo1.gif) 
![](https://github.com/MRogalski96/IGA/blob/main/gifs/IGAdemo2.gif) <br> <br>
![](https://github.com/MRogalski96/IGA/blob/main/gifs/IGAdemo13.gif) <br> <br>

# Experimental data
Our experimental data may be found at: <br>
M. Rogalski, P. Arcab, L. Stanaszek, V. Micó, C. Zuo, and M. Trusiak, “Physics-driven universal twin-image removal network for digital in-line holographic microscopy - dataset,” Jun. 2023, doi: 10.5281/ZENODO.8059636. <br>
https://zenodo.org/record/8059636

# Cite as
M. Rogalski, P. Arcab, E. Wdowiak, J. Á. Picazo-Bueno, V. Micó, M. Józwik, and M. Trusiak, “Hybrid iterating-averaging low photon budget Gabor holographic microscopy,” Submitted 2024

# Created by
Mikołaj Rogalski, <br>
mikolaj.rogalski.dokt@pw.edu.pl <br>
Institute of Micromechanics and Photonics, <br>
Warsaw University of Technology, 02-525 Warsaw, Poland <br>
