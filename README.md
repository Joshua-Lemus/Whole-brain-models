# Whole-brain models code

## Requirements
- The Virtual Brain
- Jupyter lab
- The high-resolution network (too heavy for github)

This is how I do it: 
	
	# Install conda environment for tvb
	apt install conda
	conda env create -f tvb.yml
	
	# Get the high-resolution network
	# Need to ask Lorenzo, or get it from the university's server

## Files description
- HelloWorld.ipynb is a playground for tvb stuff and more
- Hippocampus_JR.ipynb is for simulating a WBM with the hippocampus, using the high-resolution network, with a Jansen-Riten model
- tvb.yml contains the environment description fro conda
- ViewResults.ipynb is a playground for visually inspecting the high-resolution network
- utils_py contains core libraries that Lorenzo wrote

## Resources
- [source code for tvb](https://github.com/the-virtual-brain/tvb-root)
- [slide notes I made about TVB](https://unimore365-my.sharepoint.com/:p:/g/personal/366660_unimore_it/EWbUfmZKOLVAo_Au2ZyqYYABzjTtNtYvm9t4wzD-Sf-JbA?e=oxsUwo)
