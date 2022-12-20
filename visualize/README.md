# The Visualizer module
This readme provides some info on how to use the visualizer module

## Usage
```python
from Interface import visualize
# make a visualisation to Cell object
t_end = 100
cell = Interface.simrun_simtrail_to_cell_object(mdb,sim, tStop=t_end)

# create a single cell visualizer object from the cell
cv = visualize.CellVisualizer(cell)

# Show a quick plot of the cell
cv.show_cell_2d()
        
# Show a plot with the membrane voltage
cv.show_voltage_cell_2d()

# make an interactive visualisation (may take some time to load)
cv.plot_interactive_3d(downsample_time=1)

# write out to .vtk frames for analysis in e.g. ParaView
cv.write_vtk_frames(out_name="cell", out_dir=".")
```
