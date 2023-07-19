import './vegaLiteView.html'
import { Template } from 'meteor/templating';
import React, { PropTypes } from 'react';
import { VegaLite, Signal } from 'react-vega'
import { eventSelector } from 'vega-event-selector';


class VegaViewer extends React.Component {
    constructor(props) {
        super(props);
         
        this.viewManager = props.viewManager;        
        this.data_dimensions = props.data_sources;
        this.dataManager = this.viewManager.dataManager;
        this.RBCData = this.dataManager.RBCData;

        this.externalVariable = true; // Set your external boolean variable
    }

    handleBrushSelection = (event, item, brush) => {
        console.log("brush", brush);
        let selectedIndices = [];
        if (brush && brush.dim_1 && brush.dim_2) {
            selectedIndices = this.dataManager.project_data_2(this.data_dimensions).reduce(
              (indices, datum, index) => {
                const isSelected = datum.dim_1 >= brush.dim_1[0] &&
                datum.dim_1 <= brush.dim_1[1] &&
                datum.dim_2 >= brush.dim_2[0] &&
                datum.dim_2 <= brush.dim_2[1]

                if(isSelected) {
                    indices.push(index)
                }
                return indices;
              }, []);
            selectedIndices.sort();
            // Perform any desired action with the selected points            
            // Record the selection event in the viewManager
            // this.viewManager.recordSelectionEvent(selectedPoints);
        }        
        this.viewManager.notifySelectionChanged(selectedIndices);
        //console.log('selected indices:', selectedIndices);

        //this.view.change("table", this.view.changeset().insert([{"center_x":100, "center_y":100}]));
        //this.view.change("table", this.view.changeset().modify(0 , "center_x", 100));
        this.updateData(false);
    };

    addDataChangedListener(){                
        this.viewManager.OnSelectionChanged.add((selection)=>{    
            this.updateData(true);
        });
    }

    getSpec() {
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "A simple scatter plot example.",
            "params": [{
                "name": "brush",
                "select": {
                    "type": "interval",
                    "encodings": ["x", "y"],
                }               
              }],
            "data": {
                "name": "table"
            },
            "mark": {
                "type" : "point",
                "filled" : true,
            },
            "encoding": {
                "x": {"field": "dim_1", "type": "quantitative", "axis" : {"title" : this.data_dimensions[0]}},
                "y": {"field": "dim_2", "type": "quantitative", "axis" : {"title" : this.data_dimensions[1]}},                
                "color": {                    
                    "condition" : [
                        {"test" : "datum.selection_state == 1", "value" : "red"},
                        {"test" : "datum.selection_state == -1", "value" : "grey"}
                    ],
                    "value" : "blue"
                }
            },
            "config": {
                "legend": {"disable": false},
                "view": {
                    "actions": false
                }
            },
            
            "height": 200,
        }
    }

    /*
        "color": {
                    "field" : "center_x", 
                    "type": "quantitative", 
                    "scale": {"scheme": "viridis"},
                    "condition" : [
                        {"test" : "datum.selection_state == 1", "value" : "red"},
                        {"test" : "datum.selection_state == -1", "value" : "grey"}
                    ],
                    "value" : "blue"
                }

    */

    getData() {        
        return {
            table : this.getDataColumns()
        } 
    }

    getDataColumns() {
        if(this.data_dimensions.length == 2){
            return this.dataManager.project_data_2(this.data_dimensions);    
        } else {
            throw Error(this.data_dimensions);
        }
    }

    updateData(forceUpdate) {
        let newData = this.getDataColumns();
        if(forceUpdate){
            this.view.change("table", this.view.changeset().remove(()=>true).insert(newData)).runAsync();
        } else {
            this.view.change("table", this.view.changeset().remove(()=>true).insert(newData))
        }
        
    }

    render() {
        let that = this;
        return <VegaLite 
            spec = {this.getSpec()} 
            data = {this.getData()} 
            onNewView={(view) => {
                view.addSignalListener('brush', (event, brush) => {
                    that.handleBrushSelection(event, view.scenegraph().root, brush);
                });                
                that.view = view;               
                that.addDataChangedListener();
            }}/>      
    }
}

Template.vegaLiteView.onCreated(function () {
    this.viewManager = Template.currentData();
});

Template.vegaLiteView.onRendered(function () {
    this.viewManager = Template.currentData();
});

Template.vegaLiteView.helpers({

    ScatterPlot() {
        return VegaViewer;
    },

    viewManager() {
        return Template.instance().viewManager;
    }
});

export default VegaViewer