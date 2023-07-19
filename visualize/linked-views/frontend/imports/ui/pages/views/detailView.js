import './detailView.html'

import { Template } from 'meteor/templating';
import { ReactiveVar } from 'meteor/reactive-var';
import React, { useState } from 'react';
import { ReorderControl } from './reorderControl';


var vega = require('vega');
var embed = require('vega-embed/build/vega-embed.js');

const styleItemCheckbox = {
    textAlign: "left",
    fontSize: 14,
}

const styleMenuHeader = {
    textAlign: "left",
    fontSize: 14,
    fontWeight: "bold",
}

const styleDefault = {
    fontSize: 14,
}

export function httpRequest(url, responseType) {
    return new Promise(function (resolve, reject) {
      // Do the usual XHR stuff
      var req = new XMLHttpRequest();
      req.responseType = responseType;
      req.open('GET', url);
  
      req.onload = function () {
        if (req.status == 200) {
          resolve(req.response);
        }
        else {
          reject(Error(req.statusText));
        }
      };
      req.onerror = function () {
        reject(Error("Network Error"));
      };
      req.send();
    });
  }


class DownloadControl extends React.Component {
    constructor(props) {
        super(props);

        this.viewManager = props.data;
        this.tileManager = this.viewManager.tileManager;
        this.activeProfile = this.viewManager.dataManager.getActiveProfile();
        this.numChannels = this.activeProfile.channels.length;                
    }

    render() {
                
        const onDownloadCsv = (event) => {            
            let data = this.tileManager.getValuesAsCsv();
            const blob = new Blob([data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob)
            let a = document.createElement("a");
            a.setAttribute("href", url);
            a.setAttribute("download", "matrix.csv");
            document.body.appendChild(a); 
            a.click();
        }

        const onDownloadImage = (event) => {                        
            let canvas = document.getElementById("matrixCanvas");
            let imageData = canvas.toDataURL("image/png", 1).replace("image/png", "image/octet-stream");     
            let a = document.createElement("a");
            a.setAttribute("href", imageData);
            a.setAttribute("download", "matrix.png");
            document.body.appendChild(a); 
            a.click();
        }
        
        return <table>
            <tbody>
                <tr>
                    <td>
                        <div style={{fontSize:14}}><span onClick={onDownloadCsv} className="js-download fa fa-download fa-2x" style={{cursor:"pointer"}}></span></div>
                    </td>
                    <td style={{fontSize:14, paddingLeft:10}}>
                        csv-file
                    </td>
                </tr>
                <tr>
                    <td>
                        <div style={{fontSize:14}}><span onClick={onDownloadImage} className="js-download fa fa-download fa-2x" style={{cursor:"pointer"}}></span></div>
                    </td>
                    <td style={{fontSize:14, paddingLeft:10}}>
                        png-file
                    </td>
                </tr>
            </tbody>
        </table>
        
     
    }
}


Template.detailView.onCreated(function () {
    this.numSelectedTiles = new ReactiveVar(0);
    this.viewManager = Template.currentData();
    this.tileManager = this.viewManager.tileManager;
    this.dataManager = this.viewManager.dataManager;
    this.colorMap = ["#ff9896", "#9edae5", "#17becf", "#dbdb8d", "#bcbd22", "#f7b6d2", "#c5b0d5","#4A403A", "#A45D5D", "#FFC069", "#9467bd", "#8c564b", "#d62728", "#e377c2", "#ffbb78", "#ff7f0e",
    "#98df8a", "#c7c7c7", "#aec7e8", "#c49c94"];
    this.neuroglancerUrl="http://h01-dot-neuroglancer-demo.appspot.com/#!";
    this.neuroglancerwindowObject=null;
    this.selectedNeurons=new Set();

    let that = this;
    this.tileManager.OnTileSelectionChanged.add(() => {
        let numSelectedTiles = Object.keys(that.tileManager.selectedTiles).length;          
        that.numSelectedTiles.set(numSelectedTiles);

        try {
            let valuesChannel0 = [];
            let valuesChannel1 = [];
            let labels = [];
            let channelName0 = "";
            let channelName1 = "";
            let selectedTiles = Object.keys(that.tileManager.selectedTiles);
            let channelIndices = this.tileManager.getDisplayedChannelIndices();
            let hasSecondChannel = channelIndices.length == 2;

            for(let i=0; i<selectedTiles.length; i++){
                tileKey = JSON.parse(selectedTiles[i])
                let label = this.tileManager.getDescriptorForTile(tileKey); 
                labels.push(label);
                channelName0 = this.dataManager.getChannelDisplayName(channelIndices[0]);

                let value0 = this.tileManager.getValueForTile(tileKey, channelIndices[0], "tile_text");
                valuesChannel0.push(value0);
                
                if(hasSecondChannel){
                    channelName1 = this.dataManager.getChannelDisplayName(channelIndices[1]);
                    let value1 = this.tileManager.getValueForTile(tileKey, channelIndices[1], "tile_text");
                    valuesChannel1.push(value1);                    
                }
            }            

            if(valuesChannel0.length){
                let spec = undefined;
                if(hasSecondChannel){
                    spec = getBarChartsSpec2Channel(valuesChannel0, valuesChannel1, labels, channelName0, channelName1);
                } else {
                    let data = {                
                        y: valuesChannel0,
                        xLabels: labels
                    };
                    let tableData = convertData(data,this.colorMap)                
                    spec = getBarChartsSpec(tableData, channelName0);
                }
                            
                embed('#distributionvis', spec, { tooltip: { theme: 'dark' } }).then(function (res) {                    
                    this.distributionview = res.view;
                });                
            } else {             
                $('#distributionvis').empty();
            }
        } catch(e) {
            console.log(e);
        }
    });

});

Template.detailView.onRendered(function () {
    this.viewManager = Template.currentData();
    //testMatrixReorder();
    httpRequest("jsons/neuroglancer-config.json", 'json').then((response) => {
        //console.log(response);
        this.jsonRequest=response;
        let tempurl=this.neuroglancerUrl+encodeURIComponent(JSON.stringify(this.jsonRequest));
        //console.log(tempurl)
       // this.neuroglancerwindowObject= window.open(tempurl,'_blank');
       // console.log(this.neuroglancerwindowObject);
    });  
    this.viewManager.OnContainerResized.add((container) => {
        if (container.getState().name == "detail-viewer") {
            //console.log("detail view container resized");
        }
    });

    this.viewManager.OnTileClicked.add((tileKey) => {
        //console.log("tile clicked", tileKey);
    });

    this.dataManager.OnMatrixSelectionChanged.add((selection) => {
        //console.log("tree selection changed");
        // clear everything in detail view
    });

    this.viewManager.tileManager.OnTileSelectionChanged.notifyObservers();

});

Template.detailView.helpers({
    ReorderControl() {
        return ReorderControl
    },

    DownloadControl() {        
            return DownloadControl;
    },    

    viewManager() {
        return Template.instance().viewManager;
    },

    numSelectedTiles() {
        return Template.instance().numSelectedTiles.get();
    },

    neuronInspectorVisible() {
        let activeProfile = Template.instance().dataManager.getActiveProfile()        
        return activeProfile.name.indexOf("H01") != -1;
    }
})

Template.detailView.events({
    "click #reOrderMatrix": function (event, template) {
        testMatrixReorder();
    },
    "click #getSamples": function (event, template) {
        template.tileManager.getSamples((samples) => {
            let neuronIds = samples[0];
            let activeProfile=template.dataManager.getActiveProfile();            
            try {
                $('#neuronsvis').empty();
                if(activeProfile.name.indexOf('H01')!= -1){
                let neuronsTable = $('<table id="neuronsList" style="height:170px; width:100%; font-family: Arial, Helvetica, sans-serif; font-size:12px; " />');
                    for(let i = 0; i<neuronIds.length; i++){
                        let id= neuronIds[i]; 
                        let tableRow= $('<tr style="height:10px; border-bottom: 1px solid #fff;"/>');
                        let checboxTd=$('<td style=""/>');
                        let rand = Math.floor(Math.random()*16777215).toString(16);
                        let checkbox = $('<input type="checkbox" data-name="neuroglancer-neurons" id=neurons--' + id + ' value=' + id + '/>');
                        let nameTd=$('<td />');
                        $(checboxTd).append(checkbox);
                        $(nameTd).append(id);
                        $(tableRow).append(checboxTd);
                        $(tableRow).append(nameTd);
                        $(neuronsTable).append(tableRow);
                }
                $('#neuronsvis').append(neuronsTable);
                }               
                else{
                    let neuronsTable = $('<table id="neuronsList" style="height:170px; width:100%; font-family: Arial, Helvetica, sans-serif; font-size:12px; " />');
                    let neuronsString = "";
                    for(let i=0; i<neuronIds.length; i++){
                        if(i>0){
                            neuronsString += ", "
                        }
                        neuronsString += neuronIds[i].toString()
                    }    
                    let tableRow  = "<tr><td>" + neuronsString + "</tr></td>";
                    $(neuronsTable).append(tableRow);
                    $('#neuronsvis').append(neuronsTable);
                }
            } catch(e){
                console.log(e);
            }
            
        })

    },
    "click #launchNeuroGlancer": function (event, template) {
       let tempurl=template.neuroglancerUrl+encodeURIComponent(JSON.stringify(template.jsonRequest));
        template.neuroglancerwindowObject= window.open(tempurl,'_blank');
        console.log(template.neuroglancerwindowObject);
    },
    "click #clearTileSelection": function (event, template) {
        template.tileManager.clearSelectedTiles();
    },
    "click .card-header input[type='checkbox']": function (event, template) {
        let id = event.target.id.split("--");
        console.log(id)
        let request=structuredClone(template.jsonRequest)
        let segmentationLayer=request.layers[1];
        //template.neuroglancerwindowObject.history.replaceState({}, "_blank", tempurl);
        if (event.target.checked) {
         template.selectedNeurons.add(id[1]);
        }
        else {
        template.selectedNeurons.delete(id[1]);
        }
        segmentationLayer.segments=Array.from(template.selectedNeurons);
        let tempurl=template.neuroglancerUrl+encodeURIComponent(JSON.stringify(request));
        template.neuroglancerwindowObject.location.replace(tempurl);
      },
});


function convertData(barData, colorMap){    
    let xlabels=barData.xLabels;
    let y=barData.y;
    let tableData = [];
    for (let i=0; i<xlabels.length; i++) {
        let entry = { "category": xlabels[i], "count": y[i], 
        "title": "", "color": "blue" }
        tableData.push(entry)      
    }
    return tableData;
}


function getBarChartsSpec2Channel(values0, values1, labels, channelName1, channelName2){
    valuesConverted= [];
    for(let i=0; i<values0.length; i++){
        valuesConverted.push({
            category : labels[i],
            group : channelName1,
            value : values0[i]
        });
        valuesConverted.push({
            category : labels[i],
            group : channelName2,
            value : values1[i]
        });
    }    

    const getLabelPadding = (labels) => {
        let lengths = [];
        for(let i=0; i<labels.length; i++){
            lengths.push(labels[i].length)
        }
        let maxLength = Math.max(...lengths);
        return maxLength * 7;

    }
    
    let labelPadding = getLabelPadding(labels);
    return {
        "width": {"step": 20},
        "height": 200,
        "data": {
            values : valuesConverted
        },
        "mark": {
            "type": "bar",            
        },
        "spacing": 5,
        "encoding": {
          "column": {"field": "category", "title" : "", "header": {"orient": "bottom", "labelAngle" : -90, "labelPadding":labelPadding}},
          "y": {"field": "value", "title" : "", "type": "quantitative"},
          "x": {"field": "group", "axis": null},
          "color": {"field": "group", "title":""}
        },
        "config": {
          "view": {"stroke": "transparent"}
        }
      }
}



function getBarChartsSpec(data, chartTitle) {
    let mainSpec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 400,
        "height": 300,
        "title": chartTitle,
        "autosize": {
            "type": "fit",
            "contains": "padding"
        },
        "data": {
            "name": "table",
            "values": data
        },
        "mark": "bar",
        "spacing": 5,
        "encoding": {            
            "x": {
                "field": "category",
                "type": "nominal",
                "title": ""
            },
            "y": {
                "field": "count",
                "type": "quantitative",
                "title": "count"
            },
            "color": {
                "field": "color",
                "type": "nominal",                
                "title": "cell types",
                "scale": null                
            },
            "tooltip": [{
                "field": "category",
                "type": "nominal"
            }, {
                "field": "count",
                "type": "quantitative"
            }]
        }
    };
    return mainSpec;
}

