import React, { PropTypes } from 'react';
import createScatterplot from 'regl-scatterplot';
import { viridisColormapReverse, colorMapWhiteBlack } from './core/colorManager';
import CoordinatedView from './core/coordinatedView';

import { deepCopy } from './core/utilCore';
import { SelectionEvent } from './core/interactionEvent';



const styleBackground = {
  backgroundColor: "white",
  color: "white"
}

class ReglScatterPlot extends CoordinatedView {
  constructor(props) {
    super(props);

    this.myRef = React.createRef();

    this.has_color = this.data_sources.length == 3 && props.embedding == "none";
    this.embedding = props.embedding;
    this.previousSelection = [];

    this.scatterplot = undefined;
  }

  getData() {
    return {
      table: this.getDataColumns()
    }
  }


  updateSelection(interactionEvent) {
    const points = this.scatterplot.get("points");
    if (!points.length) {
      return;
    }
    const currentIndices = this.scatterplot.get("selectedPoints");
    const indicesNew = interactionEvent.applyOperations(this.name, this.table, this.previousSelection);
    this.previousSelection = deepCopy(indicesNew);
    this.scatterplot.select(indicesNew, { preventEvent: true });
  }


  handleSelect(eventArgs) {
    this.notify(new SelectionEvent(this.name, this.table).setIndices(eventArgs.points));
  }


  handleDeselect() {
    this.previousSelection = [];
    this.notify(new SelectionEvent(this.name, this.table).setDeselect());
  }


  render() {
    return <div style={styleBackground}><canvas ref={this.myRef} width={this.width} height={this.height} /></div>
  }


  componentDidMount() {
    const canvas = this.myRef.current;
    canvas.fillStyle = "white";
    if (this.data_sources.length < 2) {
      return;
    }

    const context = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();

    this.scatterplot = createScatterplot({
      canvas,
      width,
      height,
      pointSize: this.configuration.pointSize ? this.configuration.pointSize : 2,
      lassoOnLongPress: true,
      lassoColor: "#ffa500",
      pointColor: this.has_color ? viridisColormapReverse : ["#4682b4"],
      opacityInactiveMax: this.configuration.pointOpacity ? this.configuration.pointOpacity : 0.3,
      colorBy: this.has_color ? 'valueA' : undefined,
      keyMap: { alt: 'lasso', shift: 'rotate' }
    });


    let format = this.embedding == "PCA" ? "flat-normalized-PCA" : "flat-normalized";
    let that = this;
    this.dataManager.loadValues((data) => {

      that.currentData = data;

      let values = data.values;
      if (this.has_color) {
        values = values.map(v => [v[0], v[1], 0.5 * (v[2] + 1)]);
      }
      that.scatterplot.draw(values);


      if (this.viewManager.selectionOnLayoutChange.length) {
        const preselectedIndices = this.viewManager.selectionOnLayoutChange;
        this.previousSelection = deepCopy(preselectedIndices);
        this.notify(new SelectionEvent(this.name, this.table).setIndices(preselectedIndices));
      }


      that.scatterplot.subscribe("select", this.handleSelect.bind(this));
      that.scatterplot.subscribe("deselect", this.handleDeselect.bind(this));
    }, this.table, [], this.data_sources, format);

    super.componentDidMount();
  }

  componentWillUnmount() {
    super.componentWillUnmount();
    try {
      this.scatterplot.destroy();
    } catch (error) {
      console.log(error);
    }

  }
}

export default ReglScatterPlot