import React, { PropTypes } from 'react';
import { VegaLite, Signal } from 'react-vega'
import CoordinatedView from './core/coordinatedView';
import { deepCopy } from './core/utilCore';


class VegaView extends CoordinatedView {
    constructor(props) {
        super(props);

        this.state.data = [];
        this.previousSelection = [];

        const emptySpec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",           
            "height": 0.75,
            "width": 0.8,
        }
    
        this.vegaSpec = this.configuration.vegaSpec ? this.configuration.vegaSpec : emptySpec;
        this.vegaSpec.height = this.vegaSpec.height * this.height;
        this.vegaSpec.width = this.vegaSpec.width * this.width;
    }


    updateSelection(interactionEvent) {
        const indicesNew = interactionEvent.applyOperations(this.name, this.table, this.previousSelection);
        this.previousSelection = deepCopy(indicesNew);

        let that = this;
        this.dataManager.loadValues((data) => {
            let values = data.values;
            that.setState((state) => {
                state.data = values;
                return state;
            });

        }, this.table, indicesNew, this.data_sources, "expanded");
    }

    getData() {
        return {
            "table": deepCopy(this.state.data)
        };
    }

    getSpec() {        
        //console.log(this.vegaSpec);
        return this.vegaSpec
    }

    render() {
        return <VegaLite
            spec={this.getSpec()}
            data={this.getData()}
        />
    }

    componentDidMount() {
        let that = this;
        this.dataManager.loadValues((data) => {
            let values = data.values;
            that.setState((state) => {
                state.data = values;
                return state;
            });

        }, this.table, [], this.data_sources, "expanded");
        super.componentDidMount();
    }
}

export default VegaView