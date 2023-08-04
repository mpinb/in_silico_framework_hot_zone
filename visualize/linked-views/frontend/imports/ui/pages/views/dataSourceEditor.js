import React, { useState } from 'react';
import Select from 'react-select';

import { deepCopy } from './core/utilCore.js';


const styleSelect = {
    width: 350,
    fontSize: 14
}

class DataSourceEditor extends React.Component {
    constructor(props) {
        super(props);

        this.viewManager = props.viewManager;
        this.dataManager = this.viewManager.dataManager;
        this.saveFn = props.saveFn;
        this.cancelFn = props.cancelFn;
        this.minSources = props.minSources;
        this.maxSources = props.maxSources;        
        this.tableName = props.tableName;        
        this.preselectedSources = deepCopy(props.selectedSources)

        let metaData = this.dataManager.metaData.filter(table => table.name == this.tableName);
        if(!metaData.length){
            console.log("invalid view spec", this.tableName, this.dataManager.metaData)            
        }
        metaData = metaData[0];
        this.availableColumns = [];
        for (let i = 0; i < metaData.columns.length; i++) {
            this.availableColumns = this.availableColumns.concat(metaData.columns[i]);
        }
        this.availableSelections = this.availableColumns.map(x => ({
            value: x,
            label: x
        }));

        let activeSelections = [];
        for (let i = 0; i < this.preselectedSources.length; i++) {
            activeSelections.push(this.preselectedSources[i]);
        }

        let n_empty = this.maxSources - activeSelections.length;
        for (let i = 0; i < n_empty; i++) {
            activeSelections.push(undefined);
        }
        this.state = {
            activeSelections: activeSelections,
            name: props.viewName
        }

        console.log(this.state.activeSelections);
    }

    handleSelectChange(index, event) {
        this.setState((state, props) => {
            if (event === null) {
                state.activeSelections[index] = undefined;
            } else {
                state.activeSelections[index] = event.value;
            }
            return state;
        });
    }

    handleNameChange(event) {
        this.setState((state)=>{
            state.name = event.target.value;
            return state;
        });
    }

    handleSaveClick() {
        let newSelection = [];
        for (let i = 0; i < this.state.activeSelections.length; i++) {
            if (this.state.activeSelections[i] !== undefined) {
                newSelection.push(this.state.activeSelections[i]);
            }
        }
        this.saveFn(this.state.name, newSelection);
    }

    handleCancelClick() {        
        this.cancelFn();
    }

    render() {
        console.log(this.state.activeSelections);
        console.log(this.availableColumns);
        return (
            <table style={{ width: '100%' }}>
                <tbody>
                    <tr>
                        <td colSpan={2}>                        
                            <button className="blueButton" onClick={this.handleSaveClick.bind(this)}>Save</button>                        
                            <button className="blueButton" style={{marginLeft:3}} onClick={this.handleCancelClick.bind(this)}>Cancel</button>                        
                        </td>
                    </tr>
                    <tr>
                        <td colSpan={2}>
                            <div style={{ width: "500px", height: '350px', overflow: 'auto' }}>
                                <table style={{ width: '100%' }}><tbody>
                                    <tr key="name-row"><td><input type="text" value={this.state.name} onInput={this.handleNameChange.bind(this)}></input></td></tr>
                                    {this.state.activeSelections.map((selectedValue, index) => (
                                        <tr key={index}>
                                            <td>
                                                <div style={{ display: 'flex' }}>
                                                    <Select
                                                        value={selectedValue}
                                                        onChange={this.handleSelectChange.bind(this, index)}
                                                        options={this.availableSelections}
                                                        style={styleSelect}
                                                    />
                                                </div>
                                            </td>
                                        </tr>))
                                    }
                                </tbody>
                                </table>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        )
    }
}

export default DataSourceEditor