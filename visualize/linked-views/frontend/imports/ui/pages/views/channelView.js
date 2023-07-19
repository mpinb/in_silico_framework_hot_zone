import './channelView.html'
import { Template } from 'meteor/templating';
import { Colorscale } from 'react-colorscales';
import ColorscalePicker from 'react-colorscales';
import React, { useState } from 'react';
import { deepCopy } from './core/utilCore';


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


class ChannelControl extends React.Component {
    constructor(props) {
        super(props);

        this.viewManager = props.data;
        this.tileManager = this.viewManager.tileManager;
        this.activeProfile = this.viewManager.dataManager.getActiveProfile();
        this.numChannels = this.activeProfile.channels.length;
        this.hasSecond = this.numChannels > 1;

        let settings = deepCopy(this.tileManager.channelSettings)
        settings.controlInteraction = {
            editingScale1: false,
            editingScale2: false,
        }
        this.state = settings;
    }

    initUpdate(state) {
        let settings = deepCopy(state);
        delete settings.controlInteraction;
        this.tileManager.setChannelSettings(settings);
    }

    onColorscaleChanged(idx, event) {
        console.log("picked colorscale", event);
        this.setState((state, props) => {
            if (idx == 0) {
                state.scale1 = event;
                state.controlInteraction.editingScale1 = false;
            } else {
                state.scale2 = event;
                state.controlInteraction.editingScale2 = false;
            }
            this.initUpdate(state);
            return state;
        });
    }

    onCheckboxSeparateScales(event) {
        this.setState((state, props) => {
            state.separateScales = event.target.checked;
            this.initUpdate(state);
            return state;
        });
    }

    onChangeDisplay(event) {
        this.setState((state, props) => {
            state.display = event.target.value;
            this.initUpdate(state);
            return state;
        });
    }

    onChangeDisplayMode(event) {
        this.setState((state, props) => {
            state.displayMode = event.target.value;
            this.initUpdate(state);
            return state;
        });
    }

    onChangeNormalizationMode(event) {
        this.setState((state, props) => {
            state.normalizationMode = event.target.value;
            this.initUpdate(state);
            return state;
        });
    }


    getChannelDescriptor(channelIdx) {
        if (channelIdx < this.numChannels) {
            return this.viewManager.dataManager.getChannelDisplayName(channelIdx);
        } else {
            return "n/a";
        }
    }

    onColorscaleClicked(channelIdx, event) {
        this.setState((state, props) => {
            if (channelIdx == 0) {
                state.controlInteraction.editingScale1 = true;
                state.controlInteraction.editingScale2 = false;
            } else {
                state.controlInteraction.editingScale1 = false;
                state.controlInteraction.editingScale2 = true;
            }

            return state;
        });
    }

    render() {

        const getDescriptor = (channelIdx) => {
            return <table style={styleDefault}>
                <tbody>
                    <tr>
                        <td><div>channel {channelIdx + 1}:&nbsp;&nbsp;</div></td>
                        <td><div><i>{this.getChannelDescriptor(channelIdx)}</i></div></td>
                    </tr>
                </tbody>
            </table>
        };

        const getDisplayRadioButtons = () => {
            return <div style={styleDefault}>
                <table>
                    <tbody>
                        <tr>
                            <td>
                                <label>
                                    <input type="radio" value="1" name="display" onChange={this.onChangeDisplay.bind(this)} checked={this.state.display == "1"} /> 1
                                </label>
                            </td>
                            <td style={{ paddingLeft: 15 }}>
                                <label>
                                    <input type="radio" value="2" name="display" onChange={this.onChangeDisplay.bind(this)} checked={this.state.display == "2"} disabled={!this.hasSecond} /> 2
                                </label>
                            </td>
                            <td style={{ paddingLeft: 15 }}>
                                <label>
                                    <input type="radio" value="1_2" name="display" onChange={this.onChangeDisplay.bind(this)} checked={this.state.display == "1_2"} disabled={!this.hasSecond} /> 1 and 2
                                </label>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        }

        const getDisplayModeRadioButtons = () => {
            return (
                <div style={styleDefault}>
                    <table>
                        <tbody>
                            <tr>
                                <td>
                                    <label>
                                        <input type="radio" value="values" name="displayMode" onChange={this.onChangeDisplayMode.bind(this)} checked={this.state.displayMode == "values"} /> values
                                    </label>
                                </td>
                                <td style={{ paddingLeft: 15 }}>
                                    <label>
                                        <input type="radio" value="color" name="displayMode" onChange={this.onChangeDisplayMode.bind(this)} checked={this.state.displayMode == "color"} /> color
                                    </label>
                                </td>
                                <td style={{ paddingLeft: 15 }}>
                                    <label>
                                        <input type="radio" value="values_color" name="displayMode" onChange={this.onChangeDisplayMode.bind(this)} checked={this.state.displayMode == "values_color"} /> values and color
                                    </label>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            );
        }

        const getNormalizationRadioButtons = () => {
            return (
                <div style={styleDefault}>
                    <table>
                        <tbody>
                            <tr>
                                <td>
                                    <label>
                                        <input type="radio" value="max_value" name="normalizationMode" onChange={this.onChangeNormalizationMode.bind(this)} checked={this.state.normalizationMode == "max_value"} /> max value
                                    </label>
                                </td>
                            </tr><tr>
                                <td>
                                    <label>
                                        <input type="radio" value="max_value_rows" name="normalizationMode" onChange={this.onChangeNormalizationMode.bind(this)} checked={this.state.normalizationMode == "max_value_rows"} /> max value per row
                                    </label>
                                </td>
                            </tr><tr>
                                <td>
                                    <label>
                                        <input type="radio" value="max_value_cols" name="normalizationMode" onChange={this.onChangeNormalizationMode.bind(this)} checked={this.state.normalizationMode == "max_value_cols"} /> max value per column
                                    </label>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            );
        }


        const getColorscalePicker = () => {
            let channelIdx = undefined;
            if (this.state.controlInteraction.editingScale1) {
                channelIdx = 0;
            } else if (this.state.controlInteraction.editingScale2) {
                channelIdx = 1;
            }
            if (channelIdx !== undefined) {
                return <ColorscalePicker
                    onChange={this.onColorscaleChanged.bind(this, channelIdx)}
                    colorscale={this.tileManager.getDefaultColorscale()}
                    width="100%"
                />
            } else {
                return <div></div>
            }
        }

        const getColorscale2 = () => {
            if (this.hasSecond) {
                return <Colorscale colorscale={this.state.scale2} onClick={this.onColorscaleClicked.bind(this, 1)} />;
            } else {
                return <div></div>
            }
        }

        return <table style={{ width: "100%" }}>
            <tbody>
                <tr>
                    <td className='badge badge-secondary matrixHeader'>
                        channel semantics
                    </td>
                </tr>
                <tr>
                    <td>
                        {getDescriptor(0)}
                    </td>
                </tr>
                <tr>
                    <td>
                        {getDescriptor(1)}
                    </td>
                </tr>
                <tr>
                    <td>
                        <div style={{ marginTop: 20 }} className="badge badge-secondary matrixHeader">
                            display channel(s)
                        </div>
                    </td>
                </tr>
                <tr>
                    <td>
                        {getDisplayRadioButtons()}
                    </td>
                </tr>
                <tr>
                    <td className='badge badge-secondary matrixHeader'>
                        display mode
                    </td>
                </tr>
                <tr>
                    <td>
                        {getDisplayModeRadioButtons()}
                    </td>
                </tr>
                <tr>
                    <td className='badge badge-secondary matrixHeader'>
                        colorscale(s)
                    </td>
                </tr>
                <tr>
                    <td>
                        <Colorscale colorscale={this.state.scale1} onClick={this.onColorscaleClicked.bind(this, 0)} />
                    </td>
                </tr>
                <tr>
                    <td>
                        {getColorscale2()}
                    </td>
                </tr>
                <tr>
                    <td>
                        {getColorscalePicker()}
                    </td>
                </tr>
                <tr>
                    <td className='badge badge-secondary matrixHeader'>
                        normalization (colorscale)
                    </td>
                </tr>
                <tr>
                    <td>
                        {getNormalizationRadioButtons()}
                    </td>
                </tr>
                <tr>
                    <td style={styleItemCheckbox}>
                        <label><input type="checkbox" onChange={this.onCheckboxSeparateScales.bind(this)} checked={this.state.separateScales} disabled={!this.hasSecond}></input>&nbsp;normalize channels independently</label>
                    </td>
                </tr>
            </tbody>
        </table>;
    }
}


Template.channelView.onCreated(function () {
    this.viewManager = Template.currentData();
});

Template.channelView.onRendered(function () {
    this.viewManager = Template.currentData();
});

Template.channelView.helpers({
    ChannelControl() {
        return ChannelControl;
    },

    viewManager() {
        return Template.instance().viewManager;
    }
});


