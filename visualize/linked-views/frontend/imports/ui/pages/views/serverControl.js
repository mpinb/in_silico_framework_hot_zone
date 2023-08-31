import React, { useState } from 'react';


export function getSessionDataServerURL(defaultUrl) {
    let url = window.sessionStorage.getItem("dataServerURL")
    console.log(url);
    if (url) {
        return url
    } else {
        return defaultUrl
    }
}

function setDataServerUrl(url){
    window.sessionStorage.setItem("dataServerURL", url);
}


function isValidUrl(url) {    
    try {
        let foo = new URL(url);
    } catch (_) {
        return false;  
    }
    return true;
}


class ServerControl extends React.Component {
    constructor(props) {
        super(props);

        this.viewManager = props.viewManager;
        this.dataManager = this.viewManager.dataManager;

        const currentUrl = getSessionDataServerURL(Meteor.settings.public.DATA_SERVER_DEV);

        this.state = {
            url: currentUrl
        }
    }

    handleUrlChange(event) {
        this.setState((state) => {
            state.url = event.target.value;
            return state;
        });
    }

    handleSaveClick() {
        const url = this.state.url;
        if(isValidUrl(url)){
            console.log("save");
            setDataServerUrl(url);
            location.reload();
        }        
    }

    render() {
        const inputColor = isValidUrl(this.state.url) ? "white" : "LightCoral";

        return <table style={{ width: '100%' }}><tbody>
            <tr>
                <td>
                    <div className='codeTextHeader'>Data server</div>
                </td>
            </tr>
            <tr>
                <td>                    
                    <input style={{ width: '100%', backgroundColor: inputColor }} type="text" value={this.state.url} onInput={this.handleUrlChange.bind(this)}></input>                                        
                </td>
            </tr>
            <tr>
                <td>
                    <button className="blueButton" onClick={this.handleSaveClick.bind(this)}>Save</button>                            
                </td>
            </tr>
        </tbody>
        </table>
    }
}

export default ServerControl

