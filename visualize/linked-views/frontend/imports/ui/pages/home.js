import './home.html'
import '/imports/ui/css/goldenlayout-base.css';
import '/imports/ui/css/goldenlayout-light-theme.css';

import './views/projectView';
import './views/grammarView.js';
import './views/gridLayout.js';
import './views/viewSpecifications.js';

import { Template } from 'meteor/templating';
import { ViewManager } from './views/core/viewManager';



Template.home.onRendered(() => {

    const GoldenLayout = require('golden-layout');
    const viewManager = new ViewManager()

    viewManager.dataManager.loadInitialData(() => {

        let config = {
            dimensions: {
            },
            settings: {
                showPopoutIcon: false,
                reorderEnabled: false,
            },
            content: [{
                type: 'row',
                content: [
                    {
                        type: 'column',
                        width: 30,
                        content: [
                            {
                                type: 'stack',
                                id: "sidebar",
                                content: [
                                    {
                                        type: 'component',
                                        componentName: 'viewer',
                                        title: 'Session',
                                        isClosable: false,
                                        componentState: { name: 'channel-viewer', id: 'channel-viewer', uuid: "new", loaded: 'false' }
                                    },
                                    {
                                        type: 'component',
                                        componentName: 'viewer',
                                        title: 'View specifications',
                                        isClosable: false,
                                        componentState: { name: 'tree-viewer', id: 'tree-viewer', uuid: "new", loaded: 'false' }
                                    },
                                    {
                                        type: 'component',
                                        componentName: 'viewer',
                                        title: 'Interaction grammar',
                                        isClosable: false,
                                        componentState: { name: 'project-viewer', id: 'project-viewer', uuid: "new", loaded: 'false' }
                                    }                                 
                                ]
                            }
                        ]
                    },
                    {
                        type: 'column',
                        width: 70,
                        content: [
                            {
                                type: 'stack',
                                id: "sidebar",
                                content: [

                                    {
                                        type: 'component',
                                        componentName: 'viewer',
                                        title: 'Workspace',
                                        id: 'matrix-viewer',
                                        isClosable: false,
                                        componentState: { name: 'matrix-viewer', id: 'matrix-viewer', uuid: "new", loaded: 'false' }
                                    }
                                ]
                            },
                        ]
                    }              
                ]
            }]
        };
                
        let container = document.getElementById('layoutContainer');
        myLayout = new GoldenLayout(config, container);

        myLayout.registerComponent('viewer', function (container, componentState) {
            container.getElement().html('<div class="' + componentState.name + '"></div>');
            container.on('show', function () {
                if (container.getState().loaded == 'false') {
                    $(function () {
                        createTemplate(container.getState().name, viewManager);
                        container.extendState({ loaded: 'true' });
                    });
                }
            });

            container.on('resize', function () {
                if (container.getState().loaded == 'true') {
                    //viewManager.onResize(container);
                }
            });
        });

        myLayout.init();
    });

});


function createTemplate(name, viewManager) {

    if (name == 'tree-viewer') {
        Blaze.renderWithData(Template.viewSpecifications, viewManager, $('.tree-viewer')[0]);
    }

    if (name == 'project-viewer') {
        Blaze.renderWithData(Template.grammarView, viewManager, $('.project-viewer')[0]);
    }    

    
    if (name == 'channel-viewer') {
        Blaze.renderWithData(Template.projectView, viewManager, $('.channel-viewer')[0]);
    }
    

    if (name == 'matrix-viewer') {
        Blaze.renderWithData(Template.gridLayout, viewManager, $('.matrix-viewer')[0]);
    }
}