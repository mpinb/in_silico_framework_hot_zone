def biophysics_fitting_1():
    return {
        "data": {
            "grammar": {
                "views": [{
                    "configuration": {
                        "density_grid_shape": [100, 100],
                        "format": "count",
                        "pointOpacity": 0.8,
                        "pointSize": 5
                    },
                    "dataColumn": ["PCA-ephys-1", "PCA-ephys-2"],
                    "id": 1,
                    "maxNumDatasources": 2,
                    "minNumDatasources": 2,
                    "name": "2-channel",
                    "dataTable": "vaex_df",
                    "type": "density-2-channel"
                }, {
                    "configuration": {
                        "density_grid_shape": [100, 100],
                        "format": "median",
                        "pointOpacity": 0.8,
                        "pointSize": 5
                    },
                    "dataColumn": [
                        "ephys.CaDynamics_E2_v2.axon.decay",
                        "ephys.CaDynamics_E2_v2.axon.gamma",
                        "ephys.SKv3_1.axon.gSKv3_1bar"
                    ],
                    "id": 2,
                    "maxNumDatasources": 3,
                    "minNumDatasources": 3,
                    "name": "3-channel",
                    "dataTable": "vaex_df",
                    "type": "density-3-channel"
                }],
                "grid": {
                    "cols": 12,
                    "rowHeight": 100,
                    "width": 1200
                },
                "initialLayout": "L1",
                "interactions": {},
                "layouts": {
                    "L1": [{
                        "h": 6,
                        "view": "3-channel",
                        "w": 12,
                        "x": 0,
                        "y": 0
                    }]
                }
            }
        }
    }


def biophysics_fitting_2():
    return {
        "data": {
            "grammar": {
                "views": [{
                    "configuration": {
                        "density_grid_shape": [100, 100],
                        "format": "count",
                        "pointOpacity": 0.8,
                        "pointSize": 5
                    },
                    "dataColumn": [
                        "ephys.Ca_HVA.apic.gCa_HVAbar", "ephys.K_Tst.axon.gK_Tstbar"
                    ],
                    "id": 1,
                    "maxNumDatasources": 2,
                    "minNumDatasources": 2,
                    "name": "2-channel",
                    "dataTable": "vaex_df",
                    "type": "density-2-channel"
                }, {
                    "configuration": {
                        "density_grid_shape": [100, 100],
                        "format": "median",
                        "pointOpacity": 0.8,
                        "pointSize": 5
                    },
                    "dataColumn": [
                        "ephys.CaDynamics_E2_v2.axon.decay",
                        "ephys.CaDynamics_E2_v2.axon.gamma",
                        "ephys.SKv3_1.axon.gSKv3_1bar"
                    ],
                    "id": 2,
                    "maxNumDatasources": 3,
                    "minNumDatasources": 3,
                    "name": "3-channel",
                    "dataTable": "vaex_df",
                    "type": "density-3-channel"
                }],
                "grid": {
                    "cols": 12,
                    "rowHeight": 100,
                    "width": 1200
                },
                "initialLayout": "L1",
                "interactions": {},
                "layouts": {
                    "L1": [{
                        "h": 6,
                        "view": "2-channel",
                        "w": 6,
                        "x": 0,
                        "y": 0
                    }, {
                        "h": 6,
                        "view": "3-channel",
                        "w": 6,
                        "x": 6,
                        "y": 0
                    }]
                }
            }
        }
    }
