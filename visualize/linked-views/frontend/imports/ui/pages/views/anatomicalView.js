import React, { useState } from 'react';
import { deepCopy } from './core/utilCore';

import CoordinatedView from './core/coordinatedView';

import 'babylonjs-loaders';
import * as BABYLON from 'babylonjs';
import { blueRedColormap, getRGBFromString, getColormapIndex, hexToRgb } from './core/colorManager';


var greyColor = new BABYLON.Color3(50, 50, 50);
var intensity = 0.02;


// https://www.w3resource.com/javascript-exercises/javascript-math-exercise-23.php
function getId() {
  let dt = new Date().getTime();
  let uuid =
    'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      let r = (dt + Math.random() * 16) % 16 | 0;
      dt = Math.floor(dt / 16);
      return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  return uuid;
}

function createArrow(tpl, height, material) {
  let cone = BABYLON.MeshBuilder.CreateCylinder(
    getId(),
    { diameterTop: 0, diameterBottom: 50, height: 100, tessellation: 96 },
    tpl.scene);
  cone.position.y = height;
  cone.material = material;
  let line = BABYLON.Mesh.CreateLines(
    getId(), [new BABYLON.Vector3.Zero(), new BABYLON.Vector3(0, height, 0)],
    tpl.scene);
  line.color = material.diffuseColor;
  cone.setParent(line);

  return line;
}

function createAxes(tpl) {
  const size = 800;

  let rootMesh = new BABYLON.Mesh('axes', tpl.scene);

  let xMaterial = new BABYLON.StandardMaterial('xMaterial', tpl.scene);
  xMaterial.diffuseColor = new BABYLON.Color3(255, 0, 0);

  let yMaterial = new BABYLON.StandardMaterial('yMaterial', tpl.scene);
  yMaterial.diffuseColor = new BABYLON.Color3(0, 255, 0);

  let zMaterial = new BABYLON.StandardMaterial('zMaterial', tpl.scene);
  zMaterial.diffuseColor = new BABYLON.Color3(0, 0, 255);

  let arrowZ = createArrow(tpl, size, zMaterial);
  arrowZ.setParent(rootMesh);

  let arrowY = createArrow(tpl, size, yMaterial);
  arrowY.rotate(BABYLON.Axis.X, Math.PI / 2, BABYLON.Space.WORLD);
  arrowY.setParent(rootMesh);

  let arrowX = createArrow(tpl, size, xMaterial);
  arrowX.rotate(BABYLON.Axis.Z, -Math.PI / 2, BABYLON.Space.WORLD);
  arrowX.setParent(rootMesh);
}


function initMaterials(tpl) {
  var piaMat = new BABYLON.StandardMaterial('piaSurface', tpl.scene);
  if (tpl.context === 'frontView' || tpl.context === 'gallery') {
    piaMat.diffuseColor = new BABYLON.Color3(50, 50, 50);
    piaMat.alpha = 1;
  } else {
    piaMat.diffuseColor = new BABYLON.Color3(50, 50, 50);
    piaMat.alpha = 1;
  }
  piaMat.backFaceCulling = false;

  var wmMat = new BABYLON.StandardMaterial('wmSurface', tpl.scene);
  wmMat.diffuseColor = new BABYLON.Color3(50, 50, 50);
  wmMat.alpha = 1;
  wmMat.backFaceCulling = false;

  var mat1 = new BABYLON.StandardMaterial('defaultSurface', tpl.scene);
  mat1.diffuseColor = greyColor;
  mat1.alpha = 0.3;
  mat1.backFaceCulling = false;

  var mat1 = new BABYLON.StandardMaterial('transparentDendriteSurface', tpl.scene);
  mat1.diffuseColor = greyColor;
  mat1.alpha = 0.1;
  mat1.backFaceCulling = false;

  var mat2 = new BABYLON.StandardMaterial('lightSurface', tpl.scene);
  mat2.diffuseColor = new BABYLON.Color4(255, 255, 255, 0.3);
  mat2.backFaceCulling = false;

  var mat3 = new BABYLON.StandardMaterial('selectedSurface', tpl.scene);
  mat3.diffuseColor = new BABYLON.Color3(255, 0, 0);
  mat3.alpha = 0.7;
  mat3.backFaceCulling = false;

  var mat4 = new BABYLON.StandardMaterial('defaultCylinderSurface', tpl.scene);
  mat4.diffuseColor = new BABYLON.Color3(110, 110, 110);
  // mat4.emissiveColor = new BABYLON.Color3(255, 0, 0);
  mat4.specularColor = new BABYLON.Color3(255, 255, 255);
  mat4.alpha = 0.3;
  mat4.backFaceCulling = false;

  var matProbe = new BABYLON.StandardMaterial('probe', tpl.scene);
  matProbe.diffuseColor = new BABYLON.Color3(110, 110, 110);
  // mat4.emissiveColor = new BABYLON.Color3(255, 0, 0);
  matProbe.specularColor = new BABYLON.Color3(255, 255, 255);
  matProbe.alpha = 0.5;
  matProbe.backFaceCulling = false;

  var mat5 = new BABYLON.StandardMaterial('selectedCylinderSurface', tpl.scene);
  mat5.diffuseColor = new BABYLON.Color3(255, 0, 0);
  mat5.emissiveColor = new BABYLON.Color3(255, 0, 0);
  mat5.specularColor = new BABYLON.Color3(255, 0, 0);
  mat5.alpha = 0.3;
  mat5.backFaceCulling = false;

  tpl.interpolatedMaterials_blueRed = [];
  for (let i = 0; i < blueRedColormap.length; i++) {
    const rgbValues = getRGBFromString(blueRedColormap[i]);
    let matInterp = new BABYLON.StandardMaterial('blueRed_' + i.toString(), tpl.scene);
    //matInterp.diffuseColor = new BABYLON.Color3(...rgbValues);
    //matInterp.specularColor = new BABYLON.Color3(...rgbValues);
    //matInterp.alpha = 0.5;
    matInterp.emissiveColor = new BABYLON.Color3(rgbValues[0]/255, rgbValues[1]/255, rgbValues[2]/255);
    matInterp.backFaceCulling = false;
    matInterp.freeze();
    tpl.interpolatedMaterials_blueRed.push(matInterp)
  }

  tpl.materials_categorical = [];
  const colormap_categorical = tpl.viewManager.colorManager.getDefaultPropertyColors();
  for (let i = 0; i < colormap_categorical.length; i++) {
    const rgbValues = hexToRgb(colormap_categorical[i]);
    let mat = new BABYLON.StandardMaterial('categorical_' + i.toString(), tpl.scene);
    mat.diffuseColor = new BABYLON.Color3(...rgbValues);
    mat.backFaceCulling = false;
    mat.freeze();
    tpl.materials_categorical.push(mat);
  }
}


// Event handler to log camera parameters
function logCameraParameters(camera) {
  console.log("Camera Position:", camera.position);
  console.log("Camera Target:", camera.target);
  console.log("Camera Radius:", camera.radius);
  console.log("Camera Alpha:", camera.alpha);
  console.log("Camera Beta:", camera.beta);
  console.log("Camera FoV:", camera.fov);
  console.log("Camera Near Plane:", camera.minZ);
  console.log("Camera Far Plane:", camera.maxZ);
}


function createPointCloud(tpl) {

  let initPCS = () => {
    let data = tpl.pcs_data;
    nPoints = data[0].length;
    console.log(nPoints);
    const pointSize = tpl.configuration.pointSize ? tpl.configuration.pointSize : 2;
    tpl.pcs = new BABYLON.PointsCloudSystem("pcs", pointSize, tpl.scene, {
      updatable : true
    });
    let x = data[0];
    let y = data[1];
    let z = data[2];
    let exc = data[3];

    let initFunction = function (particle, i, s) {
      particle.position = new BABYLON.Vector3(x[i], z[i], y[i]);
      particle.color = new BABYLON.Color4(0.6 * exc[i], 0.6 * (1 - exc[i]), 0, 1);
    }
    
    tpl.pcs.updateParticle = function(particle) {      
      if(tpl.pcs_selection[particle.idx]){
        particle.position = new BABYLON.Vector3(x[particle.idx], z[particle.idx], y[particle.idx]);
      } else {
        particle.position = new BABYLON.Vector3(0, -10000, 0);
      }     
    }    

    tpl.pcs.beforeUpdateParticles = () => {
      tpl.pcs_initialized = false;
    }

    tpl.pcs.afterUpdateParticles = () => {
      tpl.pcs_initialized = true;
    }
    
    tpl.pcs.addPoints(nPoints, initFunction);
    tpl.pcs.buildMeshAsync().then(() => {
      tpl.pcs_initialized = true;
    });
  }

  const columns = tpl.data_sources; 
  console.log(columns);
  tpl.dataManager.loadValues((serverData) => {
    const values = serverData.values;
    data = [[],[],[],[]]    
    tpl.pcs_data = values.reduce((accum, x) => {
      accum[0].push(x[columns[0]]);
      accum[1].push(x[columns[1]]);
      accum[2].push(x[columns[2]]);
      accum[3].push(1);
      return accum;
    }, data);        
    tpl.pcs_selection = values.map((x,idx) => 1);    
    initPCS();
  }, tpl.table, [], columns, "expanded");  
}


function createScene(tpl) {
  if (tpl.scene) {
    return;
  }

  // Create a basic BJS Scene object
  tpl.scene = new BABYLON.Scene(tpl.engine);
  tpl.scene.pickable = true;
  if (tpl.configuration.name == "single-morphology") {
    tpl.scene.clearColor = new BABYLON.Color4(0.8, 0.8, 0.8, 1);
  } else {
    tpl.scene.clearColor = new BABYLON.Color4(1, 1, 1, 1);
  }


  var axis = new BABYLON.Vector3(0, 1, 0);
  var angle = Math.PI / 2;
  tpl.quaternion = new BABYLON.Quaternion.RotationAxis(axis, angle);

  // Create a FreeCamera, and set its position to {x: 0, y: 5, z: -10}

  let alpha = 0.7;
  let beta = 1.17;
  let radius = 5000;
  let centerZ = -300;

  if (tpl.configuration.name == "single-morphology") {
    tpl.camera = new BABYLON.ArcRotateCamera(
      'Camera', -3.48, 1.31, 1721, new BABYLON.Vector3(-100, 400, 0),
      tpl.scene);
    tpl.camera.lowerBetaLimit = null;
    tpl.camera.upperBetaLimit = null;
    tpl.camera.wheelDeltaPercentage = 0.02;
    tpl.camera.lowerRadiusLimit = 500;
    tpl.camera.upperRadiusLimit = 7000;
  } else if (tpl.configuration.name == "soma-barrelcortex-synapses"){
    tpl.camera = new BABYLON.ArcRotateCamera(
      'Camera', 1.55, 1.56, 2245, new BABYLON.Vector3(-100, 0, 0),
      tpl.scene);    
    tpl.camera.lowerBetaLimit = null;
    tpl.camera.upperBetaLimit = null;
    tpl.camera.wheelDeltaPercentage = 0.02;
    tpl.camera.lowerRadiusLimit = 500;
    tpl.camera.upperRadiusLimit = 7000;
  } else {
    tpl.camera = new BABYLON.ArcRotateCamera(
      'Camera', alpha, beta, radius, new BABYLON.Vector3(0, centerZ, 0),
      tpl.scene);
    // 0.64, 1.3, 5000
    // 0.92, 1.87, 5000 DENSE
    // 7.07 1.25 //DENSE cell types w grid
    // 6.73, 1.17
    // Target the camera to scene origin
    // tpl.camera.setTarget(BABYLON.Vector3.Zero());
    // Attach the camera to the canvas
    tpl.camera.lowerBetaLimit = null;
    tpl.camera.upperBetaLimit = null;
    tpl.camera.wheelDeltaPercentage = 0.02;
    tpl.camera.lowerRadiusLimit = 500;
    tpl.camera.upperRadiusLimit = 7000;
  }


  // Attach event listeners to camera properties
  // tpl.camera.onViewMatrixChangedObservable.add(logCameraParameters);
  // tpl.camera.onProjectionMatrixChangedObservable.add(logCameraParameters);

  tpl.camera.idleRotationWaitTime = 50000;
  tpl.camera.attachControl(tpl.canvas, false);

  var light = new BABYLON.HemisphericLight(
    'HemiLight', new BABYLON.Vector3(0, 0, -1), tpl.scene);
  light.intensity = 0.2 * intensity;

  var light2 = new BABYLON.HemisphericLight(
    'HemiLight2', new BABYLON.Vector3(0, 0, 1), tpl.scene);
  light2.intensity = 0.2 * intensity;

  initMaterials(tpl);

  if (tpl.configuration.showAxes) {
    createAxes(tpl);    
  }

  if(tpl.data_sources.length){
    createPointCloud(tpl);
  }

  createMorphology(tpl)
}


function createMorphology(tpl) {
  if (!tpl.morphology) {
    return;
  }

  const fixOrientation = (pts) => {
    
    if(tpl.swapYZ){      
      let ptsFixed = pts.map(p => {return [p[0],p[2],p[1]]});      
      return ptsFixed;
    } else {
      return pts;
    }          
  }

  const points = fixOrientation(tpl.morphology.points);
  const lines = tpl.morphology.lines;

  tpl.tubeHandles = [];
  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];

    let path = [];
    let pointIndices = [];
    for (let k = 1; k < line.length; k++) {
      const pointIdx = line[k];
      pointIndices.push(pointIdx);
      path.push(new BABYLON.Vector3(...points[pointIdx]))
    }

    let tube = BABYLON.MeshBuilder.CreateTube("segment" + lineIdx.toString(), { path: path, radius: 3, sideOrientation: BABYLON.Mesh.DOUBLESIDE, updatable: true }, tpl.scene);

    const middleIndex = Math.floor(pointIndices.length / 2)
    tube.pointIdx = lineIdx;
    if(tpl.configuration.name == "soma-barrelcortex-synapses"){
      tube.material = tpl.scene.getMaterialByName('transparentDendriteSurface');
    } else {
      tube.material = tpl.interpolatedMaterials_blueRed[0]
    }
    
    // create probe
    const probePosition = new BABYLON.Vector3(...points[pointIndices[middleIndex]]);
    const coneHeight = 150
    var cone = BABYLON.MeshBuilder.CreateCylinder("cone", { height: coneHeight, diameterTop: 0, diameterBottom: 20 }, tpl.scene);
    cone.rotate(BABYLON.Axis.X, Math.PI / 2, BABYLON.Space.WORLD);
    cone.rotate(BABYLON.Axis.Y, Math.PI / 2, BABYLON.Space.WORLD);
    cone.position = probePosition.subtract(new BABYLON.Vector3(0.5 * coneHeight, 0, 0));
    cone.setParent(tube);
    cone.isVisible = false;

    var sphere = BABYLON.MeshBuilder.CreateSphere("sphere", { diameter: 25 }, tpl.scene);    
    sphere.position = probePosition.subtract(new BABYLON.Vector3(coneHeight, 0, 0));
    sphere.setParent(cone);
    sphere.isVisible = false;

    // enable click events
    tube.actionManager = new BABYLON.ActionManager(tpl.scene);
    tube.actionManager.registerAction(
      new BABYLON.ExecuteCodeAction(
        BABYLON.ActionManager.OnPickTrigger,
        function (event) {          
          tpl.onTubeSelected(event.source);
        }
      )
    );

    // enable click events
    cone.actionManager = new BABYLON.ActionManager(tpl.scene);
    cone.actionManager.registerAction(
      new BABYLON.ExecuteCodeAction(
        BABYLON.ActionManager.OnPickTrigger,
        function (event) {          
          tpl.onTubeSelected(tube);
        }
      )
    );

    tpl.tubeHandles.push(tube);
  }
}




function initEngine(tpl) {
  if (tpl.engine) {
    return;
  }

  tpl.engine = new BABYLON.Engine(
    tpl.canvas, true, { preserveDrawingBuffer: true, stencil: true });
  tpl.engine.loadingUIBackgroundColor = "black";

  tpl.engine.onContextLostObservable.add(() => {
    console.log("context lost", tpl.name);
  });

  createScene(tpl);

  tpl.engine.runRenderLoop(function () {
    tpl.scene.render();
    if(tpl.pcs_initialized){
      tpl.pcs.setParticles();
    }
  });
}


const styleBackground = {
  backgroundColor: "white",
  color: "white"
}


class AnatomicalView extends CoordinatedView {
  constructor(props) {
    super(props);

    this.myRef = React.createRef();
    
    this.points = [] 

    this.name = props.name;
    this.selectedSegments = [];        
  }

  updateSelection(interactionEvent) {        
    if(!this.pcs_selection){
      return;
    }  
    this.points = interactionEvent.applyOperations(this.name, this.table, this.points);
    const selectedSet = new Set(this.points);        
    for(let i=0; i<this.pcs_selection.length; i++){
      this.pcs_selection[i] = selectedSet.has(i) ? 1 : 0;
    }
  }

  receiveData(dataType, data){
    if(dataType == "voltage_timeseries_points"){
      console.log(data.idx, this.name);

      if(data.idx === undefined){
        return;
      }

      if(data.idx % 2 == 0 && this.name != "dendrite"){
        return;
      } else if (data.idx % 2 == 1 && this.name != "dendrite 2"){
        return;
      }

      this.state.voltage_timeseries_points = data.voltage_timeseries_points;
      if(this.state.t_step){
        this.updateMembranePotential(this.state.t_step);
      } else {
        this.updateMembranePotential(0);
      }      
    } 
    if(dataType == "time"){
      this.state.t_step = data.t_step;
      this.updateMembranePotential(data.t_step);
    }
  }

  onTubeSelected(tube) {    
    const pointIdx = tube.pointIdx;    
    if (this.selectedSegments.indexOf(pointIdx) == -1) {
      this.selectedSegments.push(pointIdx);            
    } else {
      this.selectedSegments = this.selectedSegments.filter(x => x != pointIdx);      
    }    
    this.updateProbes(); 
    this.notify({
      interactionType : "select",
      selectedEntityType: "dendrite_segment",
      data : {
          data_type: "dendrite_segment",
          segment_ids : deepCopy(this.selectedSegments)
      }                
    });        
  }

  updateProbes(){
    for(let i = 0; i<this.tubeHandles.length; i++){
      const tube = this.tubeHandles[i];
      const probe = tube.getChildren()[0];
      const sphere = probe.getChildren()[0];
      if(this.selectedSegments.indexOf(tube.pointIdx) == -1){
        probe.isVisible = false;        
        sphere.isVisible = false;
      } else {
        probe.isVisible = true;
        sphere.isVisible = true;
        const idx = this.selectedSegments.indexOf(tube.pointIdx)
        const materialIdx = idx % this.materials_categorical.length;
        probe.material =  this.scene.getMaterialByName('probe');            
        sphere.material = this.materials_categorical[materialIdx];
      }
    }
  }


  updateMembranePotential(timeStep) {
    if(!this.state.voltage_timeseries_points){
      return;
    }
    
    let voltagesPoints = this.state.voltage_timeseries_points.voltage_traces[timeStep];
    for (let i = 0; i < this.tubeHandles.length; i++) {
      let tube = this.tubeHandles[i];
      let currentVoltage = voltagesPoints[tube.pointIdx];
      let colormapIdx = getColormapIndex(currentVoltage, -80, 20, blueRedColormap.length);
      tube.material = this.interpolatedMaterials_blueRed[colormapIdx]
    }
  }

  handleSelect(eventArgs) {
    this.viewManager.notifySelectionEvent(this.name, "select", deepCopy(eventArgs.points));
  }

  handleDeselect() {
    this.viewManager.notifySelectionEvent(this.name, "deselect");
  }

  render() {
    let that = this;
    return <div style={styleBackground}><canvas ref={this.myRef} width={this.width} height={this.height} /></div>
  }

  componentDidMount() {
    const canvas = this.myRef.current;
    this.canvas = canvas;

    let that = this;
        
    const morphology = this.configuration.morphology;
    if(morphology){
      this.swapYZ = this.configuration.morphologySwapYZ;
      this.viewManager.dataManager.getResource((morphology) => {
        that.morphology = morphology.jsonData;
        initEngine(that);        
      }, morphology);  
    } else {
      this.morphology = undefined;
      initEngine(that);
    }

    super.componentDidMount();
  }

  componentWillUnmount() {
    super.componentWillUnmount();   
    // free resources 
  }
}

export default AnatomicalView