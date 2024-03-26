// Create the HTML structure for the interactive figure
const figureContainer = document.createElement('div');
figureContainer.classList.add('figure-container');

// Create the first row
const firstRow = document.createElement('div');
firstRow.classList.add('row');
const firstRowBox = document.createElement('div');
firstRowBox.classList.add('box');
firstRowBox.textContent = 'In-vivo observation';
firstRow.appendChild(firstRowBox);
figureContainer.appendChild(firstRow);

// Create the second row
const secondRow = document.createElement('div');
secondRow.classList.add('row');
const networkModelBox = document.createElement('div');
networkModelBox.classList.add('box');
networkModelBox.textContent = 'Network model';
const multiScaleModelBox = document.createElement('div');
multiScaleModelBox.classList.add('box');
multiScaleModelBox.textContent = 'Multi-scale model';
const neuronModelBox = document.createElement('div');
neuronModelBox.classList.add('box');
neuronModelBox.textContent = 'Neuron model';
secondRow.appendChild(networkModelBox);
secondRow.appendChild(multiScaleModelBox);
secondRow.appendChild(neuronModelBox);
figureContainer.appendChild(secondRow);

// Create the third row
const thirdRow = document.createElement('div');
thirdRow.classList.add('row');
const mechanisticExplanationBox = document.createElement('div');
mechanisticExplanationBox.classList.add('box');
mechanisticExplanationBox.textContent = 'Mechanistic explanation';
thirdRow.appendChild(mechanisticExplanationBox);
figureContainer.appendChild(thirdRow);

// Add event listeners to the boxes in the second row
networkModelBox.addEventListener('click', () => {
    networkModelBox.classList.toggle('expanded');
});
multiScaleModelBox.addEventListener('click', () => {
    multiScaleModelBox.classList.toggle('expanded');
});
neuronModelBox.addEventListener('click', () => {
    neuronModelBox.classList.toggle('expanded');
});

// Create SVG arrows
const arrow1 = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
arrow1.innerHTML = '<line x1="50%" y1="0" x2="50%" y2="100%" stroke="black" />';
arrow1.style.position = 'absolute';
arrow1.style.top = '50%';
arrow1.style.left = '50%';
arrow1.style.transform = 'translate(-50%, -50%)';
figureContainer.insertBefore(arrow1, secondRow);

const arrow2 = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
arrow2.innerHTML = '<line x1="50%" y1="0" x2="50%" y2="100%" stroke="black" />';
arrow2.style.position = 'absolute';
arrow2.style.top = '50%';
arrow2.style.left = '50%';
arrow2.style.transform = 'translate(-50%, -50%)';
figureContainer.insertBefore(arrow2, thirdRow);

// Append the figure container to the document body
document.body.appendChild(figureContainer);