// Data
var models = [
  "GPT-4", 
  "RI-CL-7B (Ours)", 
  "GPT-3.5-Turbo", 
  "Starcoder2", 
  "Gemini-Pro", 
  "SI-CL-7B", 
  "DSC-33B", 
  "Llama3-8B-Inst", 
  "CL-34B-Python", 
  "CL-7B-Python"
];
var passAt1 = [85.81, 68.75, 67.5, 62.5, 60.0, 55.0, 53.75, 48.75, 48.25, 40.0];

var colors = [
    '#d3d3d3', '#98b0f9', '#d3d3d3', '#d3d3d3', '#d3d3d3',
    '#c870d7', '#d3d3d3', '#d3d3d3', '#d3d3d3', '#696969'
];
legendIndices = [1, 5, 9, 8]
names = ["Robo-Instruct (Ours)", "Self-Instruct", "Base Model", "Other Model"]
const data = []
// Separate models into their respective categories
for(let i = 0; i < models.length; i++) {
  let d = {
      x: [models[i]],
      y: [passAt1[i]],
      type: 'bar',
      marker: {
          color: colors[i],
          line: {
              color: '#333',
              width: 1.5
          }
      },
      text: [passAt1[i]],
      textposition: 'auto',
      hoverinfo: 'x+y',
      opacity: 0.8,
      showlegend: legendIndices.includes(i),
      name: names[legendIndices.indexOf(i)],
  };  
  data.push(d);
}

const layout = {
  xaxis: {
      title: 'Models'
  },
  yaxis: {
      title: 'Pass@1 (%)'
  }
};

Plotly.newPlot('bar-chart', data, layout);
console.log(data)