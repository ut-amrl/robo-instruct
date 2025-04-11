// // Data
// var models = [
//   "DeepSeek-R1-Distill-Qwen-32B", 
//   "GPT-4o-mini", 
//   "Starcoder2", 
//   "Starcoder2", 
//   "Gemini-Pro", 
//   "SI-CL-7B", 
//   "DSC-33B", 
//   "Llama3-8B-Inst", 
//   "CL-34B-Python", 
//   "CL-7B-Python"
// ];
// var passAt1 = [67.5, 62.5, 62.5, 
//                68.75, 67.5, 55, 40,
//                66.25, 57.5, 55, 42.5,
//                68.75, 65, 62.5, 55, 
//                65, 60, 57.5, 51.5];

// var colors = [
//     '#BDBDBD', '#BDBDBD', '#BDBDBD', 
//     '#D2A3F9', '#C7CDFB', '#FEFEFE', '#BDBDBD',
//     '#D2A3F9', '#C7CDFB', '#FEFEFE', '#BDBDBD',
//     '#D2A3F9', '#C7CDFB', '#FEFEFE', '#BDBDBD',
//     '#D2A3F9', '#C7CDFB', '#FEFEFE', '#BDBDBD'
// ];
// legendIndices = [1, 5, 9, 8]
// names = ["Robo-Instruct (Ours)", "Self-Instruct", "Base Model", "Other Model"]
// const data = []
// // Separate models into their respective categories
// for(let i = 0; i < models.length; i++) {
//   let d = {
//       x: [models[i]],
//       y: [passAt1[i]],
//       type: 'bar',
//       marker: {
//           color: colors[i],
//           line: {
//               color: '#333',
//               width: 1.5
//           }
//       },
//       text: [passAt1[i]],
//       textposition: 'auto',
//       hoverinfo: 'x+y',
//       opacity: 0.8,
//       showlegend: legendIndices.includes(i),
//       name: names[legendIndices.indexOf(i)],
//   };  
//   data.push(d);
// }

// const layout = {
//   xaxis: {
//       title: 'Models'
//   },
//   yaxis: {
//       title: 'Pass@1 (%)'
//   }
// };

// Plotly.newPlot('bar-chart', data, layout);
// console.log(data)

const models = ['CodeLlama-7B', 'Llama-3-8B', 'Qwen-7B', 'Gemma2-9B'];

const traceRobo = {
  x: models,
  y: [68.8, 66.3, 68.8, 65],
  name: 'Robo-Instruct (Ours)',
  type: 'bar',
  marker: { color: '#D2A3F9' },
  text: [68.8, 66.3, 68.8, 65],
  textposition: 'auto'
};

const traceEvol = {
  x: models,
  y: [57.5, 57.5, 65, 60],
  name: 'Evol-Instruct',
  type: 'bar',
  marker: { color: '#A3C4F9' },
  text: [57.5, 57.5, 65, 60],
  textposition: 'auto'
};

const traceSelf = {
  x: models,
  y: [55, 55, 62.5, 57.5],
  name: 'Self-Instruct',
  type: 'bar',
  marker: { color: '#C7CDFB' },
  text: [55, 55, 62.5, 57.5],
  textposition: 'auto'
};

const traceBase = {
  x: models,
  y: [40, 42.5, 55, 51.5],
  name: 'Base Model',
  type: 'bar',
  marker: { color: '#BDBDBD' },
  text: [40, 42.5, 55, 51.5],
  textposition: 'auto'
};

const layout = {
  barmode: 'group',
  xaxis: {
    title: 'Models',
    tickangle: -30
  },
  yaxis: {
    title: 'Pass@1 (%)',
    range: [0, 80]
  },
  title: {
    text: '📊 Evaluation Results',
    font: {
      size: 20
    }
  },
  legend: {
    orientation: 'h',
    y: -0.3
  }
};

Plotly.newPlot('bar-chart', [traceRobo, traceEvol, traceSelf, traceBase], layout);