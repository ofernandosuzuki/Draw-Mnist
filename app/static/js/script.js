var canvas = document.getElementById('myCanvas');
var clearButton = document.getElementById('clearButton');
var predictButton = document.getElementById('predictButton');
var predictionElement = document.getElementById('prediction');

var ctx = canvas.getContext('2d');
var painting = false;

function startDraw(e) {
    painting = true;
    draw(e);
}

function endDraw() {
    painting = false;
    ctx.beginPath();
}

function draw(e) {
    if (!painting) return;
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    // Obtenha as coordenadas do mouse relativas ao canvas
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function predict() {
  var image = canvas.toDataURL('image/png');
  fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          image: image
      })
  })
  .then(response => response.json())
  .then(data => {
      predictionElement.textContent = 'Prediction: ' + data.prediction;
  });
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mousemove', draw);
clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predict);