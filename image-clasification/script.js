let model;

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

const imgEl = document.getElementById("img");
const descEl = document.getElementById("descripcion_imagen");
var count = 0;
var net ;
var webcam;
async function app(){
	console.log("Cargando modelo de identificacion de imagenes");
  net= await mobilenet.load();
	console.log("Carga terminada")
  //clasificamos la imagen de carga
	const result = await net.classify(imgEl);
	console.log(result);
  descEl.innerHTML= JSON.stringify(result);




  //obtenemos datos del webcam
	webcam = await tf.data.webcam(webcamElement);
  //y los vamos procesando
  while (true) {
    const img = await webcam.capture();

    const result = await net.classify(img);

    const activation = net.infer(img, 'conv_preds');
    var result2;
    try {
      result2 = await classifier.predictClass(activation);
    } catch (error) {
      result2 = {};
    }

    const classes = ["Untrained", "Gato", "Dino" , "Alex", "OK","Rock"]

    document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `;

    try {
      document.getElementById("console2").innerText = `
    prediction: ${classes[result2.label]}\n
    probability: ${result2.confidences[result2.label]}
    `;
    } catch (error) {
      document.getElementById("console2").innerText="Untrained";
    }



    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

img.onload =async function() {

   try {
     result = await net.classify(img);
     descEl.innerHTML= JSON.stringify(result);
   } catch (error) {

   }
 }

async function  cambiarImagen(){
  count =count + 1;
  imgEl.src="https://picsum.photos/200/300?random=" + count;
  descEl.innerHTM = "";
  }


//add example
async function addExample (classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  //liberamos el tensor
  img.dispose()
}

const saveKnn = async () => {
    //obtenemos el dataset actual del clasificador (labels y vectores)
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = "knnClassifier";
    //lo almacenamos en el localStorage
    localStorage.setItem(storageKey, strClassifier);
};


const loadKnn = async ()=>{
    const storageKey = "knnClassifier";
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};


app()
