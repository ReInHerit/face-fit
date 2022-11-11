/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
const fs = require('fs');
const spawn = require('child_process').spawn;
//const cors = require('cors')
//const { createProxyMiddleware } = require('http-proxy-middleware');
const fetch = (url) => import('node-fetch').then(({default: fetch}) => fetch(url));
const http = require('http');
const { fork } = require('child_process');
//const request = require('request-promise')
/**
 * App Variables
 */
const app = express();
const port = process.env.PORT || "8000";
const host = "localhost";
//const api_service_url = "localhost:8000/server";
//app.use(cors())
let ref_images = [];
/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
// Info GET endpoint
//const server = http.createServer();

app.on('request', (req, res) => {
  if (req.url === 'http://localhost:8000/info') {
    const compute = fork('main.js');
    compute.send('start');
    compute.on('message', obj => {
      console.log(obj)
      const obj_str = JSON.stringify(obj)
      const script = "arraysum2.py";
      const py = spawn('python', [script, obj_str]);

      resultString = '';
      // As the stdout data stream is chunked, we need to concat all the chunks.
      py.stdout.on('data', (data) => {
//          console.log(`stderr: ${data}`);
          resultString += data.toString()
        });

// Parse the string as JSON when stdout data stream ends
      py.stdout.on('end', function () {
           let resultData = {'sum': resultString};
           let sum = resultData['sum'];
           console.log('Sum of array from Python process =', sum);
      });
      res.end(`Sum is ${obj["child"]}`);
    });
  } else {
    res.end('Ok')
  }
});

//server.listen(3000);
/**
 * Routes Definitions
 */



/**
 * Slides Definitions
 */
const directoryPath = path.join(__dirname, 'public/images');

fs.readdir(directoryPath, function (err, files) {
    if (err) {
        return console.log('Unable to scan directory: ' + err);
    }
    files.forEach(function (file) {
        const stats = fs.statSync(directoryPath +'/'+ file);
        if (stats.isFile() === true) {
            ref_images.push('images/' + file);
        }
    });
    app.get("/", (req, res) => {
        res.render("index", { title: "Home", data: ref_images});
    });
});

/**
 * Server Activation
 */
//app.listen(port, () => {
//    console.log(`Listening to requests on http://localhost:${port}`);
//});
// Start the Proxy
app.listen(port, host, () => {
   console.log(`Starting Proxy at ${host}:${port}`);
});