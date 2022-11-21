/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
const fs = require('fs');
// const spawn = require('child_process').spawn;
const bodyParser = require("body-parser");
// const fetch = (url) => import('node-fetch').then(({default: fetch}) => fetch(url));
const http = require('http');
// const { fork } = require('child_process');
// const postRoutes = require('./routes/post')
const request = require('request-promise')
/**
 * App Variables
 */
const app = express();
const port = process.env.PORT || 8000;
const host = "localhost";
let morphed_path = ''
let ref_images = [];
/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.json({limit: '50mb'}));
// Info GET endpoint
app.get('/info', (req, res) => {
    res.send({'start': 'swap'});
    res.on('message', obj => {
      console.log(obj)
      // const obj_str = JSON.stringify(obj)
    })
});

app.post('/info', function(req, res, next){ //chiedi il body da main e invialo a python
    const objs = req.body
//    console.log(objs)
//    const txt = {'test': 456}
//    res.send('response');

    request.post(
        'http://localhost:8050/DATAtoPY',
        { json: objs },
        function (error, response, body) {
            if (!error && response.statusCode === 200) {
                console.log('server',body);
                morphed_path = body

            }
        }
    );
    console.log('server_path',morphed_path)
    if (morphed_path !== "") {res.send(morphed_path)}
})




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