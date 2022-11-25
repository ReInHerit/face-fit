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
let morphs_path = path.relative("public/js", "public/morphs");
console.log(morphs_path)
/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.json({limit: '50mb'}));
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
// Info GET endpoint
app.get('/info', (req, res) => {
    res.send({'start': 'swap'});
    res.on('message', obj => {
      console.log(obj)
      // const obj_str = JSON.stringify(obj)
    })
});
app.post('/init', function(req, res, next){ //chiedi il body da main e invialo a python
    const paintings = req.body
    console.log(paintings)
//    const txt = {'test': 456}
    request.post(
        'http://localhost:8050/INIT_PAINTINGS',
        { json: ref_images },
        async function (error, response, body) {
            if (!error && response.statusCode === 200) {
                console.log('init server',body);

            }
        res.send({
            'body': body});
//        console.log('server_path',morphed_path)
        });

//    if (morphed_path !== "") {next.send(morphed_path)}
})
app.post('/info', function(req, res, next){ //chiedi il body da main e invialo a python
    const objs = req.body
//    console.log(objs)
//    const txt = {'test': 456}


    request.post(
        'http://localhost:8050/DATAtoPY',
        { json: objs },
        async function (error, response, body) {
            if (!error && response.statusCode === 200) {
                console.log('server',body);
                abs_morphed_path = body
                file_name = path.parse(body).base
                rel_morphed_path = 'http://' + host +':'+port + '/morphs/' + file_name
            }
        res.send({
            'relative_path': rel_morphed_path,
            'absolute_path': abs_morphed_path,
            'file_name': file_name});
//        console.log('server_path',morphed_path)
        });

//    if (morphed_path !== "") {next.send(morphed_path)}
})

/**
 * Routes Definitions
 */




/**
 * Server Activation
 */
app.listen(port, host, () => {
   console.log(`Starting Proxy at ${host}:${port}`);
});