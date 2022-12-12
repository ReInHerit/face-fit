/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
const fs = require('fs');
const nodemailer = require('nodemailer');
const bodyParser = require("body-parser");
// const http = require('http');
const request = require('request-promise')
/**
 * App Variables
 */
const app = express();
const port = process.env.PORT || 8000;
const host = "localhost";

let ref_images = [];


const json_path = path.join(__dirname, 'public/json');
const directoryPath = path.join(__dirname, 'public/images');
let morphs_path = path.join(__dirname, 'public/morphs');
checkPath(morphs_path)

const painting_data_file = json_path + '/painting_data.json'
const gmail_data_file = json_path + '/password_gmail.json'
const jsonData = JSON.parse(fs.readFileSync(painting_data_file, 'utf8'));
const gmail = JSON.parse(fs.readFileSync(gmail_data_file, 'utf8'));
/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.json({limit: '50mb'}));

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: gmail['email'],
        pass: gmail['password']
        },
    tls: {rejectUnauthorized: false}
    });
/**
 * Paintings' Files
 */

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
        res.render("index", { title: "FACE-FIT", data: ref_images});
    });
});

// GET POSTS

function checkPath(path){
    if (!fs.existsSync(path)) {
        fs.mkdirSync(path, { recursive: true });
    }
}
function extract_number(fileName){
    const replaced = fileName.replace(/\D/g, ''); // 👉️ '123'
    let numb;
    if (replaced !== '') {
        numb = Number(replaced); // 👉️ 123
    }
    return numb
}
async function send_mail(send_to){
    const files = await fs.promises.readdir(morphs_path);
    const morph_list = files.map(file => ({filename: file, path: `${morphs_path}/${file}`}));
    let content = 'Hello,<br>Face-Fit App here. These are the results of your matches.<br>The characters in which you impersonated yourself are:<br>';
    morph_list.forEach(morph => {
        let numb = extract_number(morph['filename']);
        if (numb<=9){
            numb = '0'+ numb;
        }
        const description = jsonData[`image${numb}.jpg`].description;
        content += `<li>${description}</li>`;
    });

    transporter.sendMail({
        from: gmail['email'],
        to: send_to,
        subject: 'Your Face-Fit Images !',
        text: content,
        html: content,
        attachments: morph_list
    }, (error, info) => {
        if (error) {
            console.log(error);
        } else {
            console.log(`Email sent: ${info.response}`);
            files.forEach(async file => {
            const filePath = `${morphs_path}/${file}`;
            await fs.promises.unlink(filePath);
            });
        }
    });
}
/**
 * Routes Definitions
 */
app.get('/info', (req, res) => {
    res.send({'start': 'swap'});
    res.on('message', obj => {
      console.log(obj)
    })
});

app.post('/morphs_to_send', function(req, res){
    const user_input = req.body
    console.log(user_input['mail'])
    send_mail(user_input['mail']).then(r =>{
        res.send({'answer': 'sent'})
    })

})

app.post('/init', function(req, res){ //chiedi il body da main e invialo a python
    const paintings = req.body
    request.post('http://localhost:8050/INIT_PAINTINGS', {json: ref_images}, async function (error, response, body) {
            if (!error && response.statusCode === 200) {
                console.log('init server',body);
            }
        res.send({'body': body});
        });
})
app.post('/info', function(req, res, next){
    const objs = req.body
    request.post('http://localhost:8050/DATAtoPY',{json: objs}, async function (error, response, body) {
        let abs_morphed_path, file_name, rel_morphed_path;
        if (!error && response.statusCode === 200) {
            abs_morphed_path = body
            file_name = path.parse(body).base
            rel_morphed_path = 'http://' + host + ':' + port + '/morphs/' + file_name
        }
        res.send({
            'relative_path': rel_morphed_path,
            'absolute_path': abs_morphed_path,
            'file_name': file_name
        });
    });
})

/**
 * Server Activation
 */
app.listen(port, host, () => {
   console.log(`Starting Proxy at ${host}:${port}`);
});
