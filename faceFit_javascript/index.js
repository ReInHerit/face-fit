/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
const fs = require('fs');
/**
 * App Variables
 */
const app = express();
const port = process.env.PORT || "8000";
let ref_images = [];
/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
/**
 * Routes Definitions
 */
// app.get("/", (req, res) => {
//
//     res.render("index", { title: "Home", data: "[0, 1]"});
// });

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
        if (stats.isFile() == true) {
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
app.listen(port, () => {
    console.log(`Listening to requests on http://localhost:${port}`);
});