
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");
const middle = document.getElementById("middle");
const canvas = document.getElementById("canvas");
const output = document.getElementById('output')
let context, RAF_timerID, flip_canvas, canvas_camera, facemesh_drawn, cw, ch;
context = canvas.getContext("2d");
ctx = output.getContext("2d");
canvas.width = middle.offsetWidth ;
canvas.height = middle.offsetWidth;
output.width = middle.offsetWidth ;
output.height = middle.offsetWidth;
let cam_landmarks = [], cam_box = [], btns=[], faces_arr=[], all_btns = [], all_btns_indices = []
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe', // or 'tfjs'
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
}
let faces, container, detector, selected;

const border_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

const leftSlick = $("#left");
const rightSlick = $("#right")

if (leftSlick.length) {
    leftSlick.slick({
        dots: false,
        infinite: false,
        speed: 300,
        focusOnSelect: false,
        slidesToShow: 3.5,
        slidesToScroll: 2,
        vertical: true,
        // arrows: false,
        draggable: true,
        asNavFor: '.asnav1Class',
        prevArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_up.png'></image></button>",
        nextArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_down.png'></image></button>"

    });
    rightSlick.slick({
        dots: false,
        infinite: true,
        speed: 300,
        focusOnSelect: false,
        slidesToShow: 3.5,
        slidesToScroll: 2,
        vertical: true,
        // arrows: false,
        draggable: true,
        asNavFor: '.asnav2Class',
        prevArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_up.png'></image></button>",
        nextArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_down.png'></image></button>"
    });

    // On init
    $(".select_ch").each(function (index, el) {
        leftSlick.slick('slickAdd', "<div>" + el.innerHTML + "</div>");
    });
    $(".show_morph").each(function (index, el) {
        rightSlick.slick('slickAdd', "<div>" + el.innerHTML + "</div>");
    });

};

function path_adjusted(url) {
    if (!/^\w+\:/i.test(url)) {
        url = url.replace(/^(\.?\/?)([\w\@])/, "$1js/$2")
    }
    return url
}

async function init() {
    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    fetch(path_adjusted('../TRIANGULATION.json')).then(response => response.json()).then(data => {TRIANGULATION=data});
    container = document.querySelector("#left");
    btns = container.querySelectorAll("div.select_ch.slick-slide:not(.slick-cloned) >button");
    all_btns = container.querySelectorAll("div.slick-slide >button");
    const prop = 'data-slick-index'

    all_btns.forEach(function (item, index, arr){
        all_btns_indices.push(arr[index].parentNode.getAttribute(prop))
    })

    const img_length = btns.length
    for (let imm_num= 0; imm_num<img_length; imm_num++){
        // console.log(imm_num + '  ' + btns[imm_num].firstChild.src )

        let f = new Face(imm_num, btns[imm_num].firstChild);
        const values = await calc_lmrks(f.image, f.which)
        f.lmrks = values[0]
        f.points = values[1];
        f.n_points = f.normalize_array(values[1])
        f.bb = values[2];
        f.angles = values[3];
        f.hull = convex_hull(values[1], values[2], f.which);
        // console.log(f)
        faces_arr.push(f)
    }
    for (let j = 0; j<all_btns.length; j++) {
        all_btns[j].onclick = function() {
            let _this = this;
            slide_id = all_btns_indices[j]
            // console.log(_this.firstChild + ' ' + j )
            faces_arr.forEach(function (item, index, arr){
                // console.log('index')

                if (_this.firstChild.src === arr[index].src){
                    // console.log(index+' '+j)
                    selected = index
                    ref_img.src = arr[index].src;

                    ref_img.onload = function(){
                        // console.log(item.lmrks)
                        drawHull(ref_img, selected)
                        // draw_landmarks(ref_img.width, ref_img.height,canvas,context,item.lmrks)
                    }
                }
                else {
                    // console.log('btn' + this + ' not matching')
                }
            })
        }
    }
    // for (let face in faces_arr){
    //     console.log(faces_arr[face].which)
    //     await faces_arr[face].landmarks();
    //     console.log(faces_arr[face])
    // }
}
function Face(which, image) {
    this.which = which;
    this.image = image;
    this.src = image.src;
    this.w = image.naturalWidth;
    this.h = image.naturalHeight;
    this.lmrks = [];
    this.points = [];
    this.bb = [];
    this.hull = [];
    this.n_points = []
    this.normalize_array= function (pix_points) {
        let n_array = [...pix_points];
        for (let p in n_array){
            n_array[p] = [Math.round(n_array[p][0])/this.w, Math.round(n_array[p][1])/this.h]
        }
        return n_array
    }

}
function drawHull(image, idx){
    const w = canvas.width
    const h = canvas.width
    const this_picture = faces_arr[idx]
    // console.log(idx +' '+ this_picture.which )
    // console.log(this_picture.n_points )

    const points_indices = this_picture.hull;
    let hull_points = [];
    for (let el in points_indices){
        let id = points_indices[el];
        let new_point = new cv.Point(this_picture.n_points[id][0]*w,this_picture.n_points[id][1]*h)
        // console.log(new_point)
        hull_points.push(new_point)
    }
    const mat = cv.imread(image)
    let larger_mat = new cv.Mat()
    const dsize = new cv.Size(w, h)
    cv.resize(mat, larger_mat, dsize, 0, 0, cv.INTER_LINEAR);
    // const mat2 = cv.imread(this_picture.src)
    // console.log(hull_points)
    for (let lin = 0;lin < hull_points.length-1; lin++){
        let p1  = hull_points[lin];
        let p2  = hull_points[lin+1];
        cv.line(larger_mat, p1, p2, [0, 255, 0, 255], 1)
    }

    cv.imshow(canvas, larger_mat);
    // cv.imshow(output, mat2)
    mat.delete()
    larger_mat.delete()
}
function calcHull(points, which) {
    if (points.length <= 1)
        return points.slice();
    // Andrew's monotone chain algorithm. Positive y coordinates correspond to "up"
    // as per the mathematical convention, instead of "down" as per the computer
    // graphics convention. This doesn't affect the correctness of the result.
    let upperHull = [];
    for (let i = 0; i < points.length; i++) {
        let p = points[i];
        while (upperHull.length >= 2) {
            let q = upperHull[upperHull.length - 1];
            let r = upperHull[upperHull.length - 2];
            if ((q[0] - r[0]) * (p[1] - r[1]) >= (q[1] - r[1]) * (p[0] - r[0]))
                upperHull.pop();
            else
                break;
        }
        upperHull.push(p);
    }
    upperHull.pop();
    let lowerHull = [];
    for (let i = points.length - 1; i >= 0; i--) {
        let p = points[i];
        while (lowerHull.length >= 2) {
            let q = lowerHull[lowerHull.length - 1];
            let r = lowerHull[lowerHull.length - 2];
            if ((q[0] - r[0]) * (p[1] - r[1]) >= (q[1] - r[1]) * (p[0] - r[0]))
                lowerHull.pop();
            else
                break;
        }
        lowerHull.push(p);
    }
    lowerHull.pop();
    if (upperHull.length === 1 && lowerHull.length === 1 && upperHull[0].x === lowerHull[0].x && upperHull[0].y === lowerHull[0].y) {
        return upperHull;
    }
    else {
        return upperHull.concat(lowerHull);
    }

}
function POINT_COMPARATOR(a, b) {
    // console.log(a +' ' +b)
    if (a[0] < b[0])
        return -1;
    else if (a[0] > b[0])
        return +1;
    else if (a[1] < b[1])
        return -1;
    else if (a[1] > b[1])
        return +1;
    else
        return 0;
}
function convex_hull(points, boundingbox, which){
    // console.log('GIRO')
    // console.log('calcolo convex' + which)
    let p1, p2, p3, p4, new_arr

    const min_x = boundingbox.xMin
    const max_x = boundingbox.xMax
    const min_y = boundingbox.yMin
    const max_y = boundingbox.yMax
    for (let pt in points){
        // console.log(points[pt])
        if (points[pt][0]===min_x){ p1 = points[pt] }
        else if (points[pt][0]===max_x){ p2 = points[pt] }

        else if (points[pt][1]===min_y){ p3 = points[pt] }

        else if (points[pt][1]===max_y){ p4 = points[pt] }
    }
    let poly = [p1, p2, p3, p4]
    new_arr = [...points]

    let _new_arr = [...new_arr]
    for (let pt in new_arr){
        if (pointInPolygon(poly, new_arr[pt]) === true){
            _new_arr = arrayRemove(_new_arr, new_arr[pt])
        }
    }
    for (let el in poly){
        if (poly[el] in _new_arr === false){_new_arr.push(poly[el])}
    }
    _new_arr.sort(POINT_COMPARATOR);
    let hull = calcHull(_new_arr, which)

    // console.log(hull)
    let _hull = [];
    for (let h in hull){
        for (let p in points){
            if (hull[h] === points[p]){
                _hull.push(p)
            }
        }
    }
    return _hull

}
function arrayRemove(arr, value) {
    return arr.filter(function(ele){
        return ele !== value;
    });
}

function pointInPolygon(polygon, point) {
    //A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    let odd = false;
    //For each edge (In this case for each point of the polygon and the previous one)
    for (let i = 0, j = polygon.length - 1; i < polygon.length; i++) {
        //If a line from the point into infinity crosses this edge
        if (((polygon[i][1] > point[1]) !== (polygon[j][1] > point[1])) // One point needs to be above, one below our y coordinate
            // ...and the edge doesn't cross our Y coordinate before our x coordinate (but between our x coordinate and infinity)
            && (point[0] < ((polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]))) {
            // Invert odd
            odd = !odd;
        }
        j = i;
    }
    //If the number of crossings was odd, the point is in the polygon
    return odd;
}

async function getHeadAngles(keypoints) {
// V: 10, 152; H: 226, 446
    this.faceVerticalCentralPoint = [
        0,
        (keypoints[10][1] + keypoints[152][1]) * 0.5,
        (keypoints[10][2] + keypoints[152][2]) * 0.5,
    ];
    const verticalAdjacent =
        keypoints[10][2] - this.faceVerticalCentralPoint[2];
    const verticalOpposite =
        keypoints[10][1] - this.faceVerticalCentralPoint[1];
    const verticalHypotenuse = this.l2Norm([
        verticalAdjacent,
        verticalOpposite,
    ]);
    const verticalCos = verticalAdjacent / verticalHypotenuse;

    this.faceHorizontalCentralPoint = [
        (keypoints[226][0] + keypoints[446][0]) * 0.5,
        0,
        (keypoints[226][2] + keypoints[446][2]) * 0.5,
    ];
    const horizontalAdjacent =
        keypoints[226][2] - this.faceHorizontalCentralPoint[2];
    const horizontalOpposite =
        keypoints[226][0] - this.faceHorizontalCentralPoint[0];
    const horizontalHypotenuse = this.l2Norm([
        horizontalAdjacent,
        horizontalOpposite,
    ]);
    const horizontalCos = horizontalAdjacent / horizontalHypotenuse;
    const first = Math.acos(verticalCos) * (180 / Math.PI);
    const second = Math.acos(horizontalCos)  * (180 / Math.PI);
    let pleft = keypoints[133]
    let pright = keypoints[362]
    let third = Math.atan2(pright[1]-pleft[1], pright[0]-pleft[0])*(180/Math.PI)

    // const third = verticalHypotenuse * horizontalHypotenuse;
    // console.log(first + ' ' + second + ' ' + verticalHypotenuse);

    return [
        first,
        second,
        third,
    ];
}

function l2Norm(vec) {
    let norm = 0;
    for (const v of vec) {
        norm += v * v;
    }
    return Math.sqrt(norm);
}

async function calc_lmrks(image, which) {
    // console.log('calcolo FACE '+ which)
    let landmarks = []
    let points_2d = []
    let bb, angles
    let faces = await detector.estimateFaces(image);
    if (faces.length>>0){
        // console.log(which)
        const keypoints = faces[0].keypoints;
        for (let land=0; land< keypoints.length; land++){
            let x = keypoints[land].x;
            let y = keypoints[land].y;
            let z = keypoints[land].z;
            landmarks.push([x, y, z])
            points_2d.push([x,y])
        }
        bb = faces[0].box
        if (landmarks !== []) {
            // console.log('not none')
            angles = await getHeadAngles(landmarks)
        }
    }
    else{
        console.log('non ce la faccio a calcolare le faces dell\'imm ' + which)
        // console.log(this.src)
    }
    return [landmarks, points_2d, bb, angles]
}

function draw(faces, w, h) {
    if (canvas) {
        if (RAF_timerID)
            cancelAnimationFrame(RAF_timerID)
        RAF_timerID = requestAnimationFrame(function () {
            RAF_timerID = null
            draw_facemesh(faces, w, h);
        });
    }
}
function draw_facemesh(faces, w,h, rgba) {
    function distance(a,b) {
        return Math.sqrt(Math.pow(a[0]-b[0],2) + Math.pow(a[1]-b[1],2))
    }

    if ((canvas.width != w) || (canvas.height != h)) {
        canvas.width  = w
        canvas.height = h
    }

    context.globalAlpha = 0.5

    context.save()

    context.clearRect(0,0,w,h)

    context.fillStyle = 'black'
    context.fillRect(0,0,w,h)

    if (!TRIANGULATION || !faces.length) {
        facemesh_drawn = false
        context.restore()
        return
    }

    facemesh_drawn = true

    cam_landmarks = []
    context.fillStyle = '#32EEDB';
    context.strokeStyle = '#32EEDB';
    context.lineWidth = 0.5;
    const keypoints = faces[0].keypoints;
    for (let land=0; land< keypoints.length; land++){
        let x = faces[0].keypoints[land].x;
        let y = faces[0].keypoints[land].y;
        let z = faces[0].keypoints[land].z;
        cam_landmarks.push([x, y, z])
        cam_box = faces[0].box

        // const points = TRIANGULATION[land];//.map(index => keypoints[index]);
        // drawPath(context, points, true);
    }
    for (let tris_indices in TRIANGULATION){
        const points_indices = [TRIANGULATION[tris_indices][0],TRIANGULATION[tris_indices][1],
            TRIANGULATION[tris_indices][2]]
        const points = [cam_landmarks[points_indices[0]], cam_landmarks[points_indices[1]],
            cam_landmarks[points_indices[2]]]
        drawPath(context, points, true);
    }

}
function draw_landmarks( w, h,canvas_where,ctx,land_array){
    // if ((canvas_where.width != w) || (canvas_where.height != h)) {
    //     canvas_where.width  = w
    //     canvas_where.height = h
    // }

    ctx.globalAlpha = 0.5

    ctx.save()

    ctx.clearRect(0,0,w,h)

    ctx.fillStyle = 'black'
    ctx.fillRect(0,0,w,h)

    if (!TRIANGULATION ) { //|| !faces.length
        facemesh_drawn = false
        ctx.restore()
        return
    }
    facemesh_drawn = true
    ctx.fillStyle = '#32EEDB';
    ctx.strokeStyle = '#32EEDB';
    ctx.lineWidth = 0.5;
    for (let tris_indices in TRIANGULATION){
        const points_indices = [TRIANGULATION[tris_indices][0],TRIANGULATION[tris_indices][1],
            TRIANGULATION[tris_indices][2]]
        const points = [land_array[points_indices[0]], land_array[points_indices[1]],
            land_array[points_indices[2]]]
        drawPath(ctx, points, true);
    }
}
function drawPath(ctx, points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < 3; i++) {
        const point = points[i];
        region.lineTo(point[0], point[1]);
    }

    if (closePath) {
        region.closePath();
    }
    ctx.stroke(region);
}
function cv_draw(webcam, bb, lmrks){
    // console.log(bb.xMin+ ' ' + bb.yMin+ ' ' + bb.width+ ' ' + bb.height)
    // let rect1,rect2,rect3,rect4
    // let rect

    output.width =  middle.offsetWidth ;
    output.height = middle.offsetWidth ;
    console.log(webcam.width+ ' ' + webcam.height+ ' ' + webcam.offsetWidth+ ' ' + webcam.offsetHeight)
    ctx.drawImage(webcam,0,0);
    let data = ctx.getImageData(0,0,webcam.offsetWidth,webcam.offsetHeight)
    // let src = cv.imread(output)
    let src2 = new cv.Mat(video.height, video.width,  cv.CV_8UC4)//cv.matFromImageData(data);
    let cap = new cv.VideoCapture(webcam);
    cap.read(src2)
    // let dst = new cv.Mat();
    // let gray = new cv.Mat();

    // rect1=((webcam.width-(bb.xMin + bb.width))/src2.cols)*webcam.offsetWidth;
    // rect2=(bb.yMin/src2.rows)*webcam.offsetHeight
    // rect3=(bb.width/src2.cols)*webcam.offsetWidth;
    // rect4=(bb.height/src2.rows)*webcam.offsetHeight

    // if (bb){
    //     console.log('yes')
    //     rect= new cv.Rect(rect1,rect2,rect3,rect4)
    // }
    // else{
    //     console.log('no')
    //     rect = new cv.Rect(0, 0, 50, 50)
    // }
    // let nuevo = cv.flip(src2,dst, 1)


    // src2=data
    // console.log(dst.cols)
    // dst = src2;
    let p1, p2, p3, p4, new_bb_xMin, new_bb_yMin, new_bb_xMax, new_bb_yMax

    // let bbpoints = [p1,p2,p3,p4]
    let passage = src2//new cv.Mat()

    p1=new cv.Point(bb.xMin*output.offsetHeight/webcam.height, bb.yMin*output.offsetHeight/webcam.height)
    p2=new cv.Point(bb.xMin*output.offsetHeight/webcam.height, bb.yMax*output.offsetHeight/webcam.height)
    p3=new cv.Point(bb.xMax*output.offsetHeight/webcam.height, bb.yMax*output.offsetHeight/webcam.height)
    p4=new cv.Point(bb.xMax*output.offsetHeight/webcam.height, bb.yMin*output.offsetHeight/webcam.height)
    const dsize = new cv.Size(webcam.offsetWidth, webcam.offsetHeight)
    cv.resize(passage, passage, dsize, 0, 0, cv.INTER_LINEAR);
    cv.line(passage, p1, p2, [255, 0, 0, 255], 1)
    cv.line(passage, p2, p3, [255, 0, 0, 255], 1)
    cv.line(passage, p3, p4, [255, 0, 0, 255], 1)
    cv.line(passage, p4, p1, [255, 0, 0, 255], 1)

    // dst = src2.roi(rect);
    console.log('image width: ' + src2.cols + '\n' +
        'image height: ' + output.width + '\n' +
        'image size: ' + video.width + '*' + video.offsetWidth + '\n' +
        'image depth: ' + webcam.width + '\n' +
        'image channels ' + video.clientWidth + '\n' +
        'image type: ' + src2.type() + '\n');

    // let crop=new cv.Mat(webcam.height, webcam.width, cv.CV_8UC1);
    // let dsize = new cv.Size(48, 48);
    // cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.imshow(output, passage);

}
const accessCamera = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: {width: 500, height: 400},
            audio: false,
        })
        .then((stream) => {
            video.srcObject = stream;

        })
        .catch(function (err0r) {
            console.log("Something went wrong!");
        });
};
function ImageItem(src) {
    this.image = new Image();
    this.src = src
}
function match(angles_cam, angles_ref){
    // console.log(angles_cam)
    // console.log(angles_ref)
    const delta = 5;
    if (angles_cam[0]>= angles_ref[0] - delta && angles_cam[0]<= angles_ref[0] + delta){
        // console.log('matched 1')
    }
    if (angles_cam[1]>= angles_ref[1] - delta && angles_cam[1]<= angles_ref[1] + delta){
        // console.log('matched 2')
    }
    if (angles_cam[2]>= angles_ref[2] - delta && angles_cam[2]<= angles_ref[2] + delta){
        // console.log('matched 3')
    }


}

async  function detectFaces(){
    let lmrks = [];
    let points = []
    let bb, angles
    let frame_roi;
    const cam_promise = calc_lmrks(video, 'cam')
    cam_promise
        .then((value) => {
            // console.log('value');
            // console.log(value);
            lmrks = value[0]//This is a fulfilled promise  👈
            points = value[1]
            bb = value[2]
            angles = value[3]
            // console.log(bb)
            if (selected) {
                match(angles, faces_arr[selected].angles)
            }
            cv_draw(video, bb, lmrks)
            // console.log(angles)
        })
        .catch((err) => {
            console.log(err);
        });
    // console.log('detectfaces ' )
    // console.log(btns_arr[selected])

    // draw(faces, video.width, video.height)
};
function resizeCanvas() {
    canvas.width =  middle.offsetWidth ;
    canvas.height = middle.offsetWidth ;
    output.width =  middle.offsetWidth ;
    output.height = middle.offsetWidth ;
    /**
     * Your drawings need to be inside this function otherwise they will be reset when
     * you resize the browser window and the canvas goes will be cleared.
     */
    if (selected){
        drawHull(ref_img, selected);
    }
}
init();
accessCamera();
//
// const cv = document.cv
// this event will be  executed when the video is loaded
// window.addEventListener('DOMContentLoaded', (event) => {
//     console.log('DOM fully loaded and parsed ' + ref_img.src);
//     // cv.rectangle(ref_img.src, (0,0), (30,30), (255,0,0),5 )
//
// });
window.addEventListener('resize', resizeCanvas, false);
video.addEventListener("loadeddata", async () => {
    setInterval(detectFaces, 1000
    );
});

////////////////////////////////////////

