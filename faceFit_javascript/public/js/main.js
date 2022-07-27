
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");
const middle = document.getElementById("middle");
const canvas = document.getElementById("canvas");
let context, RAF_timerID, facemesh_drawn, cam_face, faces, container, detector, selected;
let cam_landmarks = [], cam_box = [], btns=[], faces_arr=[], all_btns = [], all_btns_indices = []
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe', // or 'tfjs'
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
}

function canvas_size(value){
    if (value <= 600) {
        canvas.width = value;
        canvas.height = value;
    }
    else {
        canvas.width = 600;
        canvas.height = 600;
    }
}
canvas_size(middle.offsetWidth)
context = canvas.getContext("2d");
// const border_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
//     152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
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
}

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
        let f = new Face(imm_num, btns[imm_num].firstChild);
        const values = await calc_lmrks(f.image, f.which)
        f.lmrks = values[0]
        f.points = values[1];
        f.n_points = f.normalize_array(values[1])
        f.bb = values[2];
        f.angles = values[3];
        f.hull = convex_hull(values[1], values[2]);
        faces_arr.push(f)
    }
    for (let j = 0; j<all_btns.length; j++) {
        all_btns[j].onclick = function() {
            let _this = this;
            slide_id = all_btns_indices[j]
            faces_arr.forEach(function (item, index, arr){
                if (_this.firstChild.src === arr[index].src){
                    selected = index
                    ref_img.src = arr[index].src;

                    ref_img.onload = function(){
                        // drawHull(ref_img, selected)
                        // draw_landmarks(ref_img.width, ref_img.height,canvas,context,item.lmrks)
                    }
                }
                else {
                    // console.log('btn' + this + ' not matching')
                }
            })
        }
    }
}
function Face(which, image) {
    this.which = which;
    this.image = image;
    this.src = image.src;
    this.w = image.naturalWidth;
    this.h = image.naturalHeight;
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

function calcHull(points) {
    if (points.length <= 1)
        return points.slice();
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
function convex_hull(points, boundingbox){
    let p1, p2, p3, p4, new_arr
    const min_x = boundingbox.xMin
    const max_x = boundingbox.xMax
    const min_y = boundingbox.yMin
    const max_y = boundingbox.yMax
    for (let pt in points){
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
        if (!(poly[el] in _new_arr)){_new_arr.push(poly[el])}
    }
    _new_arr.sort(POINT_COMPARATOR);
    let hull = calcHull(_new_arr)
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
    return [first, second, third];
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
    let bb ={}
    let angles
    let faces = await detector.estimateFaces(image);
    if (faces.length>>0){
        // console.log(which)
        const keypoints = faces[0].keypoints;
        for (let land=0; land< keypoints.length; land++){
            let x = Math.round(keypoints[land].x);
            let y = Math.round(keypoints[land].y);
            let z = Math.round(keypoints[land].z);
            landmarks.push([x, y, z])
            points_2d.push([x,y])
        }
        bb.xMin = Math.round(faces[0].box.xMin)
        bb.xMax = Math.round(faces[0].box.xMax)
        bb.yMin = Math.round(faces[0].box.yMin)
        bb.yMax = Math.round(faces[0].box.yMax)
        bb.width = Math.round(faces[0].box.width)
        bb.height = Math.round(faces[0].box.height)
        bb.center = [bb.xMin + Math.round(bb.width/2),bb.yMin + Math.round(bb.height/2)]


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

function distance(a,b) {
    return Math.sqrt(Math.pow(a[0]-b[0],2) + Math.pow(a[1]-b[1],2))
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
function drawHull(image, idx){
    const w = canvas.width
    const h = canvas.width
    const this_picture = faces_arr[idx]
    const points_indices = this_picture.hull;
    let hull_points = [];
    for (let el in points_indices){
        let id = points_indices[el];
        let new_point = new cv.Point(this_picture.n_points[id][0]*w,this_picture.n_points[id][1]*h)
        hull_points.push(new_point)
    }
    const mat = cv.imread(image)
    let larger_mat = new cv.Mat()
    const dsize = new cv.Size(w, h)
    cv.resize(mat, larger_mat, dsize, 0, 0, cv.INTER_LINEAR);
    for (let lin = 0;lin < hull_points.length-1; lin++){
        let p1  = hull_points[lin];
        let p2  = hull_points[lin+1];
        cv.line(larger_mat, p1, p2, [0, 255, 0, 255], 1)
    }
    cv.imshow(canvas, larger_mat);
    mat.delete()
    larger_mat.delete()
}
function cv_draw(webcam, bb){
    // console.log(bb.xMin+ ' ' + bb.yMin+ ' ' + bb.width+ ' ' + bb.height)
    // let rect1,rect2,rect3,rect4
    // let rect
    let rect, dst
    // output.width =  middle.offsetWidth ;
    // output.height = middle.offsetWidth ;

    let src2 = new cv.Mat(webcam.height, webcam.width,  cv.CV_8UC4)
    // let ref = new cv.Mat(ref_img.offsetWidth,ref_img.offsetHeight)
    rect = new cv.Rect(bb.xMin,bb.yMin,bb.width,bb.height)
    // if (bb){
    //     console.log('yes')
    //     rect= new cv.Rect(rect1,rect2,rect3,rect4)
    // }
    // else{
    //     console.log('no')
    //     rect = new cv.Rect(0, 0, 50, 50)
    // }
    // let nuevo = cv.flip(src2,dst, 1)
    // let p01, p02, p03, p04, p05, p06, p07, p08
    // p01 =new cv.Point(bb.xMin, bb.yMin)
    // p02 =new cv.Point(bb.xMin,bb.yMax)
    // p03 =new cv.Point(bb.xMax,bb.yMax)
    // p04 =new cv.Point(bb.xMax,bb.yMin)
    // p05=new cv.Point(webcam.width, webcam.height)
    // p06=new cv.Point(webcam.width,0 )
    // p07=new cv.Point(0, 0)
    // p08=new cv.Point(0, webcam.height)
    // src2=data
    // console.log(dst.cols)
    // dst = src2;

    let cap = new cv.VideoCapture(webcam);
    cap.read(src2)
    dst= src2.roi(rect)
    // cv.line(passage, p01, p02, [255, 0, 0, 255], 1)
    // cv.line(passage, p05, p06, [255, 0, 0, 255], 2)
    // cv.line(passage, p06, p07, [255, 255, 0, 255], 2)
    // cv.line(passage, p07, p08, [0, 255, 0, 255], 2)
    // cv.line(passage, p08, p05, [0, 255, 255, 255], 2)
    // passage = passage.roi(rect)

    // const dsize = new cv.Size(webcam.offsetWidth, webcam.offsetHeight)
    // cv.resize(passage, passage, dsize, 0, 0, cv.INTER_LINEAR);
    // let p1, p2, p3, p4, p5,p6,p7,p8,new_bb_xMin, new_bb_yMin, new_bb_xMax, new_bb_yMax
    // p1=new cv.Point(bb.xMin*webcam.offsetHeight/webcam.height, bb.yMin*webcam.offsetWidth/webcam.width)
    // p2=new cv.Point(bb.xMin*webcam.offsetHeight/webcam.height, bb.yMax*webcam.offsetWidth/webcam.width)
    // p3=new cv.Point(bb.xMax*webcam.offsetHeight/webcam.height, bb.yMax*webcam.offsetWidth/webcam.width)
    // p4=new cv.Point(bb.xMax*webcam.offsetHeight/webcam.height, bb.yMin*webcam.offsetWidth/webcam.width)
    //
    //
    //
    // cv.line(passage, p1, p2, [255, 0, 0, 255], 1)
    // cv.line(passage, p2, p3, [255, 0, 0, 255], 1)
    // cv.line(passage, p3, p4, [255, 0, 0, 255], 1)
    // cv.line(passage, p4, p1, [255, 0, 0, 255], 1)


    // console.log(cap)
    // let crop=new cv.Mat(webcam.height, webcam.width, cv.CV_8UC1);
    // let dsize = new cv.Size(48, 48);
    // cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
    // cv.imshow(output, dst);
    src.delete();
    dst.delete();

}
function draw_mask_on_ref(webcam, cam_obj){
    const ref_bb = faces_arr[selected].bb
    let bb_cam_rect, cam_roi, bb_ref_rect, ref_roi, cam_mask, dst
    let maskInv = new cv.Mat();
    let mask = new  cv.Mat()
    let new_mask = new  cv.Mat()
    let color_new_mask = new cv.Mat()
    let sum = new cv.Mat();
    let cam_source = new cv.Mat(webcam.height, webcam.width,  cv.CV_8UC4)
    let cam_mask_inv = new cv.Mat()
    let cam_roi_gray = new cv.Mat()
    let cam_laplacian = new cv.Mat()
    let cam_sobel = new cv.Mat()
    let ref = cv.imread(ref_img)
    const points_indices = cam_obj.hull;
    let cap = new cv.VideoCapture(webcam);

    cap.read(cam_source)

    bb_cam_rect = new cv.Rect(cam_obj.bb.xMin,cam_obj.bb.yMin,cam_obj.bb.width,cam_obj.bb.height)
    bb_ref_rect = new cv.Rect(ref_bb.xMin,ref_bb.yMin,ref_bb.width,ref_bb.height);

    ///// calcolo convex hull mak
    let hull_points = [];
    for (let el in points_indices){
        let id = points_indices[el];
        let new_point = [cam_obj.n_points[id][0]*cam_source.cols,cam_obj.n_points[id][1]*cam_source.rows]
        hull_points.push(new_point)
    }
    let convexHullMat = cv.Mat.zeros(cam_obj.w, cam_obj.h, cv.CV_8UC3);
    let hull = nestedPointsArrayToMat(hull_points);
    // make a fake hulls vector
    let hulls = new cv.MatVector();
    // add the recently created hull
    hulls.push_back(hull);
    // test drawing it
    cv.drawContours(convexHullMat, hulls, 0, [255,255,255,0], -1, 8);

    // ROIs
    cam_mask = convexHullMat.roi(bb_cam_rect)
    cam_roi= cam_source.roi(bb_cam_rect)
    ref_roi = ref.roi(bb_ref_rect);

    const dsize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const dsize_back = new cv.Size(ref_roi.cols, ref_roi.rows)

    cv.cvtColor(cam_mask,cam_mask, cv.COLOR_RGBA2GRAY, 0)
    cv.bitwise_not(cam_mask, cam_mask_inv);

    cv.GaussianBlur(cam_roi, cam_roi, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT)
    cv.cvtColor(cam_roi,cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)

    cam_laplacian = laplacian(cam_roi, cam_roi.cols, cam_roi.rows)

    cv.Sobel(cam_roi_gray, cam_sobel, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)

    cv.addWeighted(cam_sobel, .75, cam_laplacian, .25, 0, mask);
    cv.bitwise_and(mask, mask, new_mask, cam_mask)
    cv.flip(new_mask, new_mask,1)
    cv.bitwise_not(new_mask, maskInv);

    cv.resize(ref_roi, ref_roi, dsize, 0, 0, cv.INTER_LINEAR )

    cv.cvtColor(new_mask, color_new_mask, cv.COLOR_GRAY2RGB, 0)
    let pass = color_new_mask.clone();
    cv.bilateralFilter(pass, color_new_mask, 5, 75, 75, cv.BORDER_DEFAULT);
    cv.cvtColor(color_new_mask, color_new_mask, cv.COLOR_RGB2RGBA, 0)
    cv.add(ref_roi,color_new_mask,sum)
    cv.add(sum,color_new_mask,sum)
    cv.resize(sum, sum, dsize_back, 0, 0, cv.INTER_LINEAR )
    cv.resize(cam_roi, cam_roi, dsize_back, 0, 0, cv.INTER_LINEAR )

    dst = ref.clone();
    for (let i = 0; i < cam_roi.rows; i++) {
        for (let j = 0; j < cam_roi.cols; j++) {
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[0] = sum.ucharPtr(i, j)[0];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[1] = sum.ucharPtr(i, j)[1];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[2] = sum.ucharPtr(i, j)[2]
        }
    }
    canvas_size(middle.offsetWidth)
    let last_size =  new cv.Size(canvas.width, canvas.height)
    cv.resize(dst, dst, last_size, 0, 0, cv.INTER_LINEAR )
    cv.imshow(canvas, dst);
    ref.delete(); dst.delete(); cam_roi.delete(); ref_roi.delete(); mask.delete();
    maskInv.delete(); sum.delete(); hull.delete(); convexHullMat.delete()
}
function nestedPointsArrayToMat(points){
    return cv.matFromArray(points.length, 1, cv.CV_32SC2, points.flat());
}
function mask_img(masked, img, mask, mask_inv){
    let imgBg = new cv.Mat(masked.cols,masked.rows,  cv.CV_8UC4);
    let imgFg = new cv.Mat(masked.cols,masked.rows,  cv.CV_8UC4);
    let sum = new cv.Mat();
        // Black-out the area of logo in ROI
    cv.bitwise_and(masked, masked, imgBg, mask_inv);

    // Take only region of logo from logo image
    cv.bitwise_and(img, img, imgFg, mask_inv);

    // Put logo in ROI and modify the main image
    cv.add(imgBg, imgFg, sum);
    imgBg.delete(); imgFg.delete()
    return sum
}
function laplacian(src, width, height) {
    const dstC1 = new cv.Mat(height, width, cv.CV_8UC1)
    const mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, 5, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}
const accessCamera = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: {width: 640, height: 480},
            audio: false,
        })
        .then((stream) => {
            video.srcObject = stream;

        })
        .catch(function (error) {
            console.log("Something went wrong!");
        });
};

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
    cam_face = new Face('cam', video)
    cam_face.w=video.width;
    cam_face.h=video.height;

    const cam_promise = calc_lmrks(video, 'cam')
    cam_promise
        .then((value) => {
            // console.log('value');
            // console.log(value);
            cam_face.lmrks = value[0];//This is a fulfilled promise  👈
            cam_face.points = value[1];
            cam_face.bb = value[2];
            cam_face.angles = value[3];
            cam_face.hull = convex_hull(value[1], value[2]);
            cam_face.n_points = cam_face.normalize_array(value[1])
            
            // console.log(bb)
            if (selected>=0) {
                draw_mask_on_ref(video, cam_face)
                match(cam_face.angles, faces_arr[selected].angles)
            }
            // cv_draw(video, video_bb)

            // console.log(angles)
        })
        .catch((err) => {
            console.log(err);
        });
}

function resizeCanvas() {
    canvas_size(middle.offsetWidth)
    // output.width =  middle.offsetWidth ;
    // output.height = middle.offsetWidth ;
    /**
     * Your drawings need to be inside this function otherwise they will be reset when
     * you resize the browser window and the canvas goes will be cleared.
     */
    if (selected>=0){
        draw_mask_on_ref(video, cam_face)
        // drawHull(ref_img, selected);
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
    setInterval(detectFaces, 100
    );
});

////////////////////////////////////////

